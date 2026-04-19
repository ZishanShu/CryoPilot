"""Topaz picking: contracts, preflight, executor, scheduler, report."""

from __future__ import annotations

import stat
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from agent.backend_registry import PICKING_BACKEND_METADATA, PICKING_INPUT_SPECS
from agent.picking_preflight import enrich_and_preflight_topaz_params, preflight_topaz_pick
from app.scheduler import Scheduler
from relion.relion_actions import RelionActionExecutor
from relion.topaz_runner import build_topaz_extract_argv, run_topaz_pick
from schemas.action import Action, ActionName
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState


class TestTopazInputContract(unittest.TestCase):
    def test_topaz_spec_in_registry(self) -> None:
        self.assertIn("topaz", PICKING_INPUT_SPECS)
        self.assertIn("topaz", PICKING_BACKEND_METADATA)
        spec = PICKING_INPUT_SPECS["topaz"]
        for f in ("micrographs_star_path", "particle_diameter_angstrom", "output_dir"):
            self.assertIn(f, spec["required_fields"])
        self.assertIn("topaz_executable", spec["optional_fields"])


class TestTopazPreflight(unittest.TestCase):
    def test_missing_micrographs_unavailable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            r = preflight_topaz_pick(
                {"particle_diameter_angstrom": 180, "output_dir": str(root / "out")},
                project_root=root,
                topaz_config_path=None,
            )
            self.assertFalse(r["available"])
            self.assertIn("micrographs_star_path", r["missing_inputs"])

    def test_missing_executable_unavailable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CtfFind").mkdir()
            star = root / "CtfFind" / "micrographs_ctf.star"
            star.write_text("data_\n", encoding="utf-8")
            outd = root / "AutoPick" / "topaz"
            outd.mkdir(parents=True)
            with patch("agent.picking_preflight.shutil.which", return_value=None):
                r = preflight_topaz_pick(
                    {
                        "micrographs_star_path": str(star),
                        "particle_diameter_angstrom": 180,
                        "output_dir": str(outd),
                    },
                    project_root=root,
                    topaz_config_path=None,
                )
            self.assertFalse(r["available"])
            self.assertTrue(any("topaz_executable" in x for x in r.get("missing_dependencies", [])))
            self.assertIn("missing_dependency", str(r.get("blocked_reason") or ""))

    def test_available_with_mock_executable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CtfFind").mkdir()
            star = root / "CtfFind" / "micrographs_ctf.star"
            star.write_text("data_\n", encoding="utf-8")
            outd = root / "AutoPick" / "topaz"
            outd.mkdir(parents=True)
            fake = root / "fake_topaz"
            fake.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
            fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
            r = preflight_topaz_pick(
                {
                    "micrographs_star_path": str(star),
                    "particle_diameter_angstrom": 180,
                    "output_dir": str(outd),
                    "topaz_executable": str(fake),
                },
                project_root=root,
                topaz_config_path=None,
            )
            self.assertTrue(r["available"])
            self.assertEqual(r["execution_mode"], "real")
            self.assertEqual(r["executable_path"], str(fake.resolve()))
            self.assertEqual(r["command_mode"], "extract")
            self.assertEqual(r["model_source"], "default_pretrained")

    def test_available_without_model_path_uses_default_pretrained(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CtfFind").mkdir()
            star = root / "CtfFind" / "micrographs_ctf.star"
            star.write_text("data_\nloop_\n_rlnMicrographName #1\nm1.mrc\n", encoding="utf-8")
            (root / "CtfFind" / "m1.mrc").write_text("img", encoding="utf-8")
            outd = root / "AutoPick" / "topaz"
            outd.mkdir(parents=True)
            fake = root / "fake_topaz"
            fake.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
            fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
            r = preflight_topaz_pick(
                {
                    "micrographs_star_path": str(star),
                    "particle_diameter_angstrom": 180,
                    "output_dir": str(outd),
                    "topaz_executable": str(fake),
                },
                project_root=root,
                topaz_config_path=None,
            )
            self.assertTrue(r["available"])
            self.assertEqual(r["model_source"], "default_pretrained")


class TestTopazCliBuild(unittest.TestCase):
    def test_extract_argv_contains_required_flags(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            star = root / "micrographs.star"
            mic = root / "m1.mrc"
            mic.write_text("img", encoding="utf-8")
            star.write_text("data_\nloop_\n_rlnMicrographName #1\nm1.mrc\n", encoding="utf-8")
            argv, out_path, meta = build_topaz_extract_argv(
                {
                    "micrographs_star_path": str(star),
                    "output_dir": str(root / "out"),
                    "particle_diameter_angstrom": 200.0,
                    "topaz_model_path": "/tmp/model.sav",
                    "topaz_threshold": -6.0,
                    "topaz_device": "0",
                },
                {},
                {"executable_path": "/usr/bin/topaz"},
            )
            self.assertEqual(argv[0:2], ["/usr/bin/topaz", "extract"])
            self.assertIn("-r", argv)
            self.assertIn("--format", argv)
            self.assertIn("star", argv)
            self.assertIn("-m", argv)
            self.assertEqual(meta["command_mode"], "extract")
            self.assertEqual(meta["model_source"], "explicit")
            self.assertTrue(out_path.endswith("picks.star"))

    def test_extract_argv_without_model_uses_default_pretrained_semantics(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            star = root / "micrographs.star"
            mic = root / "m1.mrc"
            mic.write_text("img", encoding="utf-8")
            star.write_text("data_\nloop_\n_rlnMicrographName #1\nm1.mrc\n", encoding="utf-8")
            argv, _out_path, meta = build_topaz_extract_argv(
                {
                    "micrographs_star_path": str(star),
                    "output_dir": str(root / "out"),
                    "particle_diameter_angstrom": 200.0,
                },
                {},
                {"executable_path": "/usr/bin/topaz"},
            )
            self.assertNotIn("-m", argv)
            self.assertEqual(meta["model_source"], "default_pretrained")


class TestTopazExecutor(unittest.TestCase):
    def test_preflight_blocked_skips_subprocess(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            schemes = root / "schemes.yaml"
            schemes.write_text("schemes: {}\n", encoding="utf-8")
            ex = RelionActionExecutor(schemes)
            action = Action(
                name=ActionName.RUN_AUTOPICK,
                reason="t",
                parameters={
                    "picking_backend": "topaz",
                    "picking_availability": "unavailable",
                    "picking_blocked_reason": "missing_dependency: topaz_executable",
                    "topaz_preflight": {"available": False, "executable_path": None},
                },
                priority=78,
            )
            with patch("subprocess.run") as m:
                out = ex.execute(action, project_root=root, dry_run=False, state=None)
            m.assert_not_called()
            self.assertEqual(out.get("status"), "preflight_blocked")

    def test_real_run_invokes_subprocess_and_creates_output(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CtfFind").mkdir()
            star = root / "CtfFind" / "micrographs_ctf.star"
            star.write_text("data_\nloop_\n_rlnMicrographName #1\nm1.mrc\n", encoding="utf-8")
            (root / "CtfFind" / "m1.mrc").write_text("img", encoding="utf-8")
            outd = root / "AutoPick" / "topaz"
            outd.mkdir(parents=True)
            fake = root / "topaz"
            fake.write_text(
                textwrap.dedent(
                    """#!/bin/sh
                    prev=""
                    for a in "$@"; do
                      if [ "$prev" = "-o" ]; then
                        mkdir -p "$(dirname "$a")"
                        echo coords > "$a"
                        exit 0
                      fi
                      prev="$a"
                    done
                    exit 1
                    """
                ).strip(),
                encoding="utf-8",
            )
            fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
            params = enrich_and_preflight_topaz_params(
                {
                    "picking_backend": "topaz",
                    "project_path": str(root),
                    "topaz_executable": str(fake),
                    "micrographs_star_path": str(star),
                    "particle_diameter_angstrom": 180.0,
                    "output_dir": str(outd),
                },
                project_root=root,
                topaz_config_path=None,
            )
            self.assertTrue(params["topaz_preflight"]["available"])
            pf = params["topaz_preflight"]
            er = run_topaz_pick(
                params=params,
                project_dir=root,
                topaz_config_path=None,
                preflight=pf,
                dry_run=False,
                timeout_sec=30,
            )
            self.assertEqual(er.status, "success")
            out_coords = er.metadata.get("topaz_output_path")
            assert out_coords
            self.assertTrue(Path(out_coords).is_file())
            self.assertIn("extract", er.command or "")

    def test_real_run_exit_zero_without_output_is_failed(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CtfFind").mkdir()
            star = root / "CtfFind" / "micrographs_ctf.star"
            star.write_text("data_\nloop_\n_rlnMicrographName #1\nm1.mrc\n", encoding="utf-8")
            (root / "CtfFind" / "m1.mrc").write_text("img", encoding="utf-8")
            outd = root / "AutoPick" / "topaz"
            outd.mkdir(parents=True)
            fake = root / "topaz"
            fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
            params = enrich_and_preflight_topaz_params(
                {
                    "picking_backend": "topaz",
                    "project_path": str(root),
                    "topaz_executable": str(fake),
                    "micrographs_star_path": str(star),
                    "particle_diameter_angstrom": 180.0,
                    "output_dir": str(outd),
                },
                project_root=root,
                topaz_config_path=None,
            )
            er = run_topaz_pick(
                params=params,
                project_dir=root,
                topaz_config_path=None,
                preflight=params["topaz_preflight"],
                dry_run=False,
                timeout_sec=30,
            )
            self.assertEqual(er.status, "failed")


class TestTopazSchedulerIntegration(unittest.TestCase):
    def test_enriched_topaz_params_include_preflight(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CtfFind").mkdir()
            (root / "CtfFind" / "micrographs_ctf.star").write_text("data_\n", encoding="utf-8")
            sched = Scheduler(project_root=root)
            state = SampleState(project_id="p1", project_path=str(root), stage="CtfFind", status="running")
            qc = QCMetrics(motioncorr_completed=True, ctf_completed=True)
            action = sched.policy_engine.decide_next_action(
                state, qc, log_history=False, preferred_picking_backend="topaz"
            )
            assert action is not None
            enriched = sched._enrich_autopick_action(state, action)
            self.assertEqual(enriched.parameters.get("picking_backend"), "topaz")
            self.assertIn("topaz_preflight", enriched.parameters)
            self.assertIn("picking_availability", enriched.parameters)


class TestTopazReportLines(unittest.TestCase):
    def test_markdown_shows_picking_preflight(self) -> None:
        from agent.reporter import Reporter

        state = SampleState(project_id="p1", project_path="/tmp/p", stage="CtfFind", status="running")
        qc = QCMetrics(motioncorr_completed=True, ctf_completed=True)
        next_action = Action(
            name=ActionName.RUN_AUTOPICK,
            reason="pick",
            parameters={
                "picking_backend": "topaz",
                "picking_availability": "available",
                "picking_execution_mode": "real",
                "topaz_preflight": {
                    "available": True,
                    "execution_mode": "real",
                    "executable_path": "/opt/topaz",
                    "blocked_reason": None,
                },
            },
            priority=78,
        )
        report = Reporter().build_report(
            state=state,
            qc=qc,
            recent_action=None,
            next_action=next_action,
            action_history=[],
            current_run_id=None,
        )
        md = report.to_markdown()
        self.assertIn("Picking backend: topaz", md)
        self.assertIn("Picking availability: available", md)
        self.assertIn("Picking execution mode: real", md)
        self.assertIn("Topaz executable:", md)


if __name__ == "__main__":
    unittest.main()
