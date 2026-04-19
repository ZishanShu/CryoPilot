"""cryoDRGN heterogeneity backend: contract, preflight, scheduler, executor, report."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from agent.backend_registry import HETEROGENEITY_BACKEND_METADATA, HETEROGENEITY_INPUT_SPECS
from agent.heterogeneity_preflight import enrich_and_preflight_cryodrgn_params, preflight_cryodrgn
from app.scheduler import Scheduler
from relion.relion_actions import RelionActionExecutor
from schemas.action import Action, ActionName
from schemas.sample_state import SampleState

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


class TestCryodrgnInputContract(unittest.TestCase):
    def test_cryodrgn_backend_has_spec_and_metadata(self) -> None:
        self.assertIn("cryodrgn", HETEROGENEITY_INPUT_SPECS)
        spec = HETEROGENEITY_INPUT_SPECS["cryodrgn"]
        self.assertIn("output_dir", spec["required_fields"])
        self.assertIn("particles_star_path", spec["optional_fields"])
        for opt in (
            "cryodrgn_executable",
            "cryodrgn_zdim",
            "cryodrgn_epochs",
            "downsample_box_size",
            "pose_pkl_path",
            "ctf_pkl_path",
            "particles_mrcs_path",
        ):
            self.assertIn(opt, spec["optional_fields"])
        meta = HETEROGENEITY_BACKEND_METADATA["cryodrgn"]
        self.assertEqual(meta.get("required_fields"), list(spec["required_fields"]))


class TestCryodrgnPreflight(unittest.TestCase):
    def test_missing_star_and_no_external_stack_unavailable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "out"
            out.mkdir()
            pf = preflight_cryodrgn(
                {"output_dir": str(out)},
                project_root=root,
                cryodrgn_config_path=CONFIGS / "cryodrgn.yaml",
            )
            self.assertFalse(pf["available"])
            self.assertEqual(pf["execution_mode"], "unavailable")
            self.assertIn("particles_star_path", pf.get("missing_inputs") or [])

    def test_missing_executable_unavailable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Refine3D" / "j").mkdir(parents=True)
            star = root / "Refine3D" / "j" / "run_particles.star"
            star.write_text("dummy")
            out = root / "cd_out"
            out.mkdir()
            pf = preflight_cryodrgn(
                {"particles_star_path": str(star), "output_dir": str(out), "cryodrgn_executable": "/nonexistent/cryodrgn"},
                project_root=root,
                cryodrgn_config_path=CONFIGS / "cryodrgn.yaml",
            )
            self.assertFalse(pf["available"])
            self.assertIn("cryodrgn_executable", " ".join(pf.get("missing_dependencies") or []))
            self.assertIn("missing_dependency", str(pf.get("blocked_reason") or ""))

    def test_available_with_mock_executable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Refine3D" / "j").mkdir(parents=True)
            star = root / "Refine3D" / "j" / "run_particles.star"
            star.write_text("dummy")
            out = root / "cd_out"
            out.mkdir()
            fake = root / "cryodrgn"
            fake.write_text("#!/bin/sh\necho\n")
            fake.chmod(0o755)
            pf = preflight_cryodrgn(
                {"particles_star_path": str(star), "output_dir": str(out), "cryodrgn_executable": str(fake)},
                project_root=root,
                cryodrgn_config_path=CONFIGS / "cryodrgn.yaml",
            )
            self.assertTrue(pf["available"])
            self.assertEqual(pf["execution_mode"], "real")
            self.assertEqual(pf["executable_path"], str(fake.resolve()))


class TestCryodrgnSchedulerEnrich(unittest.TestCase):
    def test_enrich_sets_availability_tags(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Refine3D" / "j").mkdir(parents=True)
            star = root / "Refine3D" / "j" / "run_particles.star"
            star.write_text("x")
            fake = root / "cryodrgn"
            fake.write_text("#!/bin/sh\n")
            fake.chmod(0o755)
            sched = Scheduler(project_root=root)
            state = SampleState(project_id="p", project_path=str(root), stage="Class3D", status="running")
            action = Action(
                name=ActionName.RUN_CRYODRGN,
                reason="r",
                parameters={
                    "heterogeneity_backend": "cryodrgn",
                    "branch_name": "heterogeneity",
                    "cryodrgn_executable": str(fake),
                },
                priority=94,
            )
            enriched = sched._enrich_cryodrgn_action(state, action)
            self.assertEqual(enriched.parameters.get("heterogeneity_backend"), "cryodrgn")
            self.assertEqual(enriched.parameters.get("heterogeneity_availability"), "available")
            self.assertEqual(enriched.parameters.get("heterogeneity_execution_mode"), "real")
            tags = sched._backend_tags(enriched)
            self.assertIn("heterogeneity_availability=available", tags)
            self.assertIn("heterogeneity_execution_mode=real", tags)

    def test_enrich_unavailable_when_executable_missing(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Refine3D" / "j").mkdir(parents=True)
            star = root / "Refine3D" / "j" / "run_particles.star"
            star.write_text("x")
            sched = Scheduler(project_root=root)
            state = SampleState(project_id="p", project_path=str(root), stage="Class3D", status="running")
            action = Action(
                name=ActionName.RUN_CRYODRGN,
                reason="r",
                parameters={"heterogeneity_backend": "cryodrgn", "branch_name": "heterogeneity"},
                priority=94,
            )
            enriched = sched._enrich_cryodrgn_action(state, action)
            self.assertEqual(enriched.parameters.get("heterogeneity_availability"), "unavailable")
            self.assertTrue(bool(enriched.parameters.get("heterogeneity_blocked_reason")))
            tags = sched._backend_tags(enriched)
            self.assertIn("heterogeneity_availability=unavailable", tags)


class TestCryodrgnExecutor(unittest.TestCase):
    def test_preflight_blocked_skips_subprocess(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ex = RelionActionExecutor(CONFIGS / "schemes.yaml")
            params = enrich_and_preflight_cryodrgn_params(
                {"heterogeneity_backend": "cryodrgn", "branch_name": "heterogeneity"},
                project_root=root,
                cryodrgn_config_path=CONFIGS / "cryodrgn.yaml",
            )
            action = Action(
                name=ActionName.RUN_CRYODRGN,
                reason="r",
                parameters=params,
                priority=94,
            )
            with patch("relion.cryodrgn_runner.subprocess.run") as m_run:
                res = ex.execute(action=action, project_root=root, dry_run=False)
            m_run.assert_not_called()
            self.assertEqual(res.get("status"), "preflight_blocked")
            self.assertFalse(res.get("success"))

    def test_pipeline_runs_subprocess_steps_when_preflight_ok(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Refine3D" / "j").mkdir(parents=True)
            star = root / "Refine3D" / "j" / "run_particles.star"
            star.write_text("x")
            fake = root / "cryodrgn"
            fake.write_text("#!/bin/sh\n")
            fake.chmod(0o755)
            params = enrich_and_preflight_cryodrgn_params(
                {
                    "heterogeneity_backend": "cryodrgn",
                    "branch_name": "heterogeneity",
                    "cryodrgn_executable": str(fake),
                },
                project_root=root,
                cryodrgn_config_path=CONFIGS / "cryodrgn.yaml",
            )
            action = Action(name=ActionName.RUN_CRYODRGN, reason="r", parameters=params, priority=94)

            def fake_run(cmd: list, **kwargs: object) -> MagicMock:
                c = " ".join(cmd)
                proc = MagicMock()
                proc.returncode = 0
                proc.stdout = ""
                proc.stderr = ""
                if "parse_pose_star" in c:
                    out = cmd[cmd.index("-o") + 1]
                    Path(out).parent.mkdir(parents=True, exist_ok=True)
                    Path(out).write_bytes(b".")
                elif "parse_ctf_star" in c:
                    out = cmd[cmd.index("-o") + 1]
                    Path(out).write_bytes(b".")
                elif "train_vae" in c:
                    oi = cmd.index("-o") + 1
                    tout = Path(cmd[oi])
                    tout.mkdir(parents=True, exist_ok=True)
                    (tout / "config.yaml").write_text("k: v")
                    (tout / "weights.pkl").write_bytes(b".")
                return proc

            ex = RelionActionExecutor(CONFIGS / "schemes.yaml")
            with patch("relion.cryodrgn_runner.subprocess.run", side_effect=fake_run):
                res = ex.execute(action=action, project_root=root, dry_run=False)
            self.assertEqual(res.get("status"), "completed")
            self.assertTrue(res.get("success"))
            self.assertEqual(res.get("heterogeneity_backend"), "cryodrgn")

    def test_no_false_completed_when_train_missing_weights(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Refine3D" / "j").mkdir(parents=True)
            star = root / "Refine3D" / "j" / "run_particles.star"
            star.write_text("x")
            fake = root / "cryodrgn"
            fake.write_text("#!/bin/sh\n")
            fake.chmod(0o755)
            params = enrich_and_preflight_cryodrgn_params(
                {"heterogeneity_backend": "cryodrgn", "branch_name": "heterogeneity", "cryodrgn_executable": str(fake)},
                project_root=root,
                cryodrgn_config_path=CONFIGS / "cryodrgn.yaml",
            )
            action = Action(name=ActionName.RUN_CRYODRGN, reason="r", parameters=params, priority=94)

            def fake_run(cmd: list, **kwargs: object) -> MagicMock:
                c = " ".join(cmd)
                proc = MagicMock()
                proc.returncode = 0
                proc.stdout = ""
                proc.stderr = ""
                if "parse_pose_star" in c:
                    Path(cmd[cmd.index("-o") + 1]).write_bytes(b".")
                elif "parse_ctf_star" in c:
                    Path(cmd[cmd.index("-o") + 1]).write_bytes(b".")
                elif "train_vae" in c:
                    oi = cmd.index("-o") + 1
                    Path(cmd[oi]).mkdir(parents=True, exist_ok=True)
                return proc

            ex = RelionActionExecutor(CONFIGS / "schemes.yaml")
            with patch("relion.cryodrgn_runner.subprocess.run", side_effect=fake_run):
                res = ex.execute(action=action, project_root=root, dry_run=False)
            self.assertEqual(res.get("status"), "failed")
            self.assertFalse(res.get("success"))


class TestCryodrgnReport(unittest.TestCase):
    def test_markdown_shows_heterogeneity_preflight(self) -> None:
        from agent.reporter import Reporter
        from schemas.qc_metrics import QCMetrics

        state = SampleState(project_id="p1", project_path="/tmp/p", stage="Class3D", status="running")
        qc = QCMetrics(class2d_completed=True)
        next_action = Action(
            name=ActionName.RUN_CRYODRGN,
            reason="heterogeneity",
            parameters={
                "heterogeneity_backend": "cryodrgn",
                "branch_name": "heterogeneity",
                "heterogeneity_availability": "unavailable",
                "heterogeneity_execution_mode": "unavailable",
                "heterogeneity_blocked_reason": "missing_dependency: cryodrgn_executable",
                "cryodrgn_preflight": {
                    "available": False,
                    "execution_mode": "unavailable",
                    "blocked_reason": "missing_dependency: cryodrgn_executable",
                    "executable_path": None,
                    "input_summary": "test",
                },
            },
            priority=94,
        )
        report = Reporter().build_report(state=state, qc=qc, recent_action=None, next_action=next_action, action_history=[])
        md = report.to_markdown()
        self.assertIn("Heterogeneity backend: cryodrgn", md)
        self.assertIn("Heterogeneity availability: unavailable", md)
        self.assertIn("Heterogeneity execution mode: unavailable", md)
        self.assertIn("missing_dependency", md)


if __name__ == "__main__":
    unittest.main()
