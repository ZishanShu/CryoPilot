"""F1 end-to-end: fixed_workflow_topaz + llm_supervised_agent reach execution.

These tests verify that, after the case->backend input mapping fix, F1 no
longer stops at preflight_blocked but actually runs the Topaz executor (using
a fake topaz shell script that emits the real PngImageFile error and exits
non-zero) and that llm_supervised_agent collects non-null supervision payloads
for preflight + execution stop_points.
"""

from __future__ import annotations

import json
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict

from agent.baseline_runner import run_experiment_case
from agent.benchmark_harness import BenchmarkHarness
from agent.case_registry import ExperimentCase
from agent.experiment_protocol import resolve_protocol
from app.scheduler import Scheduler


_PNG_STDERR = (
    "Traceback (most recent call last):\n"
    '  File "topaz/extract.py", line 1, in <module>\n'
    "AttributeError: 'PngImageFile' object has no attribute 'shape'\n"
)


def _write_fake_topaz(path: Path, *, exit_code: int = 1) -> Path:
    """Write a POSIX shell script that mimics topaz failing on PNG micrographs."""
    script = (
        "#!/bin/sh\n"
        "# Fake topaz binary for F1 replay tests.\n"
        'echo "fake-topaz start" >&2\n'
        f"cat >&2 <<'EOF'\n{_PNG_STDERR}EOF\n"
        f"exit {int(exit_code)}\n"
    )
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IRUSR | stat.S_IWUSR)
    return path


def _build_f1_case(
    *,
    micrographs_star: Path,
    output_dir: Path,
    topaz_executable: Path,
) -> ExperimentCase:
    return ExperimentCase(
        case_id="F1_cryoppp_lite_topaz_input_format_failure",
        case_type="workflow_case",
        dataset_name="cryoppp_lite",
        dataset_root=str(micrographs_star.parent),
        input_state={
            "stage": "picking",
            "expected_failure_phase": "execution",
            "expected_failure_kind": "backend_runtime_error",
            "expected_root_cause_guess": "input_format_incompatibility",
            "topaz_executable": str(topaz_executable),
        },
        available_inputs={
            "micrographs_star": str(micrographs_star),
            "particle_diameter_angstrom": 200,
            "output_dir": str(output_dir),
            "topaz_device": 0,
        },
        target_stage="picking",
        evaluation_mode="picking_coords",
        notes="F1 fixture",
    )


def _prepare_fixture(root: Path) -> Dict[str, Path]:
    data_root = root / "data"
    run_root = root / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    f1_dir = run_root / "F1"
    f1_dir.mkdir(parents=True, exist_ok=True)

    # Minimal RELION-style micrographs star with one .png file. Topaz never
    # gets a chance to read the micrograph because our fake topaz exits early,
    # but the path must resolve so argv build succeeds.
    mic_file = f1_dir / "m1.png"
    mic_file.write_bytes(b"not-a-real-png")
    star = f1_dir / "micrographs.star"
    star.write_text(
        "data_micrographs\n\nloop_\n_rlnMicrographName #1\n"
        f"{mic_file.name}\n",
        encoding="utf-8",
    )

    output_dir = f1_dir / "picks"
    output_dir.mkdir(parents=True, exist_ok=True)

    fake_topaz = run_root / "fake_topaz.sh"
    _write_fake_topaz(fake_topaz, exit_code=1)

    return {
        "project_root": root,
        "data_root": data_root,
        "run_root": run_root,
        "micrographs_star": star,
        "output_dir": output_dir,
        "topaz_executable": fake_topaz,
    }


class TestF1ReachesExecution(unittest.TestCase):
    def test_fixed_workflow_topaz_reaches_execution_and_fails(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fx = _prepare_fixture(root)
            case = _build_f1_case(
                micrographs_star=fx["micrographs_star"],
                output_dir=fx["output_dir"],
                topaz_executable=fx["topaz_executable"],
            )
            protocol = resolve_protocol("fixed_workflow_topaz")

            scheduler = Scheduler(project_root=fx["project_root"])
            harness = BenchmarkHarness(
                executor=scheduler.action_executor,
                project_root=fx["project_root"],
            )

            result = run_experiment_case(
                protocol=protocol,
                case=case,
                harness=harness,
                project_root=fx["project_root"],
                output_dir=fx["project_root"] / "out_f1",
                dry_run=False,
                persist=False,
                write_audit=True,
            )

            payload = result.to_dict()
            # The core acceptance: not preflight_blocked.
            self.assertNotEqual(payload["status"], "preflight_blocked")
            self.assertEqual(payload["stage_reached"], "execution")

            cr = payload["case_result"]
            self.assertIsNotNone(cr.get("command"))
            self.assertIsNotNone(cr.get("return_code"))
            self.assertNotEqual(cr.get("return_code"), None)
            self.assertEqual(cr.get("return_code"), 1)
            self.assertIn("PngImageFile", str(cr.get("stderr") or ""))
            # Regression: device must be normalized to int before topaz sees
            # it; no "invalid int value: 'CPU'" argparse error anywhere.
            self.assertNotIn("invalid int value", str(cr.get("stderr") or ""))
            self.assertIn("-d 0", str(cr.get("command") or ""))

            pf = cr.get("preflight") or {}
            self.assertTrue(pf.get("available"))

            # failure_audit must classify as execution / backend_runtime_error.
            audit_json = payload["audit_paths"].get("failure_audit_json")
            self.assertIsNotNone(audit_json)
            audit = json.loads(Path(audit_json).read_text(encoding="utf-8"))
            cls = audit.get("failure_classification") or {}
            self.assertEqual(cls.get("failure_phase"), "execution")
            self.assertEqual(cls.get("failure_kind"), "backend_runtime_error")
            self.assertEqual(
                cls.get("root_cause_guess"), "input_format_incompatibility"
            )

    def test_llm_supervised_agent_collects_non_null_payloads(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fx = _prepare_fixture(root)
            case = _build_f1_case(
                micrographs_star=fx["micrographs_star"],
                output_dir=fx["output_dir"],
                topaz_executable=fx["topaz_executable"],
            )
            protocol = resolve_protocol("llm_supervised_agent")

            scheduler = Scheduler(project_root=fx["project_root"])
            harness = BenchmarkHarness(
                executor=scheduler.action_executor,
                project_root=fx["project_root"],
            )
            result = run_experiment_case(
                protocol=protocol,
                case=case,
                harness=harness,
                project_root=fx["project_root"],
                output_dir=fx["project_root"] / "out_f1_llm",
                dry_run=False,
                persist=False,
                write_audit=True,
            )

            payload = result.to_dict()
            sup = payload["supervision_payloads"]
            self.assertIn("preflight", sup)
            self.assertIn("execution", sup)
            self.assertIsNotNone(sup["preflight"])
            self.assertIsNotNone(sup["execution"])
            # preflight payload should reflect availability=true
            self.assertTrue(sup["preflight"].get("available"))
            # execution payload should expose command + return_code + stderr tail
            self.assertIsNotNone(sup["execution"].get("command"))
            self.assertEqual(sup["execution"].get("return_code"), 1)
            self.assertIn("PngImageFile", str(sup["execution"].get("stderr_tail") or ""))
            self.assertNotIn(
                "invalid int value",
                str(sup["execution"].get("stderr_tail") or ""),
            )


class TestF1FixtureMappingSanity(unittest.TestCase):
    """Pre-run assertions that the fixture actually produces the contract fields."""

    def test_merge_inputs_translates_micrographs_star(self) -> None:
        from agent.baseline_runner import _merge_inputs

        case = ExperimentCase(
            case_id="stub",
            case_type="workflow_case",
            dataset_name="cryoppp_lite",
            input_state={"stage": "picking"},
            available_inputs={
                "micrographs_star": "/tmp/x/m.star",
                "particle_diameter_angstrom": 200,
                "output_dir": "/tmp/x/out",
            },
            target_stage="picking",
        )
        selection = {
            "selected_backend": "topaz",
            "category": "picking",
        }
        inputs = _merge_inputs(case, selection)
        self.assertEqual(inputs["micrographs_star_path"], "/tmp/x/m.star")
        self.assertEqual(inputs["particle_diameter_angstrom"], 200)
        self.assertEqual(inputs["output_dir"], "/tmp/x/out")


if __name__ == "__main__":
    unittest.main()
