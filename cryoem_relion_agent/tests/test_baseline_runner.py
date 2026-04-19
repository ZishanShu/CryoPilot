"""Tests for agent.baseline_runner experiment-case runner."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.baseline_runner import (
    ExperimentResult,
    build_benchmark_case,
    run_experiment_case,
)
from agent.benchmark_harness import BenchmarkHarness
from agent.case_registry import parse_case
from agent.experiment_protocol import default_protocols, parse_protocol
from app.scheduler import Scheduler


def _fixed_workflow() -> "ExperimentResult":
    return default_protocols()["fixed_workflow_topaz"]


def _rule_only() -> "ExperimentResult":
    return default_protocols()["rule_only_agent"]


def _llm_supervised() -> "ExperimentResult":
    return default_protocols()["llm_supervised_agent"]


def _picking_case() -> "ExperimentResult":
    return parse_case(
        {
            "case_id": "cryoppp_lite_topaz_smoke",
            "case_type": "workflow_case",
            "dataset_name": "cryoppp_lite",
            "target_stage": "picking",
            "evaluation_mode": "picking_coords",
        }
    )


class TestBuildBenchmarkCase(unittest.TestCase):
    def test_fixed_workflow_selects_single_backend(self) -> None:
        case_dict = build_benchmark_case(_fixed_workflow(), _picking_case())
        self.assertEqual(case_dict["category"], "picking")
        self.assertEqual(case_dict["backend"], "topaz")
        self.assertEqual(case_dict["inputs"].get("picking_backend"), "topaz")
        self.assertIn("picking_selection_decision", case_dict["inputs"])
        self.assertIn("extra", case_dict)
        self.assertEqual(
            case_dict["extra"]["protocol_name"], "fixed_workflow_topaz"
        )

    def test_rule_only_agent_chooses_registry_default(self) -> None:
        case_dict = build_benchmark_case(_rule_only(), _picking_case())
        self.assertEqual(case_dict["backend"], "topaz")
        self.assertEqual(case_dict["extra"]["protocol_name"], "rule_only_agent")


class TestRunExperimentCase(unittest.TestCase):
    def _harness(self, root: Path) -> BenchmarkHarness:
        sched = Scheduler(project_root=root)
        return BenchmarkHarness(executor=sched.action_executor, project_root=root)

    def test_fixed_workflow_preflight_blocked_persists_and_writes_failure_audit(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_experiment_case(
                protocol=_fixed_workflow(),
                case=_picking_case(),
                harness=self._harness(root),
                project_root=root,
                dry_run=False,
                persist=True,
                write_audit=True,
            )
            self.assertIsInstance(result, ExperimentResult)
            self.assertEqual(result.protocol_name, "fixed_workflow_topaz")
            self.assertEqual(result.case_id, "cryoppp_lite_topaz_smoke")
            self.assertEqual(result.selected_backend, "topaz")
            self.assertEqual(result.status, "preflight_blocked")
            self.assertEqual(result.stage_reached, "preflight")
            # audit should exist as failure audit since preflight blocked
            self.assertIn("failure_audit_json", result.audit_paths)
            self.assertTrue(Path(result.audit_paths["failure_audit_json"]).is_file())
            self.assertTrue(Path(result.audit_paths["failure_audit_md"]).is_file())
            # persistence created latest.json + history
            self.assertTrue(Path(result.benchmark_paths["latest_json"]).is_file())
            self.assertTrue(Path(result.benchmark_paths["history_json"]).is_file())

    def test_rule_only_agent_preflight_blocked_dry_run(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_experiment_case(
                protocol=_rule_only(),
                case=_picking_case(),
                harness=self._harness(root),
                project_root=root,
                dry_run=True,
                persist=True,
                write_audit=True,
            )
            self.assertEqual(result.selected_backend, "topaz")
            self.assertEqual(result.routing_mode, "rule")
            self.assertFalse(result.llm_supervision_enabled)

    def test_llm_supervised_agent_surfaces_supervision_stop_points(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_experiment_case(
                protocol=_llm_supervised(),
                case=_picking_case(),
                harness=self._harness(root),
                project_root=root,
                dry_run=True,
                persist=True,
                write_audit=False,
            )
            self.assertTrue(result.llm_supervision_enabled)
            # even if payloads are absent (dry_run) keys should still appear
            for sp in ("preflight", "execution", "evaluation", "diff"):
                self.assertIn(sp, result.supervision_payloads)

    def test_result_is_json_serializable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_experiment_case(
                protocol=_fixed_workflow(),
                case=_picking_case(),
                harness=self._harness(root),
                project_root=root,
                dry_run=True,
                persist=False,
                write_audit=False,
            )
            json.dumps(result.to_dict())

    def test_experiment_result_json_written_to_output_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "experiments_out"
            result = run_experiment_case(
                protocol=_fixed_workflow(),
                case=_picking_case(),
                harness=self._harness(root),
                project_root=root,
                output_dir=out,
                dry_run=True,
                persist=False,
                write_audit=False,
            )
            self.assertTrue((out / "experiment_result.json").is_file())
            loaded = json.loads((out / "experiment_result.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["protocol_name"], result.protocol_name)


class TestAuditInvariants(unittest.TestCase):
    def test_failure_audit_has_run_identity_when_preflight_blocked(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sched = Scheduler(project_root=root)
            harness = BenchmarkHarness(executor=sched.action_executor, project_root=root)
            result = run_experiment_case(
                protocol=_fixed_workflow(),
                case=_picking_case(),
                harness=harness,
                project_root=root,
                dry_run=False,
                persist=True,
                write_audit=True,
            )
            audit = json.loads(
                Path(result.audit_paths["failure_audit_json"]).read_text(encoding="utf-8")
            )
            self.assertIn("run_identity", audit)
            self.assertIn("failure_classification", audit)
            self.assertEqual(audit["run_identity"]["backend"], "topaz")


if __name__ == "__main__":
    unittest.main()
