"""Tests for Picking QC gate in the policy engine."""

from __future__ import annotations

import unittest

from agent.policy_engine import PolicyEngine
from schemas.action import ActionName
from schemas.qc_metrics import QCMetrics
from schemas.report import PipelineReport, ReportMeta
from schemas.sample_state import SampleState


def _qc_failed() -> dict:
    return {
        "qc_status": "failed_qc",
        "qc_flags": ["severe_under_picking", "many_empty_micrographs", "low_f1_against_ground_truth"],
        "recommended_next_action": "threshold_sweep_or_train_topaz",
        "n_picks": 43,
        "n_micrographs_total": 29,
        "zero_pick_micrograph_ratio": 0.551724,
        "ground_truth_evaluation": {
            "precision": 0.069767,
            "recall": 0.000558,
            "f1": 0.001108,
        },
    }


def _pe() -> PolicyEngine:
    return PolicyEngine(rules_path=None)


def _state(last_execution=None) -> SampleState:
    s = SampleState(project_id="p1", project_path="/tmp/p", stage="AutoPick", status="running")
    s.last_execution = last_execution
    return s


def _qc() -> QCMetrics:
    return QCMetrics(motioncorr_completed=True, ctf_completed=True)


class TestPickingQCGatePolicy(unittest.TestCase):
    def test_failed_qc_blocks_extract(self) -> None:
        state = _state(last_execution={
            "action_name": "run_autopick",
            "picking_backend": "topaz",
            "status": "success",
            "metadata": {"picking_qc": _qc_failed()},
        })
        action = _pe().decide_next_action(state, _qc())
        self.assertEqual(action.name, ActionName.REQUEST_HUMAN_REVIEW)
        self.assertNotEqual(action.name, ActionName.RUN_EXTRACT)
        self.assertEqual(action.parameters.get("branch_name"), "picking_qc")
        self.assertEqual(action.parameters.get("picking_qc_status"), "failed_qc")
        self.assertEqual(action.parameters.get("recommended_next_action"), "threshold_sweep_or_train_topaz")
        self.assertIn("severe_under_picking", action.parameters.get("picking_qc_flags", []))

    def test_passed_qc_allows_extract(self) -> None:
        state = _state(last_execution={
            "action_name": "run_autopick",
            "status": "success",
            "metadata": {"picking_qc": {"qc_status": "passed"}},
        })
        action = _pe().decide_next_action(state, _qc())
        self.assertEqual(action.name, ActionName.RUN_EXTRACT)

    def test_missing_qc_preserves_extract_behavior(self) -> None:
        state = _state(last_execution={
            "action_name": "run_autopick",
            "status": "success",
        })
        action = _pe().decide_next_action(state, _qc())
        self.assertEqual(action.name, ActionName.RUN_EXTRACT)

    def test_qc_read_from_top_level_execution(self) -> None:
        state = _state(last_execution={
            "action_name": "run_autopick",
            "status": "success",
            "picking_qc": _qc_failed(),
        })
        action = _pe().decide_next_action(state, _qc())
        self.assertEqual(action.name, ActionName.REQUEST_HUMAN_REVIEW)
        self.assertEqual(action.parameters.get("branch_name"), "picking_qc")


class TestPickingQCGateReport(unittest.TestCase):
    def test_report_markdown_shows_qc_gate_section(self) -> None:
        from schemas.action import Action

        state = _state()
        gate_action = Action(
            name=ActionName.REQUEST_HUMAN_REVIEW,
            reason="Picking QC failed after Topaz: severe_under_picking. Recommended recovery: threshold_sweep_or_train_topaz.",
            parameters={
                "branch_name": "picking_qc",
                "picking_qc_status": "failed_qc",
                "picking_qc_flags": ["severe_under_picking", "many_empty_micrographs"],
                "recommended_next_action": "threshold_sweep_or_train_topaz",
                "f1": 0.001108,
            },
            priority=95,
        )
        report = PipelineReport(state=state, qc=_qc(), next_action=gate_action)
        md = report.to_markdown()
        self.assertIn("Picking QC Gate", md)
        self.assertIn("failed_qc", md)
        self.assertIn("threshold_sweep_or_train_topaz", md)
        self.assertIn("0.001108", md)


if __name__ == "__main__":
    unittest.main()
