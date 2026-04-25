"""Tests for Picking QC gate in the policy engine.

The gate must trigger BEFORE any normal upstream decision (run_autopick or
run_extract), so that a filesystem scan that has fallen behind the actual
execution state cannot push the pipeline back into a re-pick loop.
"""

from __future__ import annotations

import unittest

from agent.policy_engine import PolicyEngine
from schemas.action import ActionName
from schemas.qc_metrics import QCMetrics
from schemas.report import PipelineReport
from schemas.sample_state import SampleState


def _qc_failed_payload() -> dict:
    return {
        "qc_status": "failed_qc",
        "qc_flags": [
            "severe_under_picking",
            "many_empty_micrographs",
            "low_f1_against_ground_truth",
        ],
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


def _state(stage: str = "AutoPick", last_execution=None) -> SampleState:
    s = SampleState(project_id="p1", project_path="/tmp/p", stage=stage, status="running")
    s.last_execution = last_execution
    return s


def _qc_ctf_done() -> QCMetrics:
    return QCMetrics(motioncorr_completed=True, ctf_completed=True)


class TestPickingQCGatePolicy(unittest.TestCase):
    def test_failed_qc_blocks_autopick_even_when_filesystem_stage_is_ctffind(self) -> None:
        """Filesystem scan still says CtfFind so decide_picking_stage would normally
        fire run_autopick. The gate must override based on persisted last_execution."""
        state = _state(
            stage="CtfFind",
            last_execution={
                "action_name": "run_autopick",
                "picking_backend": "topaz",
                "status": "success",
                "metadata": {"picking_qc": _qc_failed_payload()},
            },
        )
        action = _pe().decide_next_action(state, _qc_ctf_done())
        self.assertEqual(action.name, ActionName.REQUEST_HUMAN_REVIEW)
        self.assertNotEqual(action.name, ActionName.RUN_AUTOPICK)
        self.assertEqual(action.parameters.get("branch_name"), "picking_qc")
        self.assertEqual(action.parameters.get("picking_qc_status"), "failed_qc")
        self.assertEqual(
            action.parameters.get("recommended_next_action"),
            "threshold_sweep_or_train_topaz",
        )
        self.assertIn("severe_under_picking", action.parameters.get("picking_qc_flags", []))
        self.assertEqual(action.parameters.get("picking_backend"), "topaz")
        self.assertAlmostEqual(action.parameters.get("f1"), 0.001108)

    def test_failed_qc_blocks_extract_when_picking_completed(self) -> None:
        state = _state(
            stage="AutoPick",
            last_execution={
                "action_name": "run_autopick",
                "picking_backend": "topaz",
                "status": "success",
                "metadata": {"picking_qc": _qc_failed_payload()},
            },
        )
        action = _pe().decide_next_action(state, _qc_ctf_done())
        self.assertEqual(action.name, ActionName.REQUEST_HUMAN_REVIEW)
        self.assertNotEqual(action.name, ActionName.RUN_EXTRACT)
        self.assertEqual(action.parameters.get("branch_name"), "picking_qc")

    def test_passed_qc_does_not_block(self) -> None:
        state = _state(
            stage="AutoPick",
            last_execution={
                "action_name": "run_autopick",
                "picking_backend": "topaz",
                "status": "success",
                "metadata": {"picking_qc": {"qc_status": "passed"}},
            },
        )
        action = _pe().decide_next_action(state, _qc_ctf_done())
        self.assertEqual(action.name, ActionName.RUN_EXTRACT)

    def test_missing_qc_preserves_existing_behavior(self) -> None:
        # Without last_execution at CtfFind stage, the policy still fires run_autopick.
        state = _state(stage="CtfFind", last_execution=None)
        action = _pe().decide_next_action(state, _qc_ctf_done())
        self.assertEqual(action.name, ActionName.RUN_AUTOPICK)
        # Without last_execution at AutoPick stage, the policy still fires run_extract.
        state2 = _state(stage="AutoPick", last_execution=None)
        action2 = _pe().decide_next_action(state2, _qc_ctf_done())
        self.assertEqual(action2.name, ActionName.RUN_EXTRACT)

    def test_qc_can_be_read_from_top_level_execution(self) -> None:
        state = _state(
            stage="AutoPick",
            last_execution={
                "action_name": "run_autopick",
                "picking_backend": "topaz",
                "status": "success",
                "picking_qc": _qc_failed_payload(),
            },
        )
        action = _pe().decide_next_action(state, _qc_ctf_done())
        self.assertEqual(action.name, ActionName.REQUEST_HUMAN_REVIEW)
        self.assertEqual(action.parameters.get("branch_name"), "picking_qc")

    def test_failed_qc_with_non_topaz_backend_does_not_fire_gate(self) -> None:
        # Gate is Topaz-specific; a relion_log run should not be force-blocked.
        state = _state(
            stage="AutoPick",
            last_execution={
                "action_name": "run_autopick",
                "picking_backend": "relion_log",
                "status": "success",
                "metadata": {"picking_qc": _qc_failed_payload()},
            },
        )
        action = _pe().decide_next_action(state, _qc_ctf_done())
        self.assertEqual(action.name, ActionName.RUN_EXTRACT)

    def test_gate_is_idempotent_when_last_execution_is_human_review_dry_run(self) -> None:
        """After run-next dry-runs the gate action, last_execution is overwritten with
        a request_human_review result. The gate must still fire on subsequent calls."""
        state = _state(
            stage="CtfFind",
            last_execution={
                "action_name": "request_human_review",
                "branch_name": "picking_qc",
                "picking_backend": "topaz",
                "picking_qc_status": "failed_qc",
                "picking_qc_flags": [
                    "severe_under_picking",
                    "many_empty_micrographs",
                    "low_f1_against_ground_truth",
                ],
                "recommended_next_action": "threshold_sweep_or_train_topaz",
                "n_picks": 43,
                "n_micrographs_total": 29,
                "zero_pick_micrograph_ratio": 0.551724,
                "precision": 0.069767,
                "recall": 0.000558,
                "f1": 0.001108,
                "status": "dry_run",
                "metadata": {
                    "branch_name": "picking_qc",
                    "picking_backend": "topaz",
                    "picking_qc_status": "failed_qc",
                    "picking_qc_flags": [
                        "severe_under_picking",
                        "many_empty_micrographs",
                        "low_f1_against_ground_truth",
                    ],
                    "recommended_next_action": "threshold_sweep_or_train_topaz",
                },
            },
        )
        action = _pe().decide_next_action(state, _qc_ctf_done())
        self.assertEqual(action.name, ActionName.REQUEST_HUMAN_REVIEW)
        self.assertNotEqual(action.name, ActionName.RUN_AUTOPICK)
        self.assertEqual(action.parameters.get("branch_name"), "picking_qc")
        self.assertEqual(action.parameters.get("picking_qc_status"), "failed_qc")
        self.assertEqual(
            action.parameters.get("recommended_next_action"),
            "threshold_sweep_or_train_topaz",
        )
        self.assertAlmostEqual(action.parameters.get("f1"), 0.001108)

    def test_heterogeneity_human_review_does_not_trigger_gate(self) -> None:
        # A human_review for heterogeneity (different branch) must not activate the gate.
        state = _state(
            stage="Class2D",
            last_execution={
                "action_name": "request_human_review",
                "branch_name": "heterogeneity",
                "picking_qc_status": "",
                "status": "dry_run",
                "metadata": {
                    "branch_name": "heterogeneity",
                    "heterogeneity_backend": "human_review",
                },
            },
        )
        qc = QCMetrics(
            motioncorr_completed=True,
            ctf_completed=True,
            heterogeneity_warning=True,
        )
        action = _pe().decide_next_action(state, qc)
        self.assertNotEqual(action.name, ActionName.RUN_AUTOPICK)
        # Gate must not fire; heterogeneity review or pipeline continuation is expected.
        self.assertNotEqual(action.parameters.get("branch_name"), "picking_qc")


class TestPickingQCGateReportMarkdown(unittest.TestCase):
    """to_markdown shows compact gate section when next_action is the gate."""

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
        report = PipelineReport(state=state, qc=_qc_ctf_done(), next_action=gate_action)
        md = report.to_markdown()
        self.assertIn("Picking QC Gate", md)
        self.assertIn("failed_qc", md)
        self.assertIn("threshold_sweep_or_train_topaz", md)
        self.assertIn("0.001108", md)


if __name__ == "__main__":
    unittest.main()
