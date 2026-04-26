"""Reporter tests for compact picking backend summary."""

from __future__ import annotations

import unittest

from agent.reporter import Reporter
from schemas.action import Action, ActionName
from schemas.action_history import ActionHistoryEntry
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState


class TestPickerRouterReport(unittest.TestCase):
    def test_report_includes_picking_backend_when_available(self) -> None:
        state = SampleState(project_id="p1", project_path="/tmp/p", stage="CtfFind", status="running")
        qc = QCMetrics(motioncorr_completed=True, ctf_completed=True)
        next_action = Action(
            name=ActionName.RUN_AUTOPICK,
            reason="picking pending",
            parameters={"picking_backend": "topaz"},
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

    def test_report_compact_when_backend_absent(self) -> None:
        state = SampleState(project_id="p1", project_path="/tmp/p", stage="CtfFind", status="running")
        qc = QCMetrics(motioncorr_completed=True, ctf_completed=True)
        history = [ActionHistoryEntry(run_id="r1", stage="CtfFind", action_name="run_autopick", branch_name="main")]
        report = Reporter().build_report(
            state=state,
            qc=qc,
            recent_action=None,
            next_action=None,
            action_history=history,
            current_run_id="r1",
        )
        md = report.to_markdown()
        self.assertIn("## Actions", md)
        self.assertNotIn("Picking backend:", md)


class TestPickerRouterReportPickingQC(unittest.TestCase):
    def test_report_includes_picking_qc_section(self) -> None:
        state = SampleState(
            project_id="p1",
            project_path="/tmp/p",
            stage="AutoPick",
            status="running",
            last_execution={
                "action_name": "run_autopick",
                "status": "success",
                "picking_qc": {
                    "qc_status": "failed_qc",
                    "n_picks": 43,
                    "n_micrographs_with_picks": 29,
                    "n_micrographs_total": 29,
                    "mean_picks_per_micrograph": 1.48,
                    "zero_pick_micrograph_ratio": 0.0,
                    "ground_truth_evaluation": {
                        "tp": 6,
                        "fp": 37,
                        "fn": 5368,
                        "precision": 0.13,
                        "recall": 0.001,
                        "f1": 0.0011,
                        "matching_tolerance_px": 80.0,
                    },
                    "qc_flags": ["severe_under_picking", "low_f1_against_ground_truth"],
                    "recommended_next_action": "threshold_sweep_or_train_topaz",
                },
            },
        )
        qc = QCMetrics(motioncorr_completed=True, ctf_completed=True)
        next_action = Action(
            name=ActionName.RUN_AUTOPICK,
            reason="picking pending",
            parameters={"picking_backend": "topaz"},
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
        self.assertIn("## Picking QC", md)
        self.assertIn("QC status: failed_qc", md)
        self.assertIn("Picks: 43", md)
        self.assertIn("severe_under_picking", md)
        self.assertIn("threshold_sweep_or_train_topaz", md)

    def test_report_omits_picking_qc_section_when_absent(self) -> None:
        state = SampleState(project_id="p1", project_path="/tmp/p", stage="CtfFind", status="running")
        qc = QCMetrics(motioncorr_completed=True, ctf_completed=True)
        report = Reporter().build_report(
            state=state,
            qc=qc,
            recent_action=None,
            next_action=None,
            action_history=[],
            current_run_id=None,
        )
        md = report.to_markdown()
        self.assertNotIn("## Picking QC", md)


if __name__ == "__main__":
    unittest.main()
