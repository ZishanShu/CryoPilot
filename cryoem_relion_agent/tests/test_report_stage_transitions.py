"""Tests for compact stage-transition and branch summaries in reports."""

from __future__ import annotations

import unittest

from agent.reporter import Reporter
from schemas.action_history import ActionHistoryEntry
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState


class TestReportStageTransitions(unittest.TestCase):
    def test_report_contains_main_path_stage_summary(self) -> None:
        state = SampleState(project_id="p1", project_path="/tmp/p", stage="Refine3D", status="running")
        qc = QCMetrics(refine3d_completed=True)
        history = [
            ActionHistoryEntry(run_id="r1", stage="Picking", action_name="run_picking_extract", branch_name="main"),
            ActionHistoryEntry(run_id="r1", stage="Class2D", action_name="run_class2d_cleanup", branch_name="main"),
            ActionHistoryEntry(run_id="r1", stage="Refine3D", action_name="run_refine_ctf_polish", branch_name="main"),
        ]
        report = Reporter().build_report(
            state=state,
            qc=qc,
            recent_action=None,
            next_action=None,
            action_history=history,
            current_run_id="r1",
        )
        md = report.to_markdown()
        self.assertIn("Run ID: r1", md)
        self.assertIn("Picking -> Class2D -> Refine3D", md)

    def test_report_contains_branch_summary_when_present(self) -> None:
        state = SampleState(project_id="p1", project_path="/tmp/p", stage="PostProcess", status="running")
        qc = QCMetrics(refine3d_completed=True, current_resolution=3.8)
        history = [
            ActionHistoryEntry(run_id="r2", stage="PostProcess", action_name="delegate_structure_router", branch_name="structure_router"),
            ActionHistoryEntry(run_id="r2", stage="PostProcess", action_name="run_modelangelo", branch_name="modelangelo"),
            ActionHistoryEntry(run_id="r2", stage="Class2D", action_name="request_human_review", branch_name="heterogeneity"),
        ]
        report = Reporter().build_report(
            state=state,
            qc=qc,
            recent_action=None,
            next_action=None,
            action_history=history,
            current_run_id="r2",
        )
        md = report.to_markdown()
        self.assertIn("Branch: structure_router", md)
        self.assertIn("Branch: modelangelo", md)
        self.assertIn("Branch: heterogeneity", md)


if __name__ == "__main__":
    unittest.main()

