"""Reporter tests for compact upstream main-path summary."""

from __future__ import annotations

import unittest

from agent.reporter import Reporter
from schemas.action_history import ActionHistoryEntry
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState


class TestUpstreamReport(unittest.TestCase):
    def test_report_includes_upstream_main_path_summary(self) -> None:
        state = SampleState(project_id="p1", project_path="/tmp/p", stage="Class2D", status="running")
        qc = QCMetrics(class2d_completed=False)
        history = [
            ActionHistoryEntry(run_id="r1", stage="AutoPick", action_name="run_autopick", branch_name="main"),
            ActionHistoryEntry(run_id="r1", stage="Extract", action_name="run_extract", branch_name="main"),
            ActionHistoryEntry(run_id="r1", stage="Class2D", action_name="run_class2d", branch_name="main"),
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
        self.assertIn("Picking -> Extract -> Class2D", md)

    def test_report_compact_when_upstream_absent(self) -> None:
        state = SampleState(project_id="p2", project_path="/tmp/p2", stage="PostProcess", status="running")
        qc = QCMetrics(refine3d_completed=True)
        history = [
            ActionHistoryEntry(run_id="r2", stage="PostProcess", action_name="run_modelangelo", branch_name="modelangelo"),
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
        self.assertIn("## Stage Transitions", md)
        self.assertNotIn("Picking -> Extract -> Class2D", md)


if __name__ == "__main__":
    unittest.main()

