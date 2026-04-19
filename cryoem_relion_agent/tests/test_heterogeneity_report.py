"""Reporter tests for heterogeneity branch summaries."""

from __future__ import annotations

import unittest

from agent.reporter import Reporter
from schemas.action_history import ActionHistoryEntry
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState


class TestHeterogeneityReport(unittest.TestCase):
    def test_report_includes_heterogeneity_branch_summary(self) -> None:
        state = SampleState(project_id="p1", project_path="/tmp/p", stage="Class3D", status="running")
        qc = QCMetrics(class2d_completed=True, heterogeneity_warning=True)
        history = [
            ActionHistoryEntry(run_id="r1", stage="Class2D", action_name="run_class2d_cleanup", branch_name="main"),
            ActionHistoryEntry(run_id="r1", stage="Class3D", action_name="request_human_review", branch_name="heterogeneity"),
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
        self.assertIn("Branch: heterogeneity", md)

    def test_report_compact_when_no_heterogeneity_branch(self) -> None:
        state = SampleState(project_id="p1", project_path="/tmp/p", stage="Class2D", status="running")
        qc = QCMetrics(class2d_completed=True)
        history = [
            ActionHistoryEntry(run_id="r2", stage="Class2D", action_name="run_class2d_cleanup", branch_name="main"),
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
        self.assertNotIn("Branch: heterogeneity", md)
        self.assertIn("Branch: main", md)


if __name__ == "__main__":
    unittest.main()

