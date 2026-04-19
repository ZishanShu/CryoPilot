"""Reporter tests for compact mid-3D main-path summaries."""

from __future__ import annotations

import unittest

from agent.reporter import Reporter
from schemas.action_history import ActionHistoryEntry
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState


class TestMid3DReport(unittest.TestCase):
    def test_report_includes_mid3d_summary_when_present(self) -> None:
        state = SampleState(project_id="p1", project_path="/tmp/p", stage="PostProcess", status="running")
        qc = QCMetrics(refine3d_completed=True, current_resolution=3.8)
        history = [
            ActionHistoryEntry(run_id="r1", stage="Class2D", action_name="run_class2d", branch_name="main"),
            ActionHistoryEntry(run_id="r1", stage="InitialModel", action_name="run_init3d_or_class3d", branch_name="main"),
            ActionHistoryEntry(run_id="r1", stage="Refine3D", action_name="run_refine3d", branch_name="main"),
            ActionHistoryEntry(run_id="r1", stage="PostProcess", action_name="run_postprocess", branch_name="main"),
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
        self.assertIn("Class2D -> Init3D -> Refine3D -> Postprocess", md)

    def test_report_compact_when_mid3d_absent(self) -> None:
        state = SampleState(project_id="p2", project_path="/tmp/p2", stage="Class2D", status="running")
        qc = QCMetrics(class2d_completed=True)
        history = [ActionHistoryEntry(run_id="r2", stage="Class2D", action_name="run_class2d", branch_name="main")]
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


if __name__ == "__main__":
    unittest.main()

