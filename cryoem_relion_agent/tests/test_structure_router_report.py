"""Reporter compact summary tests for structure-routing branches."""

from __future__ import annotations

import unittest

from agent.reporter import Reporter
from schemas.action_history import ActionHistoryEntry
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState


class TestStructureRouterReport(unittest.TestCase):
    def test_report_includes_structure_routing_summary(self) -> None:
        state = SampleState(project_id="p1", project_path="/tmp/p", stage="PostProcess", status="running")
        qc = QCMetrics(refine3d_completed=True, current_resolution=3.6)
        history = [
            ActionHistoryEntry(run_id="r1", stage="PostProcess", action_name="delegate_structure_router", branch_name="structure_router"),
            ActionHistoryEntry(run_id="r1", stage="PostProcess", action_name="route_structure", branch_name="structure_router"),
            ActionHistoryEntry(run_id="r1", stage="PostProcess", action_name="run_modelangelo", branch_name="modelangelo"),
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
        self.assertIn("Branch: structure_router", md)
        self.assertIn("Branch: modelangelo", md)

    def test_report_compact_without_structure_routing(self) -> None:
        state = SampleState(project_id="p2", project_path="/tmp/p2", stage="Class2D", status="running")
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
        self.assertNotIn("Branch: structure_router", md)
        self.assertIn("Branch: main", md)


if __name__ == "__main__":
    unittest.main()

