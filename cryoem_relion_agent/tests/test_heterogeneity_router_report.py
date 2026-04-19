"""Reporter tests for compact heterogeneity backend summary."""

from __future__ import annotations

import unittest

from agent.reporter import Reporter
from schemas.action import Action, ActionName
from schemas.action_history import ActionHistoryEntry
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState


class TestHeterogeneityRouterReport(unittest.TestCase):
    def test_report_includes_heterogeneity_backend_summary(self) -> None:
        state = SampleState(project_id="p1", project_path="/tmp/p", stage="Class3D", status="running")
        qc = QCMetrics(class2d_completed=True, heterogeneity_warning=True)
        next_action = Action(
            name=ActionName.RUN_CRYODRGN,
            reason="heterogeneity detected",
            parameters={"branch_name": "heterogeneity", "heterogeneity_backend": "cryodrgn"},
            priority=94,
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
        self.assertIn("Heterogeneity backend: cryodrgn", md)

    def test_report_compact_when_backend_absent(self) -> None:
        state = SampleState(project_id="p1", project_path="/tmp/p", stage="Class3D", status="running")
        qc = QCMetrics(class2d_completed=True)
        history = [ActionHistoryEntry(run_id="r1", stage="Class3D", action_name="request_human_review", branch_name="heterogeneity")]
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
        self.assertNotIn("Heterogeneity backend:", md)


if __name__ == "__main__":
    unittest.main()
