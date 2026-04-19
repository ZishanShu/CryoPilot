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


if __name__ == "__main__":
    unittest.main()
