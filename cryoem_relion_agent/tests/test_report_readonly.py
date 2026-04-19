"""Tests that generate-report path remains read-only for history."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.scheduler import Scheduler
from schemas.action_history import ActionHistoryEntry
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState


class TestReportReadonly(unittest.TestCase):
    def test_generate_report_does_not_append_history_by_default(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sched = Scheduler(project_root=root)
            state = SampleState(project_id="p1", project_path=str(root), stage="PostProcess", status="running")
            qc = QCMetrics(refine3d_completed=True, current_resolution=3.8)
            sched.scan_project = lambda sequence_override=None, persist_state=True: (state, qc)  # type: ignore[method-assign]
            sched.history_manager.add_action_to_history(
                ActionHistoryEntry(run_id="r0", stage="Class2D", action_name="run_class2d", branch_name="main")
            )
            before = len(sched.history_manager.retrieve_action_history())
            report = sched.generate_report()
            after = len(sched.history_manager.retrieve_action_history())
            self.assertEqual(before, after)
            self.assertIsNotNone(report.next_action)
            self.assertTrue(bool(report.stage_transition_summary))


if __name__ == "__main__":
    unittest.main()

