"""Scheduler trace tests for picker backend metadata."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.scheduler import Scheduler
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState


class TestPickerRouterScheduler(unittest.TestCase):
    def test_scheduler_records_picker_backend_in_same_run_chain(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sched = Scheduler(project_root=root)
            state = SampleState(project_id="p1", project_path=str(root), stage="CtfFind", status="running")
            qc = QCMetrics(motioncorr_completed=True, ctf_completed=True)
            sched.scan_project = lambda sequence_override=None, persist_state=True: (state, qc)  # type: ignore[method-assign]

            action, _result = sched.run_next(dry_run=True)
            self.assertEqual(action.name.value, "run_autopick")
            self.assertIn(action.parameters.get("picking_backend"), {"relion_log", "topaz", "cryolo"})

            items = sched.history_manager.retrieve_action_history()
            self.assertGreaterEqual(len(items), 2)
            self.assertEqual(len({x.run_id for x in items}), 1)

            decision = [x for x in items if x.execution_status == "decided"][-1]
            execution = [x for x in items if x.action_name == "run_autopick"][-1]
            self.assertEqual(execution.parent_decision_id, decision.entry_id)
            self.assertTrue(bool(execution.previous_entry_id))
            self.assertIn("picking_backend=", execution.output_summary)


if __name__ == "__main__":
    unittest.main()
