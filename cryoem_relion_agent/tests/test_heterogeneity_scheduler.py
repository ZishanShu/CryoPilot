"""Scheduler tests for heterogeneity branch trace links."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.scheduler import Scheduler
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState


class TestHeterogeneityScheduler(unittest.TestCase):
    def test_scheduler_records_heterogeneity_in_same_run_with_links(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sched = Scheduler(project_root=root)

            state = SampleState(project_id="p1", project_path=str(root), stage="Class3D", status="running")
            qc = QCMetrics(
                motioncorr_completed=True,
                ctf_completed=True,
                picking_completed=True,
                class2d_completed=True,
                heterogeneity_warning=True,
            )
            sched.scan_project = lambda sequence_override=None, persist_state=True: (state, qc)  # type: ignore[method-assign]
            sched.action_executor.execute = lambda **_: {  # type: ignore[method-assign]
                "action_name": "request_human_review",
                "status": "dry_run",
                "output_dir": None,
            }

            _action, _result = sched.run_next(dry_run=True)
            entries = sched.history_manager.retrieve_action_history()
            self.assertGreaterEqual(len(entries), 2)
            run_ids = {e.run_id for e in entries}
            self.assertEqual(len(run_ids), 1)
            self.assertTrue(all((e.branch_name or "main") == "heterogeneity" for e in entries))

            decision = [e for e in entries if e.execution_status == "decided"][-1]
            execution = [e for e in entries if e.execution_status == "dry_run"][-1]
            self.assertEqual(execution.parent_decision_id, decision.entry_id)
            self.assertTrue(bool(execution.previous_entry_id))


if __name__ == "__main__":
    unittest.main()

