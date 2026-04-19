"""Scheduler tests for upstream chain trace-link integrity."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.scheduler import Scheduler
from schemas.action import Action, ActionName
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState


class TestUpstreamScheduler(unittest.TestCase):
    def test_scheduler_records_upstream_chain_in_same_run(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sched = Scheduler(project_root=root)
            state = SampleState(project_id="p1", project_path=str(root), stage="CtfFind", status="running")
            qc = QCMetrics(motioncorr_completed=True, ctf_completed=True)
            sched.scan_project = lambda sequence_override=None, persist_state=True: (state, qc)  # type: ignore[method-assign]
            sched._resolve_next_action = lambda *_a, **_k: Action(  # type: ignore[method-assign]
                name=ActionName.RUN_AUTOPICK,
                reason="upstream test",
                parameters={"branch_name": "main"},
                priority=78,
            )
            sched.action_executor.execute = lambda **_: {  # type: ignore[method-assign]
                "action_name": "run_autopick",
                "status": "dry_run",
                "output_dir": str(root / "AutoPick" / "job001"),
            }
            _a, _r = sched.run_next(dry_run=True)
            items = sched.history_manager.retrieve_action_history()
            self.assertGreaterEqual(len(items), 2)
            run_ids = {x.run_id for x in items}
            self.assertEqual(len(run_ids), 1)
            decision = [x for x in items if x.execution_status == "decided"][-1]
            execs = [x for x in items if x.execution_status == "dry_run"][-1]
            self.assertEqual(execs.parent_decision_id, decision.entry_id)
            self.assertTrue(bool(execs.previous_entry_id))
            self.assertEqual((execs.branch_name or "main"), "main")


if __name__ == "__main__":
    unittest.main()

