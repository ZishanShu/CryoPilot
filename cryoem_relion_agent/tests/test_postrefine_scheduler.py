"""Scheduler tests for minimal post-refinement chain trace links."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.scheduler import Scheduler
from schemas.action import Action, ActionName
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState


class TestPostrefineScheduler(unittest.TestCase):
    def test_scheduler_records_postrefine_entries_same_run(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sched = Scheduler(project_root=root)
            state = SampleState(project_id="p1", project_path=str(root), stage="Refine3D", status="running")
            qc = QCMetrics(
                motioncorr_completed=True,
                ctf_completed=True,
                class2d_completed=True,
                init3d_completed=True,
                refine3d_completed=True,
                ctf_refine_completed=False,
            )
            sched.scan_project = lambda sequence_override=None, persist_state=True: (state, qc)  # type: ignore[method-assign]
            sched._resolve_next_action = lambda *_a, **_k: Action(  # type: ignore[method-assign]
                name=ActionName.RUN_CTF_REFINE,
                reason="postrefine test",
                parameters={"branch_name": "main"},
                priority=61,
            )
            sched.action_executor.execute = lambda **_: {  # type: ignore[method-assign]
                "action_name": "run_ctf_refine",
                "status": "dry_run",
                "output_dir": str(root / "CtfRefine" / "job001"),
            }
            _a, _r = sched.run_next(dry_run=True)
            items = sched.history_manager.retrieve_action_history()
            self.assertGreaterEqual(len(items), 2)
            self.assertEqual(len({x.run_id for x in items}), 1)
            decision = [x for x in items if x.execution_status == "decided"][-1]
            execution = [x for x in items if x.execution_status == "dry_run"][-1]
            self.assertEqual(execution.action_name, "run_ctf_refine")
            self.assertEqual(execution.parent_decision_id, decision.entry_id)
            self.assertTrue(bool(execution.previous_entry_id))
            self.assertEqual((execution.branch_name or "main"), "main")


if __name__ == "__main__":
    unittest.main()
