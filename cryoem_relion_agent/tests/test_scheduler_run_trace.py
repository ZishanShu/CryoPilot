"""Tests for scheduler run-level trace linking."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.scheduler import Scheduler


class TestSchedulerRunTrace(unittest.TestCase):
    def test_single_run_uses_stable_run_id_and_parent_links(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Class2D" / "job001").mkdir(parents=True)
            (root / "Class2D" / "job001" / "classes.star").write_text("data_\nloop_\n1\n", encoding="utf-8")

            sched = Scheduler(project_root=root)
            sched.action_executor.execute = lambda **_: {  # type: ignore[method-assign]
                "action_name": "run_class2d_cleanup",
                "status": "dry_run",
                "output_dir": str(root / "Class2D" / "job001"),
            }

            sched.run_next(dry_run=True)
            entries = sched.history_manager.retrieve_action_history()
            self.assertGreaterEqual(len(entries), 2)
            run_ids = {e.run_id for e in entries}
            self.assertEqual(len(run_ids), 1)
            executed = [e for e in entries if e.execution_status == "dry_run"]
            self.assertTrue(executed)
            self.assertTrue(bool(executed[-1].parent_decision_id))
            self.assertTrue(bool(executed[-1].previous_entry_id))


if __name__ == "__main__":
    unittest.main()

