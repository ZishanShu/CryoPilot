"""Tests for linked action-history fields and backward compatibility."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.history_manager import ActionHistoryManager
from schemas.action_history import ActionHistoryEntry


class TestActionHistoryLinks(unittest.TestCase):
    def test_entry_id_exists(self) -> None:
        item = ActionHistoryEntry(stage="Class2D", action_name="run_class2d_cleanup")
        self.assertTrue(bool(item.entry_id))

    def test_run_id_exists_when_expected(self) -> None:
        item = ActionHistoryEntry(run_id="run-123", stage="Class2D", action_name="run_class2d_cleanup")
        self.assertEqual(item.run_id, "run-123")

    def test_old_style_entries_still_load(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "history.jsonl"
            p.write_text(
                '{"timestamp":"2026-01-01T00:00:00Z","stage":"Class2D","action_name":"run_class2d_cleanup","decision_reason":"x","qc_snapshot":{},"execution_status":"decided","output_summary":"y"}\n',
                encoding="utf-8",
            )
            hm = ActionHistoryManager(p)
            items = hm.retrieve_action_history()
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].branch_name, "main")
            self.assertIsNone(items[0].run_id)


if __name__ == "__main__":
    unittest.main()

