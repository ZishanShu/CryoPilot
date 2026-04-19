"""Tests for action history manager and integration points."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.history_manager import ActionHistoryManager
from agent.policy_engine import PolicyEngine
from app.scheduler import Scheduler
from schemas.action import ActionName
from schemas.action_history import ActionHistoryEntry
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState


class TestActionHistoryManager(unittest.TestCase):
    def test_add_retrieve_and_summarize(self) -> None:
        with TemporaryDirectory() as tmp:
            hf = Path(tmp) / "h.jsonl"
            hm = ActionHistoryManager(hf)
            ok = hm.add_action_to_history(
                ActionHistoryEntry(
                    run_id="test-run-1",
                    stage="Class2D",
                    action_name="run_class2d_cleanup",
                    decision_reason="2d not done",
                    execution_status="decided",
                )
            )
            self.assertTrue(ok)
            entries = hm.retrieve_action_history()
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].action_name, "run_class2d_cleanup")
            self.assertTrue(bool(entries[0].entry_id))
            self.assertTrue(bool(entries[0].run_id))
            self.assertIn("entries=1", hm.summarize_history())

    def test_backward_compat_old_style_entries_can_load(self) -> None:
        with TemporaryDirectory() as tmp:
            hf = Path(tmp) / "h.jsonl"
            hf.write_text(
                '{"timestamp":"2026-01-01T00:00:00Z","stage":"Class2D","action_name":"run_class2d_cleanup","decision_reason":"x","qc_snapshot":{},"execution_status":"decided","output_summary":"y"}\n',
                encoding="utf-8",
            )
            hm = ActionHistoryManager(hf)
            items = hm.retrieve_action_history()
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].branch_name, "main")

    def test_get_by_run_and_branch(self) -> None:
        with TemporaryDirectory() as tmp:
            hf = Path(tmp) / "h.jsonl"
            hm = ActionHistoryManager(hf)
            rid = "run123"
            hm.add_action_to_history(
                ActionHistoryEntry(run_id=rid, branch_name="main", stage="Class2D", action_name="run_class2d_cleanup")
            )
            hm.add_action_to_history(
                ActionHistoryEntry(run_id=rid, branch_name="heterogeneity", stage="Class2D", action_name="request_human_review")
            )
            self.assertEqual(len(hm.get_history_by_run(rid)), 2)
            self.assertEqual(len(hm.get_history_by_branch("heterogeneity", run_id=rid)), 1)
            self.assertIn("Class2D", hm.summarize_stage_transitions(run_id=rid))
            self.assertIn("heterogeneity", hm.summarize_branches(run_id=rid))


class TestPolicyDecisionHistory(unittest.TestCase):
    def test_policy_logs_decision_when_history_attached(self) -> None:
        with TemporaryDirectory() as tmp:
            hm = ActionHistoryManager(Path(tmp) / "history.jsonl")
            pe = PolicyEngine(rules_path=None)
            pe.set_history_manager(hm)
            state = SampleState(project_id="p1", project_path="/tmp/p", stage="Import", status="running")
            qc = QCMetrics(motioncorr_completed=False, ctf_completed=False)
            action = pe.decide_next_action(state, qc)
            assert action is not None
            self.assertEqual(action.name, ActionName.RUN_IMPORT_PREPROCESS)
            entries = hm.retrieve_action_history()
            self.assertGreaterEqual(len(entries), 1)
            self.assertEqual(entries[-1].execution_status, "decided")


class TestSchedulerHistoryIntegration(unittest.TestCase):
    def test_run_next_records_execution_history(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Class2D" / "job001").mkdir(parents=True)
            (root / "Class2D" / "job001" / "classes.star").write_text("data_\nloop_\n1\n", encoding="utf-8")

            sched = Scheduler(project_root=root)
            # Keep it isolated: avoid real external calls.
            sched.action_executor.execute = lambda **_: {  # type: ignore[method-assign]
                "action_name": "run_class2d_cleanup",
                "status": "dry_run",
                "output_dir": str(root / "Class2D" / "job001"),
            }
            _action, _result = sched.run_next(dry_run=True)
            entries = sched.history_manager.retrieve_action_history()
            self.assertGreaterEqual(len(entries), 1)
            self.assertTrue(any(e.execution_status in {"dry_run", "decided"} for e in entries))
            run_ids = {e.run_id for e in entries}
            self.assertEqual(len(run_ids), 1)
            executed = [e for e in entries if e.execution_status == "dry_run"]
            self.assertTrue(len(executed) >= 1)
            self.assertTrue(bool(executed[-1].parent_decision_id))

    def test_report_contains_action_history_section(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Class2D" / "job001").mkdir(parents=True)
            (root / "Class2D" / "job001" / "classes.star").write_text("data_\nloop_\n1\n", encoding="utf-8")
            sched = Scheduler(project_root=root)
            sched.history_manager.add_action_to_history(
                ActionHistoryEntry(
                    stage="Class2D",
                    action_name="run_class2d_cleanup",
                    decision_reason="2D classification not completed.",
                    qc_snapshot={},
                    execution_status="dry_run",
                    output_summary="scheduler run_next",
                )
            )
            report = sched.generate_report()
            md = report.to_markdown()
            self.assertIn("## Action History", md)
            self.assertIn("## Stage Transitions", md)
            self.assertIn("## Branch History", md)
            self.assertGreaterEqual(len(report.action_history), 1)


if __name__ == "__main__":
    unittest.main()

