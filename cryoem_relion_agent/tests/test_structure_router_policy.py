"""Policy/scheduler-facing tests for minimal structure routing trigger semantics."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.history_manager import ActionHistoryManager
from agent.policy_engine import PolicyEngine
from app.scheduler import Scheduler
from schemas.action import ActionName
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState


class TestStructureRouterPolicy(unittest.TestCase):
    def test_map_ready_triggers_structure_router_decision_log(self) -> None:
        with TemporaryDirectory() as tmp:
            hm = ActionHistoryManager(Path(tmp) / "h.jsonl")
            pe = PolicyEngine(rules_path=None)
            pe.set_history_manager(hm)
            pe.set_trace_context(run_id="r1")
            state = SampleState(
                project_id="p1",
                project_path="/tmp/p",
                stage="PostProcess",
                status="running",
                map_path="/tmp/map.mrc",
                sequence_path="/tmp/seq.fasta",
            )
            qc = QCMetrics(
                motioncorr_completed=True,
                ctf_completed=True,
                picking_completed=True,
                class2d_completed=True,
                init3d_completed=True,
                refine3d_completed=True,
                current_resolution=3.8,
            )
            action = pe.decide_next_action(state, qc)
            self.assertIsNone(action)
            items = hm.retrieve_action_history()
            self.assertTrue(items)
            self.assertEqual(items[-1].action_name, "delegate_structure_router")
            self.assertEqual(items[-1].branch_name, "structure_router")

    def test_sequence_presence_influences_downstream_choice(self) -> None:
        sched = Scheduler(project_root=Path("/tmp"))
        state = SampleState(
            project_id="p1",
            project_path="/tmp/p",
            stage="PostProcess",
            status="running",
            map_path="/tmp/map.mrc",
            sequence_path="/tmp/seq.fasta",
        )
        qc = QCMetrics(
            motioncorr_completed=True,
            ctf_completed=True,
            picking_completed=True,
            class2d_completed=True,
            init3d_completed=True,
            refine3d_completed=True,
            current_resolution=3.8,
        )
        action_with_seq = sched._resolve_next_action(state, qc, None)
        self.assertEqual(action_with_seq.name, ActionName.RUN_MODELANGELO)

        no_seq = state.model_copy(update={"sequence_path": None})
        action_no_seq = sched._resolve_next_action(no_seq, qc, None)
        self.assertEqual(action_no_seq.name, ActionName.RUN_VALIDATION)


if __name__ == "__main__":
    unittest.main()

