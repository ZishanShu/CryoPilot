"""Policy tests for minimal upstream Picking -> Extract -> Class2D decisions."""

from __future__ import annotations

import unittest

from agent.policy_engine import PolicyEngine
from schemas.action import ActionName
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState


class TestUpstreamPolicy(unittest.TestCase):
    def test_picking_ready_triggers_run_autopick(self) -> None:
        pe = PolicyEngine(rules_path=None)
        state = SampleState(project_id="p1", project_path="/tmp/p", stage="CtfFind", status="running")
        qc = QCMetrics(motioncorr_completed=True, ctf_completed=True)
        action = pe.decide_next_action(state, qc)
        assert action is not None
        self.assertEqual(action.name, ActionName.RUN_AUTOPICK)

    def test_picked_or_extracted_state_triggers_extract_or_class2d(self) -> None:
        pe = PolicyEngine(rules_path=None)
        state_extract = SampleState(project_id="p1", project_path="/tmp/p", stage="AutoPick", status="running")
        qc_extract = QCMetrics(motioncorr_completed=True, ctf_completed=True)
        action_extract = pe.decide_next_action(state_extract, qc_extract)
        assert action_extract is not None
        self.assertEqual(action_extract.name, ActionName.RUN_EXTRACT)

        state_class2d = SampleState(
            project_id="p1", project_path="/tmp/p", stage="Extract", status="running", n_particles_picked=50000
        )
        qc_class2d = QCMetrics(motioncorr_completed=True, ctf_completed=True, class2d_completed=False)
        action_class2d = pe.decide_next_action(state_class2d, qc_class2d)
        assert action_class2d is not None
        self.assertEqual(action_class2d.name, ActionName.RUN_CLASS2D)

    def test_poor_class2d_can_trigger_human_review(self) -> None:
        pe = PolicyEngine(rules_path=None)
        state = SampleState(project_id="p1", project_path="/tmp/p", stage="Class2D", status="running")
        qc = QCMetrics(
            motioncorr_completed=True,
            ctf_completed=True,
            picking_completed=True,
            class2d_completed=True,
            retain_ratio_2d=0.05,
        )
        action = pe.decide_next_action(state, qc)
        assert action is not None
        self.assertEqual(action.name, ActionName.REQUEST_HUMAN_REVIEW)


if __name__ == "__main__":
    unittest.main()

