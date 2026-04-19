"""Policy tests for minimal heterogeneity branch behavior."""

from __future__ import annotations

import unittest

from agent.policy_engine import PolicyEngine
from schemas.action import ActionName
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState


class TestHeterogeneityPolicy(unittest.TestCase):
    def test_heterogeneity_signal_triggers_human_review_branch(self) -> None:
        pe = PolicyEngine(rules_path=None)
        state = SampleState(
            project_id="p1",
            project_path="/tmp/p",
            stage="Class3D",
            status="running",
            n_particles_after_2d=100000,
        )
        qc = QCMetrics(
            motioncorr_completed=True,
            ctf_completed=True,
            picking_completed=True,
            class2d_completed=True,
            heterogeneity_warning=True,
        )
        action = pe.decide_next_action(state, qc)
        assert action is not None
        self.assertEqual(action.name, ActionName.REQUEST_HUMAN_REVIEW)
        self.assertEqual(action.parameters.get("branch_name"), "heterogeneity")
        self.assertIn("heterogeneity", action.reason.lower())


if __name__ == "__main__":
    unittest.main()

