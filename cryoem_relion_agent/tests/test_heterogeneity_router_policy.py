"""Policy tests for minimal heterogeneity backend routing."""

from __future__ import annotations

import unittest

from agent.policy_engine import PolicyEngine
from schemas.action import ActionName
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState


class TestHeterogeneityRouterPolicy(unittest.TestCase):
    def _heterogeneity_case(self) -> tuple[SampleState, QCMetrics]:
        state = SampleState(project_id="p1", project_path="/tmp/p", stage="Class3D", status="running")
        qc = QCMetrics(
            motioncorr_completed=True,
            ctf_completed=True,
            picking_completed=True,
            class2d_completed=True,
            heterogeneity_warning=True,
        )
        return state, qc

    def test_default_heterogeneity_backend_human_review(self) -> None:
        pe = PolicyEngine(rules_path=None)
        state, qc = self._heterogeneity_case()
        action = pe.decide_next_action(state, qc)
        assert action is not None
        self.assertEqual(action.name, ActionName.REQUEST_HUMAN_REVIEW)
        self.assertEqual(action.parameters.get("heterogeneity_backend"), "human_review")

    def test_explicit_selection_switches_to_cryodrgn(self) -> None:
        pe = PolicyEngine(rules_path=None)
        state, qc = self._heterogeneity_case()
        action = pe.decide_next_action(state, qc, preferred_heterogeneity_backend="cryodrgn")
        assert action is not None
        self.assertEqual(action.name, ActionName.RUN_CRYODRGN)
        self.assertEqual(action.parameters.get("heterogeneity_backend"), "cryodrgn")

    def test_config_selection_switches_to_cryodrgn(self) -> None:
        pe = PolicyEngine(rules_path=None)
        pe.rules["heterogeneity"]["default_backend"] = "cryodrgn"
        state, qc = self._heterogeneity_case()
        action = pe.decide_next_action(state, qc)
        assert action is not None
        self.assertEqual(action.name, ActionName.RUN_CRYODRGN)
        self.assertEqual(action.parameters.get("heterogeneity_backend"), "cryodrgn")


if __name__ == "__main__":
    unittest.main()
