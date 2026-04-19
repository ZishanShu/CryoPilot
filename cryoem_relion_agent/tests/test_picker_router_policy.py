"""Policy tests for minimal picker backend routing."""

from __future__ import annotations

import unittest

from agent.policy_engine import PolicyEngine
from schemas.action import ActionName
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState


class TestPickerRouterPolicy(unittest.TestCase):
    def test_default_picking_backend_relion_log(self) -> None:
        pe = PolicyEngine(rules_path=None)
        state = SampleState(project_id="p1", project_path="/tmp/p", stage="CtfFind", status="running")
        qc = QCMetrics(motioncorr_completed=True, ctf_completed=True)
        action = pe.decide_next_action(state, qc)
        assert action is not None
        self.assertEqual(action.name, ActionName.RUN_AUTOPICK)
        self.assertEqual(action.parameters.get("picking_backend"), "relion_log")

    def test_explicit_preferred_backend_topaz(self) -> None:
        pe = PolicyEngine(rules_path=None)
        state = SampleState(project_id="p1", project_path="/tmp/p", stage="CtfFind", status="running")
        qc = QCMetrics(motioncorr_completed=True, ctf_completed=True)
        action = pe.decide_next_action(state, qc, preferred_picking_backend="topaz")
        assert action is not None
        self.assertEqual(action.name, ActionName.RUN_AUTOPICK)
        self.assertEqual(action.parameters.get("picking_backend"), "topaz")

    def test_explicit_preferred_backend_cryolo(self) -> None:
        pe = PolicyEngine(rules_path=None)
        state = SampleState(project_id="p1", project_path="/tmp/p", stage="CtfFind", status="running")
        qc = QCMetrics(motioncorr_completed=True, ctf_completed=True)
        action = pe.decide_next_action(state, qc, preferred_picking_backend="cryolo")
        assert action is not None
        self.assertEqual(action.name, ActionName.RUN_AUTOPICK)
        self.assertEqual(action.parameters.get("picking_backend"), "cryolo")


if __name__ == "__main__":
    unittest.main()
