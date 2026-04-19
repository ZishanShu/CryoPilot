"""Policy tests for minimal post-refinement chain decisions."""

from __future__ import annotations

import unittest

from agent.policy_engine import PolicyEngine
from schemas.action import ActionName
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState


class TestPostrefinePolicy(unittest.TestCase):
    def test_refine_done_triggers_ctf_refine(self) -> None:
        pe = PolicyEngine(rules_path=None)
        state = SampleState(project_id="p1", project_path="/tmp/p", stage="Refine3D", status="running")
        qc = QCMetrics(
            motioncorr_completed=True,
            ctf_completed=True,
            picking_completed=True,
            class2d_completed=True,
            init3d_completed=True,
            refine3d_completed=True,
            ctf_refine_completed=False,
        )
        action = pe.decide_ctf_refine_stage(state, qc)
        assert action is not None
        self.assertEqual(action.name, ActionName.RUN_CTF_REFINE)

    def test_ctf_refine_done_triggers_polish(self) -> None:
        pe = PolicyEngine(rules_path=None)
        state = SampleState(project_id="p1", project_path="/tmp/p", stage="CtfRefine", status="running")
        qc = QCMetrics(
            motioncorr_completed=True,
            ctf_completed=True,
            picking_completed=True,
            class2d_completed=True,
            init3d_completed=True,
            refine3d_completed=True,
            ctf_refine_completed=True,
            polish_completed=False,
        )
        action = pe.decide_polish_stage(state, qc)
        assert action is not None
        self.assertEqual(action.name, ActionName.RUN_POLISH)

    def test_polish_done_triggers_postprocess(self) -> None:
        pe = PolicyEngine(rules_path=None)
        state = SampleState(project_id="p1", project_path="/tmp/p", stage="Polish", status="running")
        qc = QCMetrics(
            motioncorr_completed=True,
            ctf_completed=True,
            picking_completed=True,
            class2d_completed=True,
            init3d_completed=True,
            refine3d_completed=True,
            ctf_refine_completed=True,
            polish_completed=True,
            postprocess_completed=False,
        )
        action = pe.decide_postprocess_stage(state, qc)
        assert action is not None
        self.assertEqual(action.name, ActionName.RUN_POSTPROCESS)


if __name__ == "__main__":
    unittest.main()
