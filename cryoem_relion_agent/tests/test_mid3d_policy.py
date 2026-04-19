"""Policy tests for minimal mid-3D chain decisions."""

from __future__ import annotations

import unittest

from agent.policy_engine import PolicyEngine
from schemas.action import ActionName
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState


class TestMid3DPolicy(unittest.TestCase):
    def test_class2d_ready_triggers_init3d_or_class3d(self) -> None:
        pe = PolicyEngine(rules_path=None)
        state = SampleState(
            project_id="p1",
            project_path="/tmp/p",
            stage="Class2D",
            status="running",
            n_particles_after_2d=80000,
        )
        qc = QCMetrics(
            motioncorr_completed=True,
            ctf_completed=True,
            picking_completed=True,
            class2d_completed=True,
            init3d_completed=False,
        )
        action = pe.decide_next_action(state, qc)
        assert action is not None
        self.assertEqual(action.name, ActionName.RUN_INIT3D_OR_CLASS3D)

    def test_3d_ready_triggers_refine3d(self) -> None:
        pe = PolicyEngine(rules_path=None)
        state = SampleState(project_id="p1", project_path="/tmp/p", stage="InitialModel", status="running")
        qc = QCMetrics(
            motioncorr_completed=True,
            ctf_completed=True,
            picking_completed=True,
            class2d_completed=True,
            init3d_completed=True,
            refine3d_completed=False,
        )
        action = pe.decide_next_action(state, qc)
        assert action is not None
        self.assertEqual(action.name, ActionName.RUN_REFINE3D)

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
            current_resolution=3.8,
        )
        action = pe.decide_postprocess_stage(state, qc)
        assert action is not None
        self.assertEqual(action.name, ActionName.RUN_POSTPROCESS)

    def test_postprocess_ready_can_surface_structure_router(self) -> None:
        pe = PolicyEngine(rules_path=None)
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
            current_resolution=3.5,
        )
        action = pe.decide_next_action(state, qc)
        self.assertIsNone(action)


if __name__ == "__main__":
    unittest.main()

