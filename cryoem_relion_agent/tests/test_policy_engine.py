"""Unit tests for policy engine core rules."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.policy_engine import PolicyEngine
from relion.project_watcher import ProjectWatcher
from schemas.action import ActionName
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState


class TestPolicyEngine(unittest.TestCase):
    """Test core decision paths."""

    def setUp(self) -> None:
        self.engine = PolicyEngine(rules_path=None)
        self.base_state = SampleState(
            project_id="p1",
            project_path="/tmp/proj",
            # Stage must be past AutoPick or ``decide_picking_stage`` returns ``run_autopick`` first.
            stage="Class2D",
            status="running",
            n_micrographs=1000,
            n_particles_picked=200000,
            n_particles_after_2d=100000,
            n_particles_after_3d=80000,
        )

    def test_preprocess_not_done(self) -> None:
        qc = QCMetrics(
            motioncorr_completed=False,
            ctf_completed=False,
            picking_completed=False,
            class2d_completed=False,
            init3d_completed=False,
            refine3d_completed=False,
        )
        action = self.engine.decide_next_action(self.base_state, qc)
        assert action is not None
        self.assertEqual(action.name, ActionName.RUN_IMPORT_PREPROCESS)

    def test_low_2d_retain_ratio_requests_human_review(self) -> None:
        qc = QCMetrics(
            motioncorr_completed=True,
            ctf_completed=True,
            picking_completed=True,
            class2d_completed=True,
            init3d_completed=False,
            refine3d_completed=False,
            retain_ratio_2d=0.05,
        )
        action = self.engine.decide_next_action(self.base_state, qc)
        assert action is not None
        self.assertEqual(action.name, ActionName.REQUEST_HUMAN_REVIEW)

    def test_3d_init_path(self) -> None:
        qc = QCMetrics(
            motioncorr_completed=True,
            ctf_completed=True,
            picking_completed=True,
            class2d_completed=True,
            init3d_completed=False,
            refine3d_completed=False,
            retain_ratio_2d=0.6,
        )
        action = self.engine.decide_next_action(self.base_state, qc)
        assert action is not None
        self.assertEqual(action.name, ActionName.RUN_INIT3D_OR_CLASS3D)

    def test_map_ready_prefers_structure_routing_default(self) -> None:
        state = self.base_state.model_copy()
        state.map_path = "/tmp/proj/PostProcess/map.mrc"
        state.sequence_path = "/tmp/proj/seq.fasta"
        qc = QCMetrics(
            motioncorr_completed=True,
            ctf_completed=True,
            picking_completed=True,
            class2d_completed=True,
            init3d_completed=True,
            refine3d_completed=True,
            ctf_refine_completed=True,
            polish_completed=True,
            postprocess_completed=True,
            current_resolution=3.8,
        )
        action = self.engine.decide_next_action(state, qc)
        self.assertIsNone(action)

    def test_map_ready_without_sequence_fallback_validation(self) -> None:
        state = self.base_state.model_copy()
        state.map_path = "/tmp/proj/PostProcess/map.mrc"
        state.sequence_path = None
        qc = QCMetrics(
            motioncorr_completed=True,
            ctf_completed=True,
            picking_completed=True,
            class2d_completed=True,
            init3d_completed=True,
            refine3d_completed=True,
            ctf_refine_completed=True,
            polish_completed=True,
            postprocess_completed=True,
            current_resolution=3.8,
        )
        action = self.engine.decide_next_action(state, qc)
        assert action is not None
        self.assertEqual(action.name, ActionName.RUN_VALIDATION)
        self.assertIn("missing_sequence", str(action.parameters.get("routing_reason")))

    def test_stage_prefers_refine3d_over_initialmodel(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "InitialModel/job001").mkdir(parents=True, exist_ok=True)
            (root / "InitialModel/job001/init.star").write_text("data_\ncompleted\n", encoding="utf-8")
            (root / "Refine3D/job002").mkdir(parents=True, exist_ok=True)
            (root / "Refine3D/job002/refined_map.mrc").write_text("fake map", encoding="utf-8")
            watcher = ProjectWatcher(root)
            self.assertEqual(watcher.detect_stage(), "Refine3D")

    def test_stage_class2d_when_only_class2d_has_output(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Class2D/job001").mkdir(parents=True, exist_ok=True)
            (root / "Class2D/job001/classes.star").write_text("data_\nloop_\n1@x.mrcs\n", encoding="utf-8")
            watcher = ProjectWatcher(root)
            self.assertEqual(watcher.detect_stage(), "Class2D")

    def test_stage_ignores_empty_late_stage_directory(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Class2D/job001").mkdir(parents=True, exist_ok=True)
            (root / "Class2D/job001/classes.star").write_text("data_\ncompleted\n", encoding="utf-8")
            (root / "Refine3D/job002").mkdir(parents=True, exist_ok=True)
            watcher = ProjectWatcher(root)
            self.assertEqual(watcher.detect_stage(), "Class2D")


if __name__ == "__main__":
    unittest.main()
