"""Stage smoke tests on tiny fixture projects."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.scheduler import Scheduler
from relion.project_watcher import ProjectWatcher
from tests.helpers.fixture_projects import build_fixture_project


class TestStageSmokeFixtures(unittest.TestCase):
    def test_stage_and_next_action_recommendations(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = [
                ("picking_ready", "CtfFind", "run_autopick"),
                ("extract_ready", "AutoPick", "run_extract"),
                ("class2d_ready", "Extract", "run_class2d"),
                ("init3d_ready", "Class2D", "run_init3d_or_class3d"),
                ("refine3d_ready", "InitialModel", "run_refine3d"),
            ]
            for name, expected_stage, expected_action in cases:
                proj = build_fixture_project(root, name)
                stage = ProjectWatcher(proj).detect_stage()
                self.assertEqual(stage, expected_stage, msg=f"fixture={name}")
                action = Scheduler(project_root=proj).decide_next()
                self.assertEqual(action.name.value, expected_action, msg=f"fixture={name}")

    def test_postprocess_ready_can_recommend_structure_modeling(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            proj = build_fixture_project(root, "postprocess_ready")
            fasta = proj / "demo.fasta"
            fasta.write_text(">A\nM\n", encoding="utf-8")
            stage = ProjectWatcher(proj).detect_stage()
            self.assertEqual(stage, "PostProcess")
            action = Scheduler(project_root=proj).decide_next(sequence_override=fasta)
            self.assertEqual(action.name.value, "run_modelangelo")


if __name__ == "__main__":
    unittest.main()

