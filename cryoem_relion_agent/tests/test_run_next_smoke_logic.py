"""Smoke tests for run-next style decision/execution flow on fixtures."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.scheduler import Scheduler
from tests.helpers.fixture_projects import build_fixture_project


class TestRunNextSmokeLogic(unittest.TestCase):
    def test_run_next_dry_run_on_upstream_midchain_fixtures(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = [
                ("picking_ready", "run_autopick"),
                ("extract_ready", "run_extract"),
                ("class2d_ready", "run_class2d"),
                ("init3d_ready", "run_init3d_or_class3d"),
                ("refine3d_ready", "run_refine3d"),
            ]
            for name, expected_action in cases:
                proj = build_fixture_project(root, name)
                action, execution = Scheduler(project_root=proj).run_next(dry_run=True)
                self.assertEqual(action.name.value, expected_action, msg=f"fixture={name}")
                self.assertIsInstance(execution, dict)
                self.assertEqual(execution.get("status"), "dry_run", msg=f"fixture={name}")

    def test_run_next_dry_run_postprocess_fixture(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            proj = build_fixture_project(root, "postprocess_ready")
            fasta = proj / "demo.fasta"
            fasta.write_text(">A\nM\n", encoding="utf-8")
            action, execution = Scheduler(project_root=proj).run_next(
                dry_run=True,
                sequence_override=fasta,
            )
            self.assertEqual(action.name.value, "run_modelangelo")
            self.assertEqual(execution.get("status"), "dry_run")


if __name__ == "__main__":
    unittest.main()

