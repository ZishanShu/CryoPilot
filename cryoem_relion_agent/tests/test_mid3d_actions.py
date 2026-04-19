"""Tests for minimal mid-3D action wrappers dry-run behavior."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from relion.relion_actions import run_init3d_or_class3d_action, run_postprocess_action, run_refine3d_action
from relion.scheme_runner import SchemeRunner


class TestMid3DActions(unittest.TestCase):
    def test_run_init3d_or_class3d_action_dry_run(self) -> None:
        with TemporaryDirectory() as tmp:
            runner = SchemeRunner(Path(tmp) / "schemes.yaml")
            res = run_init3d_or_class3d_action(runner=runner, project_dir=Path(tmp), dry_run=True)
            self.assertEqual(res.status, "dry_run")
            self.assertIn("relion_refine", res.command or "")

    def test_run_refine3d_action_dry_run(self) -> None:
        with TemporaryDirectory() as tmp:
            runner = SchemeRunner(Path(tmp) / "schemes.yaml")
            res = run_refine3d_action(runner=runner, project_dir=Path(tmp), dry_run=True)
            self.assertEqual(res.status, "dry_run")
            self.assertIn("relion_refine", res.command or "")

    def test_run_postprocess_action_dry_run(self) -> None:
        with TemporaryDirectory() as tmp:
            runner = SchemeRunner(Path(tmp) / "schemes.yaml")
            res = run_postprocess_action(runner=runner, project_dir=Path(tmp), dry_run=True)
            self.assertEqual(res.status, "dry_run")
            self.assertIn("relion_postprocess", res.command or "")


if __name__ == "__main__":
    unittest.main()

