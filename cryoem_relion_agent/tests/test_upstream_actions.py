"""Tests for upstream RELION action wrappers dry-run behavior."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from relion.relion_actions import run_autopick_action, run_class2d_action, run_extract_action
from relion.scheme_runner import SchemeRunner


class TestUpstreamActions(unittest.TestCase):
    def test_run_autopick_action_dry_run(self) -> None:
        with TemporaryDirectory() as tmp:
            runner = SchemeRunner(Path(tmp) / "schemes.yaml")
            res = run_autopick_action(runner=runner, project_dir=Path(tmp), dry_run=True)
            self.assertEqual(res.status, "dry_run")
            self.assertIn("relion_autopick", res.command or "")

    def test_run_extract_action_dry_run(self) -> None:
        with TemporaryDirectory() as tmp:
            runner = SchemeRunner(Path(tmp) / "schemes.yaml")
            res = run_extract_action(runner=runner, project_dir=Path(tmp), dry_run=True)
            self.assertEqual(res.status, "dry_run")
            self.assertIn("relion_preprocess", res.command or "")

    def test_run_class2d_action_dry_run(self) -> None:
        with TemporaryDirectory() as tmp:
            runner = SchemeRunner(Path(tmp) / "schemes.yaml")
            res = run_class2d_action(runner=runner, project_dir=Path(tmp), dry_run=True)
            self.assertEqual(res.status, "dry_run")
            self.assertIn("relion_refine", res.command or "")


if __name__ == "__main__":
    unittest.main()

