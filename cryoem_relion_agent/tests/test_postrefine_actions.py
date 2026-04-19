"""Tests for post-refinement action wrappers dry-run behavior."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from relion.relion_actions import run_ctf_refine_action, run_polish_action
from relion.scheme_runner import SchemeRunner


class TestPostrefineActions(unittest.TestCase):
    def test_run_ctf_refine_action_dry_run(self) -> None:
        with TemporaryDirectory() as tmp:
            runner = SchemeRunner(Path(tmp) / "schemes.yaml")
            res = run_ctf_refine_action(runner=runner, project_dir=Path(tmp), dry_run=True)
            self.assertEqual(res.status, "dry_run")
            self.assertIn("relion_ctf_refine", res.command or "")

    def test_run_polish_action_dry_run(self) -> None:
        with TemporaryDirectory() as tmp:
            runner = SchemeRunner(Path(tmp) / "schemes.yaml")
            res = run_polish_action(runner=runner, project_dir=Path(tmp), dry_run=True)
            self.assertEqual(res.status, "dry_run")
            self.assertIn("relion_particle_polish", res.command or "")


if __name__ == "__main__":
    unittest.main()
