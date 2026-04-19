"""Tests for picker backend dry-run command generation."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from relion.relion_actions import run_autopick_action
from relion.scheme_runner import SchemeRunner


class TestPickerRouterActions(unittest.TestCase):
    def test_run_autopick_relion_log_dry_run(self) -> None:
        with TemporaryDirectory() as tmp:
            runner = SchemeRunner(Path(tmp) / "schemes.yaml")
            res = run_autopick_action(
                runner=runner,
                project_dir=Path(tmp),
                dry_run=True,
                backend="relion_log",
            )
            self.assertEqual(res.status, "dry_run")
            self.assertIn("relion_autopick", res.command or "")

    def test_run_autopick_topaz_dry_run(self) -> None:
        with TemporaryDirectory() as tmp:
            runner = SchemeRunner(Path(tmp) / "schemes.yaml")
            res = run_autopick_action(
                runner=runner,
                project_dir=Path(tmp),
                dry_run=True,
                backend="topaz",
            )
            self.assertEqual(res.status, "dry_run")
            self.assertIn("topaz", res.command or "")

    def test_run_autopick_cryolo_dry_run(self) -> None:
        with TemporaryDirectory() as tmp:
            runner = SchemeRunner(Path(tmp) / "schemes.yaml")
            res = run_autopick_action(
                runner=runner,
                project_dir=Path(tmp),
                dry_run=True,
                backend="cryolo",
            )
            self.assertEqual(res.status, "dry_run")
            self.assertIn("cryolo_predict.py", res.command or "")


if __name__ == "__main__":
    unittest.main()
