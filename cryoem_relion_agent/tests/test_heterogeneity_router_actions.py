"""Tests for heterogeneity router action wrappers."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from relion.relion_actions import RelionActionExecutor, run_cryodrgn_action
from relion.scheme_runner import SchemeRunner
from schemas.action import Action, ActionName


class TestHeterogeneityRouterActions(unittest.TestCase):
    def test_run_cryodrgn_action_dry_run(self) -> None:
        with TemporaryDirectory() as tmp:
            runner = SchemeRunner(Path(tmp) / "schemes.yaml")
            res = run_cryodrgn_action(
                runner=runner,
                project_dir=Path(tmp),
                dry_run=True,
            )
            self.assertEqual(res.status, "dry_run")
            self.assertIn("cryodrgn", res.command or "")

    def test_human_review_path_remains_valid(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            executor = RelionActionExecutor(root / "schemes.yaml")
            action = Action(
                name=ActionName.REQUEST_HUMAN_REVIEW,
                reason="heterogeneity fallback",
                parameters={"branch_name": "heterogeneity", "heterogeneity_backend": "human_review"},
                priority=95,
            )
            result = executor.execute(action=action, project_root=root, dry_run=True)
            self.assertEqual(result.get("action_name"), "request_human_review")
            self.assertEqual(result.get("status"), "dry_run")


if __name__ == "__main__":
    unittest.main()
