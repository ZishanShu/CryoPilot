"""Unit tests for execution backbone (ExecutionResult + SchemeRunner + action wrappers)."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from relion.relion_actions import run_autopick_action
from relion.scheme_runner import SchemeRunner
from schemas.execution_result import ExecutionResult


class TestExecutionResult(unittest.TestCase):
    def test_to_dict_and_success_property(self) -> None:
        er = ExecutionResult(
            action_name="run_x",
            command="echo hi",
            dry_run=True,
            status="dry_run",
            metadata={"k": "v"},
        )
        data = er.to_dict()
        self.assertEqual(data["action_name"], "run_x")
        self.assertEqual(data["status"], "dry_run")
        self.assertTrue(er.is_success)


class TestSchemeRunnerBackbone(unittest.TestCase):
    def test_run_scheme_dry_run_returns_structured_result(self) -> None:
        with TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "schemes.yaml"
            cfg.write_text("schemes:\n  run_demo: \"echo hello\"\n", encoding="utf-8")
            runner = SchemeRunner(cfg)
            result = runner.run_scheme("run_demo", dry_run=True)
            self.assertEqual(result.status, "dry_run")
            self.assertIn("echo hello", result.command or "")
            self.assertEqual(result.return_code, 0)

    def test_run_command_subprocess_failure_graceful(self) -> None:
        with TemporaryDirectory() as tmp:
            runner = SchemeRunner(Path(tmp) / "none.yaml")
            with patch("relion.scheme_runner.subprocess.run", side_effect=FileNotFoundError("missing")):
                result = runner.run_command(
                    action_name="run_missing",
                    argv=["definitely_missing_binary"],
                    dry_run=False,
                )
            self.assertEqual(result.status, "failed")
            self.assertIn("command not found", result.error_message or "")
            self.assertEqual(result.return_code, 127)


class TestRelionActionWrappers(unittest.TestCase):
    def test_autopick_action_dry_run_has_command(self) -> None:
        with TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "schemes.yaml"
            cfg.write_text("schemes: {}\n", encoding="utf-8")
            runner = SchemeRunner(cfg)
            result = run_autopick_action(
                runner=runner,
                project_dir=Path(tmp),
                output_dir=Path(tmp) / "AutoPick" / "job001",
                dry_run=True,
            )
            self.assertEqual(result.status, "dry_run")
            self.assertIn("relion_autopick", result.command or "")

    def test_run_scheme_failed_return_code(self) -> None:
        with TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "schemes.yaml"
            cfg.write_text("schemes:\n  run_demo: \"echo hello\"\n", encoding="utf-8")
            runner = SchemeRunner(cfg)
            completed = MagicMock()
            completed.returncode = 2
            completed.stdout = "o"
            completed.stderr = "e"
            with patch("relion.scheme_runner.subprocess.run", return_value=completed):
                result = runner.run_scheme("run_demo", dry_run=False)
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.return_code, 2)


if __name__ == "__main__":
    unittest.main()

