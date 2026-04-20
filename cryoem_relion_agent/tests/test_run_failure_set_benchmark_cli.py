"""Tests for the run_failure_set_benchmark CLI."""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.run_failure_set_benchmark import main as run_failure_main


class TestRunFailureSetBenchmarkCLI(unittest.TestCase):
    def test_cli_runs_defaults_and_persists(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = run_failure_main(
                    [
                        "--project-root",
                        str(root),
                        "--data-root",
                        str(root / "data"),
                        "--run-root",
                        str(root / "runs"),
                        "--case-root",
                        str(root / "cases_wd"),
                    ]
                )
            self.assertEqual(rc, 0)
            latest = root / ".cryoem_agent" / "experiments" / "e2_failure_set" / "latest.json"
            latest_md = root / ".cryoem_agent" / "experiments" / "e2_failure_set" / "latest.md"
            self.assertTrue(latest.is_file())
            self.assertTrue(latest_md.is_file())
            payload = json.loads(latest.read_text(encoding="utf-8"))
            self.assertEqual(payload["benchmark_name"], "e2_failure_set")
            self.assertEqual(len(payload["cases"]), 4)

    def test_cli_subset_of_sources(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = run_failure_main(
                    [
                        "--project-root",
                        str(root),
                        "--advice-sources",
                        "rule_only",
                        "llm",
                        "--data-root",
                        str(root / "data"),
                        "--run-root",
                        str(root / "runs"),
                        "--case-root",
                        str(root / "cases_wd"),
                        "--no-persist",
                    ]
                )
            self.assertEqual(rc, 0)
            # No persistence should have happened.
            self.assertFalse(
                (root / ".cryoem_agent" / "experiments" / "e2_failure_set" / "latest.json").exists()
            )
            # stdout must include both rule_only and llm entries.
            stdout = buf.getvalue()
            self.assertIn("rule_only", stdout)
            self.assertIn("llm", stdout)
            self.assertNotIn('"advice_source": "none"', stdout)


if __name__ == "__main__":
    unittest.main()
