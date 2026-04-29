"""Runtime portability diagnostics for backend executables and stale paths."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from typer.testing import CliRunner

from agent.backend_runtime import (
    check_path_access,
    preflight_backend,
    scan_stale_paths,
)
from app.main import app


class TestBackendRuntime(unittest.TestCase):
    def test_stale_root_paths_are_detected(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "config.yaml"
            cfg.write_text(
                "exe: /root/anaconda3/envs/model_angelo/bin/model_angelo\n"
                "weights: /root/share/szs/CryoAgent/model-angelo/public/model_angelo_weights\n",
                encoding="utf-8",
            )

            report = scan_stale_paths(root)

            paths = {item["path"] for item in report["findings"]}
            self.assertIn("/root/anaconda3/envs/model_angelo/bin/model_angelo", paths)
            self.assertIn("/root/share/szs/CryoAgent/model-angelo/public/model_angelo_weights", paths)
            reasons = {reason for item in report["findings"] for reason in item["stale_reasons"]}
            self.assertIn("under_root", reasons)
            self.assertIn("root_anaconda3", reasons)
            self.assertIn("root_share", reasons)

    def test_permission_error_becomes_not_accessible_status(self) -> None:
        with patch("agent.backend_runtime.Path.exists", side_effect=PermissionError("denied")):
            status = check_path_access("/tmp/blocked")

        self.assertTrue(status["not_accessible"])
        self.assertIn("PermissionError", status["error"])
        self.assertIsNone(status["exists"])

    def test_dry_run_allows_missing_executable_with_warning(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = preflight_backend(
                "topaz",
                project_root=root,
                output_dir=root / "out",
                dry_run=True,
                params={"executable": str(root / "missing_topaz")},
            )

            self.assertTrue(result["dry_run_available"])
            self.assertFalse(result["real_run_available"])
            self.assertTrue(result["available"])
            self.assertIn("topaz_executable", result["missing_dependencies"])
            self.assertTrue(any("dry_run_available" in w for w in result["warnings"]))

    def test_real_run_blocks_missing_executable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = preflight_backend(
                "topaz",
                project_root=root,
                output_dir=root / "out",
                dry_run=False,
                params={"executable": str(root / "missing_topaz")},
            )

            self.assertFalse(result["available"])
            self.assertFalse(result["real_run_available"])
            self.assertIn("topaz_executable", result["missing_dependencies"])
            self.assertIn("missing_runtime_dependencies", result["blocked_reason"])


class TestBackendRuntimeCLI(unittest.TestCase):
    def test_check_backends_writes_json_for_registered_runtime_backends(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            seq = root / "seq.fasta"
            seq.write_text(">a\nA\n", encoding="utf-8")
            out_json = root / "backends.json"
            out_md = root / "backends.md"

            res = runner.invoke(
                app,
                [
                    "check-backends",
                    "--project",
                    str(root),
                    "--sequence",
                    str(seq),
                    "--out-json",
                    str(out_json),
                    "--out-md",
                    str(out_md),
                ],
            )

            self.assertEqual(res.exit_code, 0, res.output)
            data = json.loads(out_json.read_text(encoding="utf-8"))
            for backend in ("modelangelo", "emodelx", "mica", "topaz", "cryodrgn"):
                self.assertIn(backend, data["backends"])
            self.assertTrue(out_md.read_text(encoding="utf-8").startswith("# CryoAgent backend runtime preflight"))

    def test_diagnose_environment_writes_stale_path_findings(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "local_config.yaml").write_text("old: /root/share/old_weights\n", encoding="utf-8")
            out_json = root / "diagnose.json"
            out_md = root / "diagnose.md"

            res = runner.invoke(
                app,
                [
                    "diagnose-environment",
                    "--project",
                    str(root),
                    "--out-json",
                    str(out_json),
                    "--out-md",
                    str(out_md),
                ],
            )

            self.assertEqual(res.exit_code, 0, res.output)
            data = json.loads(out_json.read_text(encoding="utf-8"))
            project_findings = data["project_stale_paths"]["findings"]
            self.assertTrue(any(item["path"] == "/root/share/old_weights" for item in project_findings))
            self.assertIn("stale_paths", data)


if __name__ == "__main__":
    unittest.main()
