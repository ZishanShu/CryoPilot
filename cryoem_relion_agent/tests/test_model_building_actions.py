"""Dry-run / placeholder execution for optional model-building backends."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from relion.relion_actions import RelionActionExecutor
from schemas.action import Action, ActionName


class TestModelBuildingPlaceholderActions(unittest.TestCase):
    def test_run_emodelx_preflight_blocked_no_subprocess(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            schemes = root / "schemes.yaml"
            schemes.write_text("schemes:\n  run_emodelx: \"echo\"\n", encoding="utf-8")
            ex = RelionActionExecutor(schemes)
            action = Action(
                name=ActionName.RUN_EMODELX,
                reason="test",
                parameters={
                    "model_building_backend": "emodelx",
                    "model_building_availability": "unavailable",
                    "model_building_execution_mode": "unavailable",
                    "model_building_blocked_reason": "missing_dependency: emodelx_weights",
                },
                priority=80,
            )
            with patch("relion.emodelx_runner.subprocess.run") as mock_run:
                out = ex.execute(action=action, project_root=root, dry_run=False, state=None)
            mock_run.assert_not_called()
            self.assertEqual(out.get("status"), "preflight_blocked")
            self.assertEqual(out.get("execution_mode"), "unavailable")
            self.assertEqual(out.get("model_building_backend"), "emodelx")
            self.assertIsInstance(out.get("model_building_preflight"), dict)
            self.assertTrue(bool(out.get("error_message")))

    def test_run_emodelx_real_exec_success(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "EModelX"
            out_dir = root / "emodelx_out"
            repo.mkdir(parents=True)
            (repo / "run.py").write_text("print('ok')", encoding="utf-8")
            action = Action(
                name=ActionName.RUN_EMODELX,
                reason="test",
                parameters={
                    "map_path": str(root / "m.mrc"),
                    "sequence_path": str(root / "s.fasta"),
                    "output_dir": str(out_dir),
                    "emodelx_repo_dir": str(repo),
                    "emodelx_protocol": "temp_free",
                    "model_building_backend": "emodelx",
                    "model_building_availability": "available",
                    "model_building_execution_mode": "real",
                    "model_building_preflight": {
                        "backend_name": "emodelx",
                        "available": True,
                        "execution_mode": "real",
                        "protocol": "temp_free",
                        "repo_path": str(repo),
                        "run_py_path": str(repo / "run.py"),
                        "python_path": "/usr/bin/python3",
                        "output_path": str(out_dir),
                    },
                },
                priority=80,
            )
            (root / "m.mrc").write_text("map", encoding="utf-8")
            (root / "s.fasta").write_text(">a\nA", encoding="utf-8")
            ex = RelionActionExecutor(root / "schemes.yaml")

            def _fake_run(cmd, **kwargs):
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "result.cif").write_text("cif", encoding="utf-8")
                p = MagicMock()
                p.returncode = 0
                p.stdout = "ok"
                p.stderr = ""
                return p

            with patch("relion.emodelx_runner.subprocess.run", side_effect=_fake_run) as mock_run:
                out = ex.execute(action=action, project_root=root, dry_run=False, state=None)
            mock_run.assert_called_once()
            self.assertEqual(out.get("status"), "completed")
            self.assertTrue(out.get("success"))
            self.assertEqual(out.get("execution_mode"), "real")
            self.assertEqual(out.get("cwd"), str(repo))
            self.assertEqual(out.get("output_dir"), str(out_dir))
            self.assertEqual(out.get("model_building_backend"), "emodelx")
            self.assertIn("--protocol", out.get("command") or "")
            self.assertEqual(out.get("repo_path"), str(repo))

    def test_run_emodelx_no_artifacts_not_completed(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "EModelX"
            out_dir = root / "emodelx_out"
            repo.mkdir(parents=True)
            (repo / "run.py").write_text("print('ok')", encoding="utf-8")
            action = Action(
                name=ActionName.RUN_EMODELX,
                reason="test",
                parameters={
                    "map_path": str(root / "m.mrc"),
                    "sequence_path": str(root / "s.fasta"),
                    "output_dir": str(out_dir),
                    "emodelx_repo_dir": str(repo),
                    "model_building_backend": "emodelx",
                    "model_building_availability": "available",
                    "model_building_execution_mode": "real",
                    "model_building_preflight": {
                        "backend_name": "emodelx",
                        "available": True,
                        "execution_mode": "real",
                        "protocol": "temp_free",
                        "repo_path": str(repo),
                        "run_py_path": str(repo / "run.py"),
                        "python_path": "/usr/bin/python3",
                        "output_path": str(out_dir),
                    },
                },
                priority=80,
            )
            (root / "m.mrc").write_text("map", encoding="utf-8")
            (root / "s.fasta").write_text(">a\nA", encoding="utf-8")
            ex = RelionActionExecutor(root / "schemes.yaml")

            def _fake_run(cmd, **kwargs):
                p = MagicMock()
                p.returncode = 0
                p.stdout = "ok"
                p.stderr = ""
                return p

            with patch("relion.emodelx_runner.subprocess.run", side_effect=_fake_run):
                out = ex.execute(action=action, project_root=root, dry_run=False, state=None)
            self.assertEqual(out.get("status"), "failed")
            self.assertFalse(out.get("success"))

    def test_run_mica_real_exec_success(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "MICA"
            out_dir = root / "mica_out"
            repo.mkdir(parents=True)
            (repo / "run.py").write_text("print('ok')", encoding="utf-8")
            ex = RelionActionExecutor(root / "schemes.yaml")
            action = Action(
                name=ActionName.RUN_MICA,
                reason="test",
                parameters={
                    "map_path": str(root / "m.mrc"),
                    "sequence_path": str(root / "s.fasta"),
                    "af3_dir": str(root / "af3_results"),
                    "output_dir": str(out_dir),
                    "mica_repo_dir": str(repo),
                    "model_building_backend": "mica",
                    "model_building_availability": "available",
                    "model_building_execution_mode": "real",
                    "model_building_preflight": {
                        "backend_name": "mica",
                        "available": True,
                        "execution_mode": "real",
                        "repo_path": str(repo),
                        "run_py_path": str(repo / "run.py"),
                        "python_path": "/usr/bin/python3",
                        "af3_dir": str(root / "af3_results"),
                        "output_path": str(out_dir),
                    },
                },
                priority=90,
            )
            (root / "m.mrc").write_text("map", encoding="utf-8")
            (root / "s.fasta").write_text(">a\nA", encoding="utf-8")
            (root / "af3_results").mkdir(parents=True)
            (root / "af3_results" / "model.cif").write_text("cif", encoding="utf-8")

            def _fake_run(cmd, **kwargs):
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "final_model.pdb").write_text("pdb", encoding="utf-8")
                p = MagicMock()
                p.returncode = 0
                p.stdout = "ok"
                p.stderr = ""
                return p

            with patch("relion.mica_runner.subprocess.run", side_effect=_fake_run) as mock_run:
                out = ex.execute(action=action, project_root=root, dry_run=False, state=None)
            mock_run.assert_called_once()
            self.assertEqual(out.get("status"), "completed")
            self.assertTrue(out.get("success"))
            self.assertEqual(out.get("execution_mode"), "real")
            self.assertEqual(out.get("model_building_backend"), "mica")
            self.assertEqual(out.get("cwd"), str(repo))
            self.assertEqual(out.get("af3_dir"), str(root / "af3_results"))

    def test_run_mica_no_artifacts_not_completed(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "MICA"
            out_dir = root / "mica_out"
            repo.mkdir(parents=True)
            (repo / "run.py").write_text("print('ok')", encoding="utf-8")
            ex = RelionActionExecutor(root / "schemes.yaml")
            action = Action(
                name=ActionName.RUN_MICA,
                reason="test",
                parameters={
                    "map_path": str(root / "m.mrc"),
                    "sequence_path": str(root / "s.fasta"),
                    "af3_dir": str(root / "af3_results"),
                    "output_dir": str(out_dir),
                    "mica_repo_dir": str(repo),
                    "model_building_backend": "mica",
                    "model_building_availability": "available",
                    "model_building_execution_mode": "real",
                    "model_building_preflight": {
                        "backend_name": "mica",
                        "available": True,
                        "execution_mode": "real",
                        "repo_path": str(repo),
                        "run_py_path": str(repo / "run.py"),
                        "python_path": "/usr/bin/python3",
                        "af3_dir": str(root / "af3_results"),
                        "output_path": str(out_dir),
                    },
                },
                priority=90,
            )
            (root / "m.mrc").write_text("map", encoding="utf-8")
            (root / "s.fasta").write_text(">a\nA", encoding="utf-8")
            (root / "af3_results").mkdir(parents=True)
            (root / "af3_results" / "model.cif").write_text("cif", encoding="utf-8")

            def _fake_run(cmd, **kwargs):
                p = MagicMock()
                p.returncode = 0
                p.stdout = "ok"
                p.stderr = ""
                return p

            with patch("relion.mica_runner.subprocess.run", side_effect=_fake_run):
                out = ex.execute(action=action, project_root=root, dry_run=False, state=None)
            self.assertEqual(out.get("status"), "failed")
            self.assertFalse(out.get("success"))

    def test_run_mica_preflight_blocked_no_subprocess(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            schemes = root / "schemes.yaml"
            schemes.write_text("schemes:\n  run_mica: \"echo should_not_run\"\n", encoding="utf-8")
            ex = RelionActionExecutor(schemes)
            action = Action(
                name=ActionName.RUN_MICA,
                reason="test",
                parameters={
                    "map_path": str(root / "m.mrc"),
                    "sequence_path": str(root / "s.fasta"),
                    "model_building_backend": "mica",
                    "model_building_availability": "unavailable",
                    "model_building_execution_mode": "unavailable",
                    "model_building_blocked_reason": "missing_required_inputs: af3_dir",
                },
                priority=90,
            )
            with patch("relion.mica_runner.subprocess.run") as mock_run:
                out = ex.execute(action=action, project_root=root, dry_run=False, state=None)
            mock_run.assert_not_called()
            self.assertEqual(out.get("status"), "preflight_blocked")
            self.assertEqual(out.get("execution_mode"), "unavailable")
            self.assertIn("missing_required_inputs", str(out.get("model_building_blocked_reason") or ""))


if __name__ == "__main__":
    unittest.main()
