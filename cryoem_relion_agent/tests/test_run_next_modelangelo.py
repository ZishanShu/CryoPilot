"""Tests for ModelAngelo execution path (command build, dry-run, subprocess mock)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from agent.reporter import Reporter
from app.scheduler import Scheduler
from agent.structure_router import ModelingContext
from relion.model_angelo_runner import _resolve_executable, run_model_angelo
from relion.relion_actions import RelionActionExecutor
from schemas.action import Action, ActionName
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState


def _make_action(map_path: str | None, seq_path: str | None) -> Action:
    params: dict = {}
    if map_path is not None:
        params["map_path"] = map_path
    if seq_path is not None:
        params["sequence_path"] = seq_path
    return Action(
        name=ActionName.RUN_MODELANGELO,
        reason="test",
        parameters=params,
        priority=70,
    )


class TestModelAngeloCommandBuild(unittest.TestCase):
    """Command argv and dry-run behavior."""

    def test_constructs_model_angelo_build_argv(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            m = root / "m.mrc"
            f = root / "p.fasta"
            m.write_bytes(b"\x00")
            f.write_text(">A\nM\n", encoding="utf-8")
            cfg = root / "configs" / "model_builders.yaml"
            cfg.parent.mkdir(parents=True)
            cfg.write_text(
                "model_angelo:\n  executable: model_angelo\n  subcommand_with_seq: [build]\n"
                "  flags: {map: '-v', protein_fasta: '-pf', output: '-o'}\n"
                "  output_runs_subdir: modelangelo_runs\n",
                encoding="utf-8",
            )
            action = _make_action(str(m), str(f))
            out = run_model_angelo(action, root, state=None, dry_run=True, model_builders_yaml=cfg)
            self.assertEqual(out["status"], "dry_run")
            self.assertTrue(out["dry_run"])
            self.assertIn("model_angelo", out["command"] or "")
            self.assertIn("build", out["command"] or "")
            self.assertIn("-v", out["command"] or "")
            self.assertIn("-pf", out["command"] or "")
            self.assertIn("-o", out["command"] or "")
            argv = out.get("command_argv") or []
            self.assertEqual(argv[0], "model_angelo")
            self.assertIn("build", argv)
            self.assertIn("modelangelo_runs", out["output_dir"] or "")

    def test_dry_run_does_not_call_subprocess(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            m = root / "m.mrc"
            f = root / "p.fasta"
            m.write_bytes(b"\x00")
            f.write_text(">A\nM\n", encoding="utf-8")
            cfg = root / "model_builders.yaml"
            cfg.write_text("model_angelo:\n  executable: model_angelo\n", encoding="utf-8")
            action = _make_action(str(m), str(f))
            with patch("relion.model_angelo_runner.subprocess.run") as mock_run:
                out = run_model_angelo(action, root, state=None, dry_run=True, model_builders_yaml=cfg)
                mock_run.assert_not_called()
            self.assertIsNone(out["return_code"])
            self.assertEqual(out["status"], "dry_run")

    def test_missing_sequence_path_error(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            m = root / "m.mrc"
            m.write_bytes(b"\x00")
            cfg = root / "model_builders.yaml"
            cfg.write_text("{}", encoding="utf-8")
            action = _make_action(str(m), None)
            state = SampleState(
                project_id="p",
                project_path=str(root),
                stage="x",
                status="running",
                map_path=str(m),
                sequence_path=None,
            )
            out = run_model_angelo(action, root, state=state, dry_run=False, model_builders_yaml=cfg)
            self.assertEqual(out["status"], "error_validation")
            self.assertIn("sequence", (out.get("error_message") or "").lower())

    def test_missing_map_path_error(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            f = root / "p.fasta"
            f.write_text(">A\nM\n", encoding="utf-8")
            cfg = root / "model_builders.yaml"
            cfg.write_text("{}", encoding="utf-8")
            action = _make_action(None, str(f))
            state = SampleState(
                project_id="p",
                project_path=str(root),
                stage="x",
                status="running",
                map_path=None,
                sequence_path=str(f),
            )
            out = run_model_angelo(action, root, state=state, dry_run=False, model_builders_yaml=cfg)
            self.assertEqual(out["status"], "error_validation")
            self.assertIn("map", (out.get("error_message") or "").lower())

    def test_absolute_executable_path_in_argv(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_bin = root / "envs" / "model_angelo" / "bin" / "model_angelo"
            fake_bin.parent.mkdir(parents=True)
            fake_bin.write_text("#!/bin/sh\necho\n", encoding="utf-8")
            fake_bin.chmod(0o755)
            m = root / "m.mrc"
            f = root / "p.fasta"
            m.write_bytes(b"\x00")
            f.write_text(">A\nM\n", encoding="utf-8")
            cfg = root / "model_builders.yaml"
            cfg.write_text(
                f'model_angelo:\n  executable: "{fake_bin}"\n  subcommand_with_seq: build\n',
                encoding="utf-8",
            )
            action = _make_action(str(m), str(f))
            out = run_model_angelo(action, root, state=None, dry_run=True, model_builders_yaml=cfg)
            self.assertEqual(out["status"], "dry_run")
            argv = out.get("command_argv") or []
            self.assertEqual(argv[0], str(fake_bin.resolve()))

    def test_missing_configured_executable_returns_error(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            m = root / "m.mrc"
            f = root / "p.fasta"
            m.write_bytes(b"\x00")
            f.write_text(">A\nM\n", encoding="utf-8")
            cfg = root / "model_builders.yaml"
            cfg.write_text(
                'model_angelo:\n  executable: "/nonexistent/path/model_angelo"\n',
                encoding="utf-8",
            )
            action = _make_action(str(m), str(f))
            out = run_model_angelo(action, root, state=None, dry_run=True, model_builders_yaml=cfg)
            self.assertEqual(out["status"], "dry_run")
            self.assertTrue(out["success"])
            self.assertIsNone(out["return_code"])
            self.assertTrue(out.get("warnings"))
            self.assertIn("executable not found", " ".join(out.get("warnings") or []))

    def test_modelangelo_resolve_executable_permission_error_no_traceback(self) -> None:
        with patch("agent.backend_runtime.Path.exists", side_effect=PermissionError("denied")):
            exe, err, diag = _resolve_executable("/root/anaconda3/envs/model_angelo/bin/model_angelo")

        self.assertIn("/root/anaconda3", exe)
        self.assertIsNotNone(err)
        self.assertIn("not accessible", err or "")
        access = diag.get("access") or {}
        self.assertTrue(access.get("not_accessible"))

    def test_run_modelangelo_dry_run_allows_stale_executable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            m = root / "m.mrc"
            f = root / "p.fasta"
            m.write_bytes(b"\x00")
            f.write_text(">A\nM\n", encoding="utf-8")
            cfg = root / "model_builders.yaml"
            cfg.write_text(
                "model_angelo:\n"
                "  executable: /root/anaconda3/envs/model_angelo/bin/model_angelo\n"
                "  torch_home: /root/share/szs/CryoAgent/model-angelo/public/model_angelo_weights\n",
                encoding="utf-8",
            )
            action = _make_action(str(m), str(f))

            out = run_model_angelo(action, root, state=None, dry_run=True, model_builders_yaml=cfg)

            self.assertEqual(out["status"], "dry_run")
            self.assertTrue(out["success"])
            self.assertIsNone(out["return_code"])
            self.assertIn("[DRY-RUN] Would execute:", out["command"])
            self.assertTrue(out.get("warnings"))
            self.assertIn("ModelAngelo real execution blocked", " ".join(out.get("warnings") or []))

    def test_modelangelo_yaml_alias_merges(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_bin = root / "ma"
            fake_bin.write_text("#!/bin/sh\n", encoding="utf-8")
            fake_bin.chmod(0o755)
            m = root / "m.mrc"
            f = root / "p.fasta"
            m.write_bytes(b"\x00")
            f.write_text(">A\nM\n", encoding="utf-8")
            cfg = root / "model_builders.yaml"
            cfg.write_text(
                f'modelangelo:\n  executable: "{fake_bin}"\n  subcommand_with_seq: "build"\n',
                encoding="utf-8",
            )
            action = _make_action(str(m), str(f))
            out = run_model_angelo(action, root, state=None, dry_run=True, model_builders_yaml=cfg)
            self.assertEqual(out["status"], "dry_run")
            argv = out.get("command_argv") or []
            self.assertEqual(argv[0], str(fake_bin.resolve()))
            self.assertIn("build", argv)

    def test_uses_local_weights_cache_when_present(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_bin = root / "model_angelo"
            fake_bin.write_text("#!/bin/sh\n", encoding="utf-8")
            fake_bin.chmod(0o755)
            torch_home = root / "weights"
            (torch_home / "hub" / "checkpoints" / "model_angelo_v1.0").mkdir(parents=True)
            m = root / "m.mrc"
            f = root / "p.fasta"
            m.write_bytes(b"\x00")
            f.write_text(">A\nM\n", encoding="utf-8")
            cfg = root / "model_builders.yaml"
            cfg.write_text(
                f'model_angelo:\n  executable: "{fake_bin}"\n  torch_home: "{torch_home}"\n',
                encoding="utf-8",
            )
            action = _make_action(str(m), str(f))
            out = run_model_angelo(action, root, state=None, dry_run=True, model_builders_yaml=cfg)
            self.assertEqual(out["status"], "dry_run")
            self.assertIn("cached=True", out.get("stdout") or "")

    def test_missing_weights_cache_keeps_download_behavior(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_bin = root / "model_angelo"
            fake_bin.write_text("#!/bin/sh\n", encoding="utf-8")
            fake_bin.chmod(0o755)
            torch_home = root / "weights_missing"
            m = root / "m.mrc"
            f = root / "p.fasta"
            m.write_bytes(b"\x00")
            f.write_text(">A\nM\n", encoding="utf-8")
            cfg = root / "model_builders.yaml"
            cfg.write_text(
                f'model_angelo:\n  executable: "{fake_bin}"\n  torch_home: "{torch_home}"\n',
                encoding="utf-8",
            )
            action = _make_action(str(m), str(f))
            out = run_model_angelo(action, root, state=None, dry_run=True, model_builders_yaml=cfg)
            self.assertEqual(out["status"], "dry_run")
            self.assertIn("cached=False", out.get("stdout") or "")

    def test_torch_home_env_overrides_config(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_bin = root / "model_angelo"
            fake_bin.write_text("#!/bin/sh\n", encoding="utf-8")
            fake_bin.chmod(0o755)
            cfg_home = root / "cfg_home"
            env_home = root / "env_home"
            (env_home / "hub" / "checkpoints" / "model_angelo_v1.0").mkdir(parents=True)
            m = root / "m.mrc"
            f = root / "p.fasta"
            m.write_bytes(b"\x00")
            f.write_text(">A\nM\n", encoding="utf-8")
            cfg = root / "model_builders.yaml"
            cfg.write_text(
                f'model_angelo:\n  executable: "{fake_bin}"\n  torch_home: "{cfg_home}"\n',
                encoding="utf-8",
            )
            action = _make_action(str(m), str(f))
            with patch.dict("os.environ", {"TORCH_HOME": str(env_home)}, clear=False):
                out = run_model_angelo(action, root, state=None, dry_run=True, model_builders_yaml=cfg)
            self.assertEqual(out["status"], "dry_run")
            self.assertIn(f"TORCH_HOME={env_home.resolve()}", out.get("stdout") or "")


class TestModelAngeloSubprocess(unittest.TestCase):
    """Non-dry-run subprocess and CIF detection."""

    def test_subprocess_run_called_when_not_dry_run(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            m = root / "m.mrc"
            f = root / "p.fasta"
            m.write_bytes(b"\x00")
            f.write_text(">A\nM\n", encoding="utf-8")
            cfg = root / "model_builders.yaml"
            cfg.write_text("{}", encoding="utf-8")
            action = _make_action(str(m), str(f))
            mock_completed = MagicMock()
            mock_completed.returncode = 0
            mock_completed.stdout = "ok"
            mock_completed.stderr = ""
            with patch("relion.model_angelo_runner.subprocess.run", return_value=mock_completed) as mock_run:
                out = run_model_angelo(action, root, state=None, dry_run=False, model_builders_yaml=cfg)
                mock_run.assert_called_once()
            self.assertEqual(out["status"], "completed")
            self.assertIsNotNone(out.get("output_dir"))

    def test_generated_model_path_when_cif_exists(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            m = root / "m.mrc"
            f = root / "p.fasta"
            m.write_bytes(b"\x00")
            f.write_text(">A\nM\n", encoding="utf-8")
            cfg = root / "model_builders.yaml"
            cfg.write_text("{}", encoding="utf-8")
            action = _make_action(str(m), str(f))
            mock_completed = MagicMock()
            mock_completed.returncode = 0
            mock_completed.stdout = ""
            mock_completed.stderr = ""

            def _fake_run(argv, **kwargs):  # noqa: ANN001
                out_dir = Path(argv[-1])
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "out.cif").write_text("data_", encoding="utf-8")
                return mock_completed

            with patch("relion.model_angelo_runner.subprocess.run", side_effect=_fake_run):
                out = run_model_angelo(action, root, state=None, dry_run=False, model_builders_yaml=cfg)
            self.assertTrue(out.get("generated_model_path", "").endswith("out.cif"))

    def test_subprocess_receives_torch_home_env(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_bin = root / "model_angelo"
            fake_bin.write_text("#!/bin/sh\n", encoding="utf-8")
            fake_bin.chmod(0o755)
            torch_home = root / "weights"
            (torch_home / "hub" / "checkpoints" / "model_angelo_v1.0").mkdir(parents=True)
            m = root / "m.mrc"
            f = root / "p.fasta"
            m.write_bytes(b"\x00")
            f.write_text(">A\nM\n", encoding="utf-8")
            cfg = root / "model_builders.yaml"
            cfg.write_text(
                f'model_angelo:\n  executable: "{fake_bin}"\n  torch_home: "{torch_home}"\n',
                encoding="utf-8",
            )
            action = _make_action(str(m), str(f))
            mock_completed = MagicMock()
            mock_completed.returncode = 0
            mock_completed.stdout = "ok"
            mock_completed.stderr = ""
            with patch("relion.model_angelo_runner.subprocess.run", return_value=mock_completed) as mock_run:
                out = run_model_angelo(action, root, state=None, dry_run=False, model_builders_yaml=cfg)
                kwargs = mock_run.call_args.kwargs
                self.assertIn("env", kwargs)
                self.assertEqual(kwargs["env"].get("TORCH_HOME"), str(torch_home.resolve()))
            self.assertEqual(out["status"], "completed")


class TestRelionActionExecutorModelAngelo(unittest.TestCase):
    """Executor dispatches run_modelangelo."""

    def test_executor_calls_runner(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            schemes = root / "schemes.yaml"
            schemes.write_text("schemes: {}\n", encoding="utf-8")
            m = root / "m.mrc"
            f = root / "p.fasta"
            m.write_bytes(b"\x00")
            f.write_text(">A\nM\n", encoding="utf-8")
            mb = root / "model_builders.yaml"
            mb.write_text("{}", encoding="utf-8")
            ex = RelionActionExecutor(schemes)
            ex.model_builders_yaml = mb
            action = _make_action(str(m), str(f))
            with patch("relion.relion_actions.run_model_angelo") as mock_run:
                mock_run.return_value = {"action_name": "run_modelangelo", "status": "dry_run"}
                ex.execute(action, root, dry_run=True, state=None)
                mock_run.assert_called_once()

    def test_run_next_dry_run_does_not_crash_on_stale_modelangelo_path(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            schemes = root / "schemes.yaml"
            schemes.write_text("schemes: {}\n", encoding="utf-8")
            mb = root / "model_builders.yaml"
            mb.write_text(
                "model_angelo:\n"
                "  executable: /root/anaconda3/envs/model_angelo/bin/model_angelo\n"
                "  torch_home: /root/share/szs/CryoAgent/model-angelo/public/model_angelo_weights\n",
                encoding="utf-8",
            )
            m = root / "m.mrc"
            f = root / "p.fasta"
            m.write_bytes(b"\x00")
            f.write_text(">A\nM\n", encoding="utf-8")
            action = Action(
                name=ActionName.RUN_MODELANGELO,
                reason="test",
                parameters={
                    "map_path": str(m),
                    "sequence_path": str(f),
                    "model_building_backend": "modelangelo",
                    "model_building_availability": "unavailable",
                    "model_building_execution_mode": "unavailable",
                    "model_building_blocked_reason": "missing_dependency: modelangelo_executable",
                    "model_building_missing_inputs": [],
                    "model_building_missing_dependencies": ["modelangelo_executable"],
                },
                priority=70,
            )
            ex = RelionActionExecutor(schemes)

            out = ex.execute(action, root, dry_run=True, state=None)

            self.assertEqual(out["status"], "dry_run")
            self.assertTrue(out["success"])
            self.assertTrue(out.get("warnings"))


class TestRunNextModelAngeloSelectionConsistency(unittest.TestCase):
    def test_run_next_dry_run_selected_backend_matches_action_name(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "agent" / "configs"
            cfg.mkdir(parents=True)
            for name in ("policy_rules.yaml", "router_rules.yaml", "schemes.yaml"):
                src = Path(__file__).resolve().parents[1] / "configs" / name
                (cfg / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            (cfg / "model_builders.yaml").write_text(
                "model_angelo:\n"
                "  executable: /root/anaconda3/envs/model_angelo/bin/model_angelo\n"
                "  torch_home: /root/share/szs/CryoAgent/model-angelo/public/model_angelo_weights\n",
                encoding="utf-8",
            )
            project = root / "project"
            project.mkdir()
            m = project / "postprocess_masked.mrc"
            f = project / "seq.fasta"
            m.write_bytes(b"\x00")
            f.write_text(">A\nM\n", encoding="utf-8")
            sched = Scheduler(project_root=project, base_dir=root / "agent")
            state = SampleState(
                project_id="p",
                project_path=str(project),
                stage="PostProcess",
                status="running",
                map_path=str(m),
                sequence_path=str(f),
            )
            qc = QCMetrics(
                motioncorr_completed=True,
                ctf_completed=True,
                picking_completed=True,
                class2d_completed=True,
                init3d_completed=True,
                refine3d_completed=True,
                current_resolution=3.8,
            )
            with patch.object(sched, "scan_project", return_value=(state, qc)):
                action, result = sched.run_next(dry_run=True, sequence_override=f)

            decision = action.parameters.get("model_building_selection_decision") or {}
            self.assertEqual(action.name, ActionName.RUN_MODELANGELO)
            self.assertEqual(result.get("action_name"), "run_modelangelo")
            self.assertEqual(action.parameters.get("model_building_backend"), "modelangelo")
            self.assertEqual(result.get("model_building_backend"), "modelangelo")
            self.assertEqual(decision.get("selected_backend"), "modelangelo")
            self.assertFalse(decision.get("fallback_applied"))
            self.assertNotEqual(decision.get("selection_reason"), "map_sequence_present_and_emodelx_available")
            self.assertEqual(result.get("status"), "dry_run")
            self.assertTrue(result.get("success"))

            history_lines = sched.history_file.read_text(encoding="utf-8").splitlines()
            execution_entries = [json.loads(line) for line in history_lines if '"execution_status": "dry_run"' in line]
            self.assertTrue(execution_entries)
            summary = execution_entries[-1]["output_summary"]
            self.assertIn("model_building_backend=modelangelo", summary)
            self.assertNotIn("selected=run_emodelx", summary)

    def test_modelangelo_unavailable_does_not_claim_emodelx_selected_unless_action_is_emodelx(self) -> None:
        sched = Scheduler(Path("/tmp"))
        state = SampleState(
            project_id="p",
            project_path="/tmp/proj",
            stage="PostProcess",
            status="running",
            map_path="/tmp/map.mrc",
            sequence_path="/tmp/seq.fasta",
        )
        action = Action(
            name=ActionName.RUN_MODELANGELO,
            reason="test",
            parameters={
                "map_path": "/tmp/map.mrc",
                "sequence_path": "/tmp/seq.fasta",
                "model_building_backend": "modelangelo",
            },
            priority=70,
        )

        enriched = sched._enrich_model_building_action(state, action)

        decision = enriched.parameters.get("model_building_selection_decision") or {}
        self.assertEqual(enriched.name, ActionName.RUN_MODELANGELO)
        self.assertEqual(enriched.parameters.get("model_building_backend"), "modelangelo")
        self.assertEqual(decision.get("selected_backend"), "modelangelo")
        self.assertFalse(decision.get("fallback_applied"))
        self.assertNotEqual(decision.get("selection_reason"), "map_sequence_present_and_emodelx_available")

    def test_emodelx_action_keeps_selected_backend_emodelx(self) -> None:
        sched = Scheduler(Path("/tmp"))
        state = SampleState(
            project_id="p",
            project_path="/tmp/proj",
            stage="PostProcess",
            status="running",
            map_path="/tmp/map.mrc",
            sequence_path="/tmp/seq.fasta",
        )
        action = Action(
            name=ActionName.RUN_EMODELX,
            reason="test",
            parameters={
                "map_path": "/tmp/map.mrc",
                "sequence_path": "/tmp/seq.fasta",
                "model_building_backend": "emodelx",
            },
            priority=80,
        )

        enriched = sched._enrich_model_building_action(state, action)

        decision = enriched.parameters.get("model_building_selection_decision") or {}
        self.assertEqual(enriched.name, ActionName.RUN_EMODELX)
        self.assertEqual(enriched.parameters.get("model_building_backend"), "emodelx")
        self.assertEqual(decision.get("selected_backend"), "emodelx")


class TestReportExecutionSection(unittest.TestCase):
    """Report includes Execution from state.last_execution."""

    def test_markdown_has_execution_block(self) -> None:
        state = SampleState(
            project_id="p1",
            project_path="/tmp/p",
            stage="PostProcess",
            status="running",
            last_execution={
                "action_name": "run_modelangelo",
                "status": "dry_run",
                "command": "model_angelo build -v /m -pf /s -o /out",
                "dry_run": True,
                "return_code": 0,
                "output_dir": "/tmp/out",
                "generated_model_path": None,
                "error_message": None,
            },
        )
        qc = QCMetrics()
        report = Reporter().build_report(state=state, qc=qc, recent_action=None, next_action=None)
        assert report.execution is not None
        md = report.to_markdown()
        self.assertIn("## Execution", md)
        self.assertIn("run_modelangelo", md)
        self.assertIn("/tmp/out", md)


TUTORIAL50 = Path("/root/share/szs/CryoAgent/Tutorial5.0")
MODELANGELO_EXE = Path("/root/anaconda3/envs/model_angelo/bin/model_angelo")


@unittest.skipUnless(
    TUTORIAL50.is_dir()
    and (TUTORIAL50 / "PostProcess" / "job035" / "postprocess_masked.mrc").is_file()
    and MODELANGELO_EXE.is_file(),
    "Tutorial5.0 fixture or ModelAngelo executable not present",
)
class TestTutorial50DryRunIntegration(unittest.TestCase):
    """run-next --dry-run on Tutorial5.0 routes to run_modelangelo."""

    def test_dry_run_returns_modelangelo(self) -> None:
        from app.scheduler import Scheduler

        with TemporaryDirectory() as tmp:
            fasta = Path(tmp) / "demo.fasta"
            fasta.write_text(">A\nM\n", encoding="utf-8")
            sched = Scheduler(project_root=TUTORIAL50)
            action, result = sched.run_next(dry_run=True, sequence_override=fasta)
            self.assertEqual(action.name, ActionName.RUN_MODELANGELO)
            self.assertEqual(result.get("status"), "dry_run")
            self.assertIn("model_angelo", result.get("command") or "")
            payload = json.loads(json.dumps({"execution": result}))
            self.assertEqual(payload["execution"]["dry_run"], True)


if __name__ == "__main__":
    unittest.main()
