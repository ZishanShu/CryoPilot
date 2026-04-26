"""Scheduler / executor integration for the cryodrgn_ai reconstruction branch."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from app.scheduler import Scheduler
from relion.relion_actions import RelionActionExecutor
from schemas.action import Action, ActionName
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _fake_cryodrgn(root: Path) -> Path:
    exe = root / "cryodrgn"
    exe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    exe.chmod(0o755)
    return exe


class TestSchedulerCryodrgnAiBranch(unittest.TestCase):
    def test_scheduler_selects_branch_from_state_context(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            star = root / "particles.star"
            star.write_text("data_\n", encoding="utf-8")
            exe = _fake_cryodrgn(root)
            state = SampleState(
                project_id="p",
                project_path=str(root),
                stage="Class2D",
                status="running",
                last_execution={
                    "reconstruction_context": {
                        "no_reliable_initial_model": True,
                        "particles_star_or_mrcs": str(star),
                        "ctf_metadata": "relion_star",
                        "apix": 1.2,
                        "box_size": 256,
                        "downsample_box": 64,
                        "cryodrgn_executable": str(exe),
                    }
                },
            )
            qc = QCMetrics(motioncorr_completed=True, ctf_completed=True, class2d_completed=True)
            sched = Scheduler(project_root=root)
            action = sched._resolve_next_action(state, qc)
            self.assertEqual(action.name, ActionName.RUN_CRYODRGN_AI)
            self.assertEqual(action.parameters["selected_backend"], "cryodrgn_ai")

    def test_failed_subprocess_records_failed_status_and_stderr_path(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            particles = root / "particles.mrcs"
            ctf = root / "ctf.pkl"
            particles.write_bytes(b"m")
            ctf.write_bytes(b"c")
            exe = _fake_cryodrgn(root)
            action = Action(
                name=ActionName.RUN_CRYODRGN_AI,
                reason="test",
                parameters={
                    "particles_star_or_mrcs": str(particles),
                    "ctf_metadata": str(ctf),
                    "cryodrgn_executable": str(exe),
                    "execution_mode": "real",
                    "run_dir": str(root / "run"),
                },
                priority=67,
            )

            def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:
                proc = MagicMock()
                if cmd[1:3] == ["abinit", "-h"]:
                    proc.returncode = 0
                    proc.stdout = "help"
                    proc.stderr = ""
                    return proc
                proc.returncode = 2
                proc.stdout = ""
                proc.stderr = "abinit failed"
                return proc

            executor = RelionActionExecutor(CONFIGS / "schemes.yaml")
            with patch("backends.reconstruction.cryodrgn_ai_backend._gpu_available", return_value=True), patch(
                "backends.reconstruction.cryodrgn_ai_backend.subprocess.run", side_effect=fake_run
            ):
                result = executor.execute(action=action, project_root=root, dry_run=False)

            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["success"])
            self.assertTrue(str(result.get("stderr_log")).endswith("stderr.log"))
            self.assertIn("subprocess_failed", result.get("failed_reason") or "")


class TestSchedulerSampleVolumesParams(unittest.TestCase):
    def test_scheduler_passes_sample_volumes_params_from_reconstruction_context(self) -> None:
        """sample_volumes, n_sampled_volumes, eval_vol_* from reconstruction_context
        must appear in the RUN_CRYODRGN_AI action parameters."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            star = root / "particles.star"
            star.write_text("data_\n", encoding="utf-8")
            exe = _fake_cryodrgn(root)
            state = SampleState(
                project_id="p",
                project_path=str(root),
                stage="Class2D",
                status="running",
                last_execution={
                    "reconstruction_context": {
                        "no_reliable_initial_model": True,
                        "particles_star_or_mrcs": str(star),
                        "ctf_metadata": "relion_star",
                        "apix": 1.2,
                        "box_size": 256,
                        "downsample_box": 64,
                        "cryodrgn_executable": str(exe),
                        "sample_volumes": True,
                        "n_sampled_volumes": 5,
                        "eval_vol_apix": 1.2,
                        "eval_vol_box_size": 256,
                        "eval_vol_device": "cpu",
                    }
                },
            )
            qc = QCMetrics(motioncorr_completed=True, ctf_completed=True, class2d_completed=True)
            sched = Scheduler(project_root=root)
            action = sched._resolve_next_action(state, qc)
            self.assertEqual(action.name, ActionName.RUN_CRYODRGN_AI)
            p = action.parameters
            self.assertTrue(p.get("sample_volumes"))
            self.assertEqual(p.get("n_sampled_volumes"), 5)
            self.assertAlmostEqual(float(p.get("eval_vol_apix")), 1.2)
            self.assertEqual(p.get("eval_vol_box_size"), 256)
            self.assertEqual(p.get("eval_vol_device"), "cpu")


class TestReporterCryodrgnAiSampledMaps(unittest.TestCase):
    def test_reporter_shows_sampled_volume_count_and_candidate_map_count(self) -> None:
        """Reporter extra dict and markdown must show sampled volume / candidate map counts."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = SampleState(
                project_id="p",
                project_path=str(root),
                stage="Class3D",
                status="running",
                last_execution={
                    "action_name": "run_cryodrgn_ai",
                    "status": "completed",
                    "output_artifacts": {
                        "sampled_volumes": [
                            "/run/sampled_volumes/vol_000.mrc",
                            "/run/sampled_volumes/vol_001.mrc",
                            "/run/sampled_volumes/vol_002.mrc",
                        ],
                        "model_weights": ["/run/abinit/weights.pkl"],
                        "latent_embeddings": ["/run/abinit/z.pkl"],
                        "mrc_files": [
                            "/run/sampled_volumes/vol_000.mrc",
                            "/run/sampled_volumes/vol_001.mrc",
                            "/run/sampled_volumes/vol_002.mrc",
                        ],
                        "representative_z_path": "/run/representative_z.txt",
                        "run_dir": "/run",
                    },
                    "candidate_maps": [
                        {
                            "source_backend": "cryodrgn_ai",
                            "map_path": f"/run/sampled_volumes/vol_00{i}.mrc",
                            "qc_status": "pending",
                            "recommended_next_action": "route_to_model_building",
                        }
                        for i in range(3)
                    ],
                    "recommended_downstream_action": "route_to_model_building",
                },
            )
            qc = QCMetrics(motioncorr_completed=True, ctf_completed=True, class2d_completed=True, init3d_completed=True)
            sched = Scheduler(project_root=root)
            with patch.object(Scheduler, "scan_project", return_value=(state, qc)):
                report = sched.generate_report(refresh_state=False)

            # Extra dict must have counts
            self.assertEqual(report.extra.get("cryodrgn_ai_sampled_volume_count"), "3")
            self.assertEqual(report.extra.get("cryodrgn_ai_candidate_map_count"), "3")

            # Markdown must include the sampled maps section
            md = report.to_markdown()
            self.assertIn("cryodrgn_ai Sampled Maps", md)
            self.assertIn("Sampled volumes: 3", md)
            self.assertIn("Candidate maps: 3", md)
            self.assertIn("route_to_model_building", md)


class TestCandidateMapStatePersistence(unittest.TestCase):
    def test_candidate_map_model_building_dry_run_preserves_candidate_state(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            seq = root / "seq.fasta"
            seq.write_text(">a\nAAAA\n", encoding="utf-8")
            qc = QCMetrics(
                motioncorr_completed=True,
                ctf_completed=True,
                class2d_completed=True,
                init3d_completed=True,
                refine3d_completed=True,
                postprocess_completed=True,
                current_resolution=3.2,
            )
            candidate_maps = [
                {
                    "source_backend": "cryodrgn_ai",
                    "source_run_dir": str(root),
                    "map_path": str(root / "sampled_volumes" / f"vol_{idx:03d}.mrc"),
                    "latent_source": str(root / "representative_z.txt"),
                    "weights_path": str(root / "abinit" / "weights.pkl"),
                    "qc_status": "pending",
                    "recommended_next_action": "route_to_model_building",
                }
                for idx in range(5)
            ]
            routing_records = [
                {
                    "candidate_index": idx,
                    "source_backend": "cryodrgn_ai",
                    "source_run_dir": str(root),
                    "map_path": str(root / "sampled_volumes" / f"vol_{idx:03d}.mrc"),
                    "latent_source": str(root / "representative_z.txt"),
                    "weights_path": str(root / "abinit" / "weights.pkl"),
                    "recommended_backend": "modelangelo",
                    "recommended_action": "run_modelangelo",
                    "downstream_action": "run_modelangelo",
                    "preflight": {"available": True, "execution_mode": "dry_run"},
                }
                for idx in range(5)
            ]
            state = SampleState(
                project_id="p",
                project_path=str(root),
                stage="PostProcess",
                status="running",
                sequence_path=str(seq),
                candidate_maps=copy.deepcopy(candidate_maps),
                last_execution={
                    "action_name": "run_cryodrgn_ai",
                    "status": "completed",
                    "candidate_maps": copy.deepcopy(candidate_maps),
                },
            )
            action = Action(
                name=ActionName.RUN_MODELANGELO,
                reason="candidate map routing",
                parameters={
                    "branch_name": "candidate_map_model_building",
                    "candidate_map_path": candidate_maps[0]["map_path"],
                    "candidate_index": 0,
                    "candidate_map_routing_record": routing_records[0],
                    "candidate_map_routing_records": routing_records,
                    "candidate_maps": copy.deepcopy(candidate_maps),
                    "upstream_candidate_maps": copy.deepcopy(candidate_maps),
                    "candidate_map_count": len(candidate_maps),
                    "map_path": candidate_maps[0]["map_path"],
                    "sequence_path": str(seq),
                    "model_building_backend": "modelangelo",
                    "routing_triggered": True,
                },
                priority=58,
            )
            sched = Scheduler(project_root=root)
            with patch.object(Scheduler, "scan_project", return_value=(state, qc)), patch.object(
                Scheduler, "_resolve_next_action", return_value=action
            ), patch.object(
                sched.action_executor,
                "execute",
                return_value={
                    "action_name": "run_modelangelo",
                    "status": "dry_run",
                    "dry_run": True,
                    "output_dir": str(root / "ModelBuilding" / "ModelAngelo" / "candidate_000"),
                },
            ):
                selected_action, _result = sched.run_next(dry_run=True)
            self.assertEqual(selected_action.name, ActionName.RUN_MODELANGELO)

            saved = sched.state_manager.load_state()
            assert saved is not None
            self.assertEqual(len(saved.candidate_maps), 5)
            self.assertEqual(saved.last_execution.get("action_name"), "run_modelangelo")
            self.assertEqual(saved.last_execution.get("status"), "dry_run")
            self.assertEqual(len(saved.last_execution.get("candidate_maps") or []), 5)
            self.assertEqual(len(saved.last_execution.get("candidate_map_routing_records") or []), 5)
            batch_plans = saved.last_execution.get("candidate_map_batch_model_building_plans")
            if batch_plans is not None:
                self.assertIsInstance(batch_plans, list)
                self.assertEqual(saved.last_execution.get("candidate_map_batch_count"), len(batch_plans))
                self.assertEqual(saved.last_execution.get("candidate_map_batch_top_k"), 5)
            # When execute() does not return these fields, scheduler preserves rather
            # than fabricates them; assert optional behavior for compatibility.
            selection_reason = saved.last_execution.get("candidate_map_selection_reason")
            if selection_reason is not None:
                self.assertEqual(selection_reason, "highest_qc_score")
            selected_score = saved.last_execution.get("selected_candidate_qc_score")
            if selected_score is not None:
                self.assertGreaterEqual(float(selected_score), 0.0)
            selected_idx = saved.last_execution.get("selected_candidate_index")
            if selected_idx is not None:
                self.assertEqual(selected_idx, saved.last_execution.get("candidate_index"))
            self.assertTrue(
                str(saved.last_execution.get("candidate_map_routing_records")[0]["map_path"]).endswith(
                    "sampled_volumes/vol_000.mrc"
                )
            )
            self.assertTrue(
                str(saved.last_execution.get("candidate_map_routing_records")[0]["weights_path"]).endswith(
                    "abinit/weights.pkl"
                )
            )
            self.assertTrue(
                str(saved.last_execution.get("candidate_map_routing_records")[0]["latent_source"]).endswith(
                    "representative_z.txt"
                )
            )


if __name__ == "__main__":
    unittest.main()
