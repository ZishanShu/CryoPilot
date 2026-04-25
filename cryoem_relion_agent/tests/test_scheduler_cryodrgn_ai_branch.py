"""Scheduler / executor integration for the cryodrgn_ai reconstruction branch."""

from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
