"""Tests for ``run-next`` CLI picking-backend options.

Covers:
- default ``run-next`` does not pass ``preferred_picking_backend``.
- ``--picking-backend topaz`` threads through to the scheduler.
- Topaz-specific flags map to ``picking_overrides`` dict with canonical keys.
- Scheduler populates ``action.parameters`` from ``picking_overrides`` and
  keeps metadata consistent on the execution result.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple
from unittest.mock import patch

from typer.testing import CliRunner

from app.main import app
from app.scheduler import Scheduler
from schemas.action import Action, ActionName
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState


class _Capture:
    def __init__(self) -> None:
        self.kwargs: Dict[str, Any] = {}

    def make_stub(self):
        cap = self

        def _fake(self, *args, **kwargs):  # type: ignore[no-redef]
            cap.kwargs = dict(kwargs)
            backend = kwargs.get("preferred_picking_backend") or "relion_log"
            params: Dict[str, Any] = {"picking_backend": backend}
            params.update(kwargs.get("picking_overrides") or {})
            action = Action(
                name=ActionName.RUN_AUTOPICK,
                reason="stub",
                parameters=params,
                priority=78,
            )
            return action, {"status": "dry_run", "metadata": {"picking_backend": backend}}

        return _fake


class TestRunNextCLIDefaultPreserved(unittest.TestCase):
    def test_default_run_next_does_not_pass_picking_backend(self) -> None:
        runner = CliRunner()
        cap = _Capture()
        with TemporaryDirectory() as tmp:
            with patch.object(Scheduler, "run_next", cap.make_stub()):
                res = runner.invoke(app, ["run-next", "--project", tmp, "--dry-run"])
            self.assertEqual(res.exit_code, 0, msg=res.output)
            self.assertIsNone(cap.kwargs.get("preferred_picking_backend"))
            # When no overrides provided, the scheduler receives ``None``.
            self.assertIsNone(cap.kwargs.get("picking_overrides"))


class TestRunNextCLITopazRouting(unittest.TestCase):
    def test_topaz_cli_threads_backend_and_overrides(self) -> None:
        runner = CliRunner()
        cap = _Capture()
        with TemporaryDirectory() as tmp:
            star = Path(tmp) / "m.star"
            star.write_text("", encoding="utf-8")
            output_dir = Path(tmp) / "topaz_out"
            with patch.object(Scheduler, "run_next", cap.make_stub()):
                res = runner.invoke(
                    app,
                    [
                        "run-next",
                        "--project", tmp,
                        "--dry-run",
                        "--picking-backend", "topaz",
                        "--micrographs-star", str(star),
                        "--particle-diameter-angstrom", "200",
                        "--topaz-threshold", "-6",
                        "--topaz-device", "cpu",
                        "--topaz-output-dir", str(output_dir),
                    ],
                )
            self.assertEqual(res.exit_code, 0, msg=res.output)
            self.assertEqual(cap.kwargs.get("preferred_picking_backend"), "topaz")
            overrides = cap.kwargs.get("picking_overrides") or {}
            self.assertEqual(overrides.get("micrographs_star_path"), str(star))
            self.assertEqual(overrides.get("particle_diameter_angstrom"), 200.0)
            self.assertEqual(overrides.get("topaz_threshold"), -6.0)
            self.assertEqual(overrides.get("topaz_device"), "cpu")
            self.assertEqual(overrides.get("output_dir"), str(output_dir))

    def test_cli_accepts_cryolo_and_relion_log(self) -> None:
        runner = CliRunner()
        for backend in ("cryolo", "relion_log"):
            cap = _Capture()
            with TemporaryDirectory() as tmp:
                with patch.object(Scheduler, "run_next", cap.make_stub()):
                    res = runner.invoke(
                        app,
                        ["run-next", "--project", tmp, "--dry-run", "--picking-backend", backend],
                    )
                self.assertEqual(res.exit_code, 0, msg=res.output)
                self.assertEqual(cap.kwargs.get("preferred_picking_backend"), backend)


class TestSchedulerPickingOverrideWiring(unittest.TestCase):
    """End-to-end wiring: scheduler injects overrides + enriches Topaz action."""

    def _stub_state(
        self, root: Path
    ) -> Tuple[SampleState, QCMetrics]:
        state = SampleState(
            project_id="p1",
            project_path=str(root),
            stage="CtfFind",
            status="running",
        )
        qc = QCMetrics(motioncorr_completed=True, ctf_completed=True)
        return state, qc

    def test_scheduler_populates_action_parameters_from_overrides(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            star = root / "m.star"
            star.write_text("", encoding="utf-8")
            sched = Scheduler(project_root=root)
            stub = self._stub_state(root)
            sched.scan_project = lambda sequence_override=None, persist_state=True: stub  # type: ignore[method-assign]

            action, result = sched.run_next(
                dry_run=True,
                preferred_picking_backend="topaz",
                picking_overrides={
                    "micrographs_star_path": str(star),
                    "particle_diameter_angstrom": 200.0,
                    "topaz_threshold": -6.0,
                    "topaz_device": "cpu",
                },
            )
            self.assertEqual(action.name.value, "run_autopick")
            self.assertEqual(action.parameters.get("picking_backend"), "topaz")
            self.assertEqual(action.parameters.get("micrographs_star_path"), str(star))
            self.assertEqual(action.parameters.get("particle_diameter_angstrom"), 200.0)
            self.assertEqual(action.parameters.get("topaz_threshold"), -6.0)
            self.assertEqual(action.parameters.get("topaz_device"), "cpu")
            self.assertIsInstance(result, dict)
            md = result.get("metadata") or {}
            self.assertEqual(md.get("picking_backend"), "topaz")

    def test_default_backend_preserved_without_override(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sched = Scheduler(project_root=root)
            stub = self._stub_state(root)
            sched.scan_project = lambda sequence_override=None, persist_state=True: stub  # type: ignore[method-assign]

            action, _ = sched.run_next(dry_run=True)
            self.assertEqual(action.name.value, "run_autopick")
            self.assertEqual(action.parameters.get("picking_backend"), "relion_log")


if __name__ == "__main__":
    unittest.main()
