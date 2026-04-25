"""Scheduler trace tests for picker backend metadata."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app.scheduler import Scheduler
from schemas.action import Action, ActionName
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState


class TestPickerRouterScheduler(unittest.TestCase):
    def test_scheduler_records_picker_backend_in_same_run_chain(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sched = Scheduler(project_root=root)
            state = SampleState(project_id="p1", project_path=str(root), stage="CtfFind", status="running")
            qc = QCMetrics(motioncorr_completed=True, ctf_completed=True)
            sched.scan_project = lambda sequence_override=None, persist_state=True: (state, qc)  # type: ignore[method-assign]

            action, _result = sched.run_next(dry_run=True)
            self.assertEqual(action.name.value, "run_autopick")
            self.assertIn(action.parameters.get("picking_backend"), {"relion_log", "topaz", "cryolo"})

            items = sched.history_manager.retrieve_action_history()
            self.assertGreaterEqual(len(items), 2)
            self.assertEqual(len({x.run_id for x in items}), 1)

            decision = [x for x in items if x.execution_status == "decided"][-1]
            execution = [x for x in items if x.action_name == "run_autopick"][-1]
            self.assertEqual(execution.parent_decision_id, decision.entry_id)
            self.assertTrue(bool(execution.previous_entry_id))
            self.assertIn("picking_backend=", execution.output_summary)


class TestSchedulerPickingQCAttachment(unittest.TestCase):
    """``_attach_picking_qc`` merges QC into the execution result without crashing."""

    def _star(self, path: Path, lines: list[str]) -> None:
        body = (
            "data_\n\nloop_\n"
            "_rlnMicrographName #1\n"
            "_rlnCoordinateX #2\n"
            "_rlnCoordinateY #3\n"
            "_rlnAutopickFigureOfMerit #4\n"
            + "\n".join(lines)
            + "\n"
        )
        path.write_text(body, encoding="utf-8")

    def test_picking_qc_attached_when_topaz_picks_present(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            picks = root / "picks.star"
            self._star(picks, ["m1.mrc 1 2 -1.0", "m1.mrc 3 4 -1.5"])
            result: dict = {
                "action_name": "run_autopick",
                "status": "success",
                "topaz_output_path": str(picks),
                "metadata": {"topaz_output_path": str(picks)},
            }
            action = Action(
                name=ActionName.RUN_AUTOPICK,
                reason="t",
                parameters={
                    "picking_backend": "topaz",
                    "picking_qc_thresholds": {
                        "min_total_picks": 1,
                        "min_mean_picks_per_micrograph": 0.1,
                    },
                },
                priority=78,
            )
            Scheduler._attach_picking_qc(result, action)
            self.assertIn("picking_qc", result)
            self.assertEqual(result["picking_qc"]["n_picks"], 2)
            self.assertEqual(result["metadata"]["picking_qc"]["n_picks"], 2)
            self.assertEqual(result["picking_qc"]["qc_status"], "passed")

    def test_picking_qc_error_recorded_when_evaluator_raises(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            picks = root / "picks.star"
            picks.write_text("garbage\n", encoding="utf-8")
            result: dict = {
                "action_name": "run_autopick",
                "status": "success",
                "topaz_output_path": str(picks),
                "metadata": {},
            }
            action = Action(
                name=ActionName.RUN_AUTOPICK,
                reason="t",
                parameters={"picking_backend": "topaz"},
                priority=78,
            )
            with patch(
                "agent.picking_qc.evaluate_picking_output",
                side_effect=RuntimeError("boom"),
            ):
                Scheduler._attach_picking_qc(result, action)
            self.assertEqual(result.get("picking_qc_error"), "boom")
            self.assertEqual(result["metadata"]["picking_qc_error"], "boom")
            self.assertNotIn("picking_qc", result)

    def test_picking_qc_skipped_when_picks_path_missing(self) -> None:
        result: dict = {"action_name": "run_autopick", "status": "success", "metadata": {}}
        action = Action(
            name=ActionName.RUN_AUTOPICK,
            reason="t",
            parameters={"picking_backend": "topaz"},
            priority=78,
        )
        Scheduler._attach_picking_qc(result, action)
        # Nothing crashes, nothing added.
        self.assertNotIn("picking_qc", result)
        self.assertNotIn("picking_qc_error", result)


if __name__ == "__main__":
    unittest.main()
