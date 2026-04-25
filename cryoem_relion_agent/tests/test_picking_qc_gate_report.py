"""End-to-end report tests for the Picking QC gate.

These tests pre-populate ``sample_state.json`` with a ``last_execution`` that
contains a failed Topaz ``picking_qc`` payload. Then ``generate_report`` is
invoked in read-only mode (no state refresh) and the resulting markdown must:
- show next_action == request_human_review (not run_autopick),
- include a "Picking QC Gate" section,
- not append a new action_history entry.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.scheduler import Scheduler
from schemas.sample_state import SampleState
from tests.helpers.fixture_projects import build_fixture_project


def _failed_qc_payload() -> dict:
    return {
        "qc_status": "failed_qc",
        "qc_flags": [
            "severe_under_picking",
            "many_empty_micrographs",
            "low_f1_against_ground_truth",
        ],
        "recommended_next_action": "threshold_sweep_or_train_topaz",
        "n_picks": 43,
        "n_micrographs_total": 29,
        "zero_pick_micrograph_ratio": 0.551724,
        "ground_truth_evaluation": {
            "precision": 0.069767,
            "recall": 0.000558,
            "f1": 0.001108,
        },
    }


def _seed_failed_topaz_state(proj: Path) -> Path:
    """Write sample_state.json with a Topaz failed_qc last_execution and return path."""
    agent_dir = proj / ".cryoem_agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    state_path = agent_dir / "sample_state.json"
    seed = SampleState(
        project_id="fixture_pick",
        project_path=str(proj.resolve()),
        stage="CtfFind",
        status="running",
        last_execution={
            "action_name": "run_autopick",
            "picking_backend": "topaz",
            "status": "success",
            "metadata": {
                "picking_backend": "topaz",
                "picking_qc": _failed_qc_payload(),
            },
        },
    )
    state_path.write_text(
        json.dumps(seed.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return state_path


class TestPickingQCGateReport(unittest.TestCase):
    def test_generate_report_shows_picking_qc_gate_when_last_execution_failed_qc(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            proj = build_fixture_project(root, "picking_ready")
            _seed_failed_topaz_state(proj)

            sched = Scheduler(project_root=proj)
            out_md = proj / "report.md"
            report = sched.generate_report(out_md=out_md, refresh_state=False)

            self.assertIsNotNone(report.next_action)
            self.assertEqual(report.next_action.name.value, "request_human_review")
            self.assertEqual(
                report.next_action.parameters.get("branch_name"), "picking_qc"
            )

            md = out_md.read_text(encoding="utf-8")
            self.assertIn("Picking QC Gate", md)
            self.assertIn("failed_qc", md)
            self.assertIn("threshold_sweep_or_train_topaz", md)

    def test_next_action_is_request_human_review_not_run_autopick(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            proj = build_fixture_project(root, "picking_ready")
            _seed_failed_topaz_state(proj)

            sched = Scheduler(project_root=proj)
            report = sched.generate_report(refresh_state=False)
            self.assertIsNotNone(report.next_action)
            self.assertEqual(report.next_action.name.value, "request_human_review")
            self.assertNotEqual(report.next_action.name.value, "run_autopick")

    def test_generate_report_does_not_append_action_history(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            proj = build_fixture_project(root, "picking_ready")
            _seed_failed_topaz_state(proj)

            sched = Scheduler(project_root=proj)
            history_before = list(sched.history_manager.retrieve_action_history())
            sched.generate_report(refresh_state=False)
            history_after = list(sched.history_manager.retrieve_action_history())
            self.assertEqual(
                len(history_before),
                len(history_after),
                msg="generate_report must not append action_history",
            )

    def test_scan_project_preserves_last_execution_from_persisted_state(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            proj = build_fixture_project(root, "picking_ready")
            _seed_failed_topaz_state(proj)

            sched = Scheduler(project_root=proj)
            state, _qc = sched.scan_project(persist_state=False)
            self.assertIsNotNone(state.last_execution)
            self.assertIn("picking_qc", state.last_execution.get("metadata", {}))


if __name__ == "__main__":
    unittest.main()
