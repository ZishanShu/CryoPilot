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
from schemas.action_history import ActionHistoryEntry
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
        "n_micrographs_with_picks": 13,
        "n_micrographs_total": 29,
        "zero_pick_micrograph_ratio": 0.551724,
        "matching_tolerance_px": 80.0,
        "coordinate_scale": 4.0,
        "ground_truth_evaluation": {
            "precision": 0.069767,
            "recall": 0.000558,
            "f1": 0.001108,
            "matching_tolerance_px": 80.0,
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

    def test_picking_backend_in_report_is_topaz_even_when_history_has_relion_log(self) -> None:
        """If a later relion_log dry-run is logged in action_history, the report must
        still attribute the failed Picking QC to the Topaz execution that produced it."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            proj = build_fixture_project(root, "picking_ready")
            _seed_failed_topaz_state(proj)

            sched = Scheduler(project_root=proj)
            # Seed action_history with a later relion_log dry-run that would otherwise
            # poison _extract_picking_backend's history fallback.
            sched.history_manager.add_action_to_history(
                ActionHistoryEntry(
                    run_id="later_run",
                    stage="CtfFind",
                    action_name="run_autopick",
                    branch_name="main",
                    decision_reason="dry-run sanity",
                    execution_status="decided",
                    output_summary="policy decided run_autopick (picking_backend=relion_log)",
                )
            )

            out_md = proj / "report.md"
            report = sched.generate_report(out_md=out_md, refresh_state=False)

            self.assertEqual(report.next_action.name.value, "request_human_review")
            self.assertEqual(report.extra.get("picking_backend"), "topaz")

            md = out_md.read_text(encoding="utf-8")
            self.assertIn("Picking QC Gate", md)
            self.assertIn("Picking backend: topaz", md)
            self.assertIn("Backend: topaz", md)
            # Compact gate section contributes "Status: failed_qc" and the rec.
            self.assertIn("threshold_sweep_or_train_topaz", md)
            # The Picking QC section's Backend: must NOT be relion_log.
            qc_section = md.split("## Picking QC", 1)[-1]
            self.assertNotIn("Backend: relion_log", qc_section)

    def test_scan_project_preserves_last_execution_from_persisted_state(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            proj = build_fixture_project(root, "picking_ready")
            _seed_failed_topaz_state(proj)

            sched = Scheduler(project_root=proj)
            state, _qc = sched.scan_project(persist_state=False)
            self.assertIsNotNone(state.last_execution)
            self.assertIn("picking_qc", state.last_execution.get("metadata", {}))

    def test_run_next_persists_gate_evidence_and_generate_report_remains_blocked(self) -> None:
        """End-to-end idempotency: run-next dry-run overwrites last_execution with a
        request_human_review result, but the gate evidence is preserved so a subsequent
        generate-report still returns request_human_review and shows the QC gate."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            proj = build_fixture_project(root, "picking_ready")
            _seed_failed_topaz_state(proj)

            # Step 1: run-next dry-run — gate fires, returns request_human_review.
            sched = Scheduler(project_root=proj)
            action, result = sched.run_next(dry_run=True)
            self.assertEqual(action.name.value, "request_human_review")
            self.assertEqual(action.parameters.get("branch_name"), "picking_qc")
            self.assertEqual(action.parameters.get("picking_qc_status"), "failed_qc")

            # Step 2: result must carry gate evidence so last_execution is recoverable.
            self.assertEqual(result.get("branch_name"), "picking_qc")
            self.assertEqual(result.get("picking_backend"), "topaz")
            self.assertEqual(result.get("picking_qc_status"), "failed_qc")
            self.assertIsNotNone(result.get("picking_qc_flags"))
            md = result.get("metadata") or {}
            self.assertEqual(md.get("branch_name"), "picking_qc")
            self.assertEqual(md.get("picking_qc_status"), "failed_qc")

            # Step 3: reload sample_state.json — last_execution must still carry evidence.
            import json as _json
            state_path = proj / ".cryoem_agent" / "sample_state.json"
            raw = _json.loads(state_path.read_text(encoding="utf-8"))
            le = raw.get("last_execution") or {}
            self.assertEqual(le.get("branch_name"), "picking_qc")
            self.assertEqual(le.get("picking_qc_status"), "failed_qc")

            # Step 4: generate-report on a fresh scheduler instance reads the
            # persisted state and must still return request_human_review.
            sched2 = Scheduler(project_root=proj)
            out_md = proj / "report_after_run_next.md"
            report = sched2.generate_report(out_md=out_md, refresh_state=False)
            self.assertEqual(report.next_action.name.value, "request_human_review")
            self.assertEqual(report.next_action.parameters.get("branch_name"), "picking_qc")
            self.assertEqual(report.extra.get("picking_backend"), "topaz")

            md_text = out_md.read_text(encoding="utf-8")
            self.assertIn("Picking QC Gate", md_text)
            self.assertIn("threshold_sweep_or_train_topaz", md_text)
            self.assertNotIn("run_autopick", md_text.split("Next action:")[1][:50])

    def test_run_next_dry_run_command_uses_picking_qc_branch_label(self) -> None:
        """The dry-run echo command for the gate must say "picking_qc branch",
        not "heterogeneity branch"."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            proj = build_fixture_project(root, "picking_ready")
            _seed_failed_topaz_state(proj)

            sched = Scheduler(project_root=proj)
            _action, result = sched.run_next(dry_run=True)
            cmd = str(result.get("command") or "")
            self.assertIn("picking_qc branch", cmd)
            self.assertNotIn("heterogeneity branch", cmd)

    def test_report_shows_full_picking_qc_metrics_after_gate_persist(self) -> None:
        """After run-next dry-runs the gate, generate-report must surface
        n_micrographs_with_picks=13 / 29 and tol=80.0px (not Nonepx)."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            proj = build_fixture_project(root, "picking_ready")
            _seed_failed_topaz_state(proj)

            sched = Scheduler(project_root=proj)
            sched.run_next(dry_run=True)

            sched2 = Scheduler(project_root=proj)
            out_md = proj / "report.md"
            sched2.generate_report(out_md=out_md, refresh_state=False)
            md_text = out_md.read_text(encoding="utf-8")

            self.assertIn("Micrographs with picks: 13 / 29", md_text)
            self.assertIn("tol=80.0px", md_text)
            self.assertIn("Human review requested for picking_qc branch", md_text)
            self.assertNotIn("heterogeneity branch", md_text)
            self.assertNotIn("None / 29", md_text)
            self.assertNotIn("Nonepx", md_text)

    def test_regression_relion_log_run_without_failed_qc_unblocked(self) -> None:
        """Regression: a relion_log last_execution without picking_qc must NOT trigger
        the gate; the pipeline must return run_autopick normally."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            proj = build_fixture_project(root, "picking_ready")
            # Seed a relion_log run (no picking_qc) as the last_execution.
            agent_dir = proj / ".cryoem_agent"
            agent_dir.mkdir(parents=True, exist_ok=True)
            import json as _json
            seed = {
                "project_id": "fixture_pick",
                "project_path": str(proj.resolve()),
                "stage": "CtfFind",
                "status": "running",
                "n_particles_picked": 0,
                "n_particles_after_2d": 0,
                "last_execution": {
                    "action_name": "run_autopick",
                    "picking_backend": "relion_log",
                    "status": "dry_run",
                    "metadata": {"picking_backend": "relion_log"},
                },
                "updated_at": "2025-01-01T00:00:00Z",
            }
            (agent_dir / "sample_state.json").write_text(
                _json.dumps(seed, indent=2), encoding="utf-8"
            )
            sched = Scheduler(project_root=proj)
            report = sched.generate_report(refresh_state=False)
            self.assertNotEqual(report.next_action.name.value, "request_human_review")
            self.assertEqual(report.next_action.name.value, "run_autopick")


if __name__ == "__main__":
    unittest.main()
