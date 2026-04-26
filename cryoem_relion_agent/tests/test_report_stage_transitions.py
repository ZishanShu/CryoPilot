"""Tests for compact stage-transition and branch summaries in reports."""

from __future__ import annotations

import unittest

from agent.reporter import Reporter
from schemas.action_history import ActionHistoryEntry
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState


class TestReportStageTransitions(unittest.TestCase):
    def test_report_contains_main_path_stage_summary(self) -> None:
        state = SampleState(project_id="p1", project_path="/tmp/p", stage="Refine3D", status="running")
        qc = QCMetrics(refine3d_completed=True)
        history = [
            ActionHistoryEntry(run_id="r1", stage="Picking", action_name="run_picking_extract", branch_name="main"),
            ActionHistoryEntry(run_id="r1", stage="Class2D", action_name="run_class2d_cleanup", branch_name="main"),
            ActionHistoryEntry(run_id="r1", stage="Refine3D", action_name="run_refine_ctf_polish", branch_name="main"),
        ]
        report = Reporter().build_report(
            state=state,
            qc=qc,
            recent_action=None,
            next_action=None,
            action_history=history,
            current_run_id="r1",
        )
        md = report.to_markdown()
        self.assertIn("Run ID: r1", md)
        self.assertIn("Picking -> Class2D -> Refine3D", md)

    def test_report_contains_branch_summary_when_present(self) -> None:
        state = SampleState(project_id="p1", project_path="/tmp/p", stage="PostProcess", status="running")
        qc = QCMetrics(refine3d_completed=True, current_resolution=3.8)
        history = [
            ActionHistoryEntry(run_id="r2", stage="PostProcess", action_name="delegate_structure_router", branch_name="structure_router"),
            ActionHistoryEntry(run_id="r2", stage="PostProcess", action_name="run_modelangelo", branch_name="modelangelo"),
            ActionHistoryEntry(run_id="r2", stage="Class2D", action_name="request_human_review", branch_name="heterogeneity"),
        ]
        report = Reporter().build_report(
            state=state,
            qc=qc,
            recent_action=None,
            next_action=None,
            action_history=history,
            current_run_id="r2",
        )
        md = report.to_markdown()
        self.assertIn("Branch: structure_router", md)
        self.assertIn("Branch: modelangelo", md)
        self.assertIn("Branch: heterogeneity", md)

    def test_report_candidate_routing_block_shows_provenance_fields(self) -> None:
        state = SampleState(
            project_id="p1",
            project_path="/tmp/p",
            stage="PostProcess",
            status="running",
            candidate_maps=[
                {
                    "source_backend": "cryodrgn_ai",
                    "source_run_dir": "/tmp/p/CryoDRGN_AI/job001",
                    "map_path": "/tmp/p/CryoDRGN_AI/job001/sampled_volumes/vol_000.mrc",
                    "latent_source": "/tmp/p/CryoDRGN_AI/job001/representative_z.txt",
                    "weights_path": "/tmp/p/CryoDRGN_AI/job001/abinit/weights.pkl",
                }
            ],
            last_execution={
                "action_name": "run_cryodrgn_ai",
                "status": "completed",
                "candidate_maps": [
                    {
                        "source_backend": "cryodrgn_ai",
                        "source_run_dir": "/tmp/p/CryoDRGN_AI/job001",
                        "map_path": "/tmp/p/CryoDRGN_AI/job001/sampled_volumes/vol_000.mrc",
                        "latent_source": "/tmp/p/CryoDRGN_AI/job001/representative_z.txt",
                        "weights_path": "/tmp/p/CryoDRGN_AI/job001/abinit/weights.pkl",
                    }
                ],
            },
        )
        qc = QCMetrics(refine3d_completed=True, current_resolution=3.2)
        report = Reporter().build_report(
            state=state,
            qc=qc,
            recent_action=None,
            next_action=None,
            action_history=[],
            current_run_id=None,
        )
        md = report.to_markdown()
        self.assertIn("## Candidate Map Routing", md)
        self.assertIn("vol_000.mrc", md)
        self.assertIn("weights.pkl", md)
        self.assertIn("representative_z.txt", md)
        self.assertIn("modelangelo", md)
        self.assertIn("## Candidate Map QC Ranking", md)
        self.assertIn("Selected candidate index", md)
        self.assertIn("## Candidate Map Batch Model-Building Plan", md)
        self.assertIn("Top-K count", md)


if __name__ == "__main__":
    unittest.main()

