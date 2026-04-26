from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.candidate_map_batch_model_building import (
    build_candidate_model_building_plan,
    build_candidate_model_building_plans_from_state,
)
from schemas.sample_state import SampleState


class TestCandidateMapBatchModelBuilding(unittest.TestCase):
    def test_batch_plan_returns_top_k_candidates(self) -> None:
        records = [
            {
                "candidate_index": idx,
                "project_path": "/tmp/proj",
                "qc_rank": idx,
                "qc_score": 10.0 - idx,
                "qc_status": "passed",
                "map_path": f"/tmp/proj/sampled_volumes/vol_{idx:03d}.mrc",
                "source_backend": "cryodrgn_ai",
                "source_run_dir": "/tmp/proj/CryoDRGN_AI/job_agent",
                "latent_source": "/tmp/proj/representative_z.txt",
                "weights_path": "/tmp/proj/abinit/weights.pkl",
                "recommended_backend": "modelangelo",
                "downstream_action": "run_modelangelo",
                "routing_reason": "sequence_path_present_modelangelo_default",
                "preflight": {"available": True, "execution_mode": "dry_run"},
                "action_parameters": {"output_dir": f"/tmp/proj/CandidateMapModelBuilding/candidate_{idx:03d}"},
            }
            for idx in range(5)
        ]
        plans = build_candidate_model_building_plan(records, top_k=3, dry_run=True)
        self.assertEqual(len(plans), 3)
        self.assertEqual([p["candidate_index"] for p in plans], [0, 1, 2])
        self.assertTrue(all(p["dry_run"] is True for p in plans))
        self.assertTrue(all(p["planned_status"] == "planned" for p in plans))

    def test_batch_plan_preserves_qc_and_provenance_fields(self) -> None:
        record = {
            "candidate_index": 2,
            "project_path": "/tmp/proj",
            "qc_rank": 0,
            "qc_score": 87.5,
            "qc_status": "passed",
            "map_path": "/tmp/proj/sampled_volumes/vol_002.mrc",
            "source_backend": "cryodrgn_ai",
            "source_run_dir": "/tmp/proj/CryoDRGN_AI/job_agent",
            "latent_source": "/tmp/proj/representative_z.txt",
            "weights_path": "/tmp/proj/abinit/weights.pkl",
            "recommended_backend": "modelangelo",
            "downstream_action": "run_modelangelo",
            "routing_reason": "highest_qc_score",
            "preflight": {
                "available": True,
                "execution_mode": "dry_run",
                "command": "model_angelo build -v /tmp/proj/sampled_volumes/vol_002.mrc -pf /tmp/proj/seq.fasta -o /tmp/proj/CandidateMapModelBuilding/candidate_002",
            },
            "action_parameters": {"output_dir": "/tmp/proj/CandidateMapModelBuilding/candidate_002"},
        }
        plans = build_candidate_model_building_plan([record], top_k=5, dry_run=True)
        self.assertEqual(len(plans), 1)
        plan = plans[0]
        self.assertEqual(plan["qc_rank"], 0)
        self.assertAlmostEqual(float(plan["qc_score"]), 87.5)
        self.assertEqual(plan["latent_source"], "/tmp/proj/representative_z.txt")
        self.assertEqual(plan["weights_path"], "/tmp/proj/abinit/weights.pkl")
        self.assertEqual(plan["output_dir"], "/tmp/proj/CandidateMapModelBuilding/candidate_002")
        self.assertEqual(plan["selected_backend"], "modelangelo")
        self.assertEqual(plan["selected_action"], "run_modelangelo")
        self.assertIn("model_angelo build", plan["command"] or "")
        self.assertIsInstance(plan["model_building_preflight"], dict)

    def test_batch_plan_output_dir_is_deterministic_per_candidate(self) -> None:
        records = [
            {
                "candidate_index": idx,
                "project_path": "/tmp/proj",
                "qc_rank": idx,
                "qc_score": 1.0,
                "qc_status": "warning",
                "map_path": f"/tmp/proj/sampled_volumes/vol_{idx:03d}.mrc",
                "source_backend": "cryodrgn_ai",
                "source_run_dir": "/tmp/proj/CryoDRGN_AI/job_agent",
                "latent_source": "/tmp/proj/representative_z.txt",
                "weights_path": "/tmp/proj/abinit/weights.pkl",
                "recommended_backend": "modelangelo",
                "downstream_action": "run_modelangelo",
                "routing_reason": "highest_qc_score",
                "preflight": {"available": True, "execution_mode": "dry_run"},
            }
            for idx in range(3)
        ]
        plans = build_candidate_model_building_plan(records, top_k=3, dry_run=True)
        self.assertEqual(
            [p["output_dir"] for p in plans],
            [
                "/tmp/proj/CandidateMapModelBuilding/candidate_000",
                "/tmp/proj/CandidateMapModelBuilding/candidate_001",
                "/tmp/proj/CandidateMapModelBuilding/candidate_002",
            ],
        )

    def test_build_plans_from_state_no_candidates_returns_empty(self) -> None:
        state = SampleState(
            project_id="p",
            project_path="/tmp/proj",
            stage="PostProcess",
            status="running",
            sequence_path="/tmp/proj/seq.fasta",
        )
        plans = build_candidate_model_building_plans_from_state(state, top_k=5)
        self.assertEqual(plans, [])

    def test_build_plans_from_state_respects_top_k(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            seq = root / "seq.fasta"
            seq.write_text(">a\nA\n", encoding="utf-8")
            candidate_maps = []
            for idx in range(4):
                m = root / "sampled_volumes" / f"vol_{idx:03d}.mrc"
                m.parent.mkdir(parents=True, exist_ok=True)
                m.write_text("map", encoding="utf-8")
                candidate_maps.append(
                    {
                        "source_backend": "cryodrgn_ai",
                        "source_run_dir": str(root / "CryoDRGN_AI" / "job_agent"),
                        "map_path": str(m),
                        "latent_source": str(root / "representative_z.txt"),
                        "weights_path": str(root / "abinit" / "weights.pkl"),
                    }
                )
            state = SampleState(
                project_id="p",
                project_path=str(root),
                stage="PostProcess",
                status="running",
                sequence_path=str(seq),
                candidate_maps=candidate_maps,
                last_execution={"candidate_maps": candidate_maps},
            )
            plans = build_candidate_model_building_plans_from_state(state, top_k=2)
            self.assertEqual(len(plans), 2)
            self.assertTrue(all(str(p["output_dir"]).startswith(str(root / "CandidateMapModelBuilding")) for p in plans))


if __name__ == "__main__":
    unittest.main()
