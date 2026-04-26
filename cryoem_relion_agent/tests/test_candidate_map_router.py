from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.candidate_map_router import create_candidate_map_routing_records, read_candidate_maps
from agent.reporter import Reporter
from app.scheduler import Scheduler
from schemas.action import ActionName
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState


def _candidate(path: Path, idx: int) -> dict:
    return {
        "source_backend": "cryodrgn_ai",
        "source_run_dir": str(path.parent),
        "map_path": str(path),
        "latent_source": str(path.parent / "representative_z.txt"),
        "weights_path": str(path.parent / "abinit" / "weights.pkl"),
        "qc_status": "not_evaluated",
        "recommended_next_action": "route_to_model_building",
    }


class TestCandidateMapRouter(unittest.TestCase):
    def test_reads_5_maps_and_creates_5_records(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            seq = root / "seq.fasta"
            seq.write_text(">a\nA\n", encoding="utf-8")
            candidates = []
            for idx in range(6):
                m = root / f"candidate_{idx}.mrc"
                m.write_text("map", encoding="utf-8")
                candidates.append(_candidate(m, idx))
            state = SampleState(
                project_id="p",
                project_path=str(root),
                stage="PostProcess",
                status="running",
                sequence_path=str(seq),
                last_execution={"candidate_maps": candidates},
            )

            self.assertEqual(len(read_candidate_maps(state)), 6)
            records = create_candidate_map_routing_records(state)
            self.assertEqual(len(records), 5)
            self.assertTrue(all(r["recommended_backend"] == "modelangelo" for r in records))
            self.assertTrue(all(r["downstream_action"] == "run_modelangelo" for r in records))
            self.assertEqual(records[0]["candidate_index"], 0)

    def test_missing_map_path_marked_unavailable(self) -> None:
        state = {
            "project_path": "/tmp/project",
            "sequence_path": "/tmp/seq.fasta",
            "candidate_maps": [{"source_backend": "cryodrgn_ai", "map_path": "/missing/map.mrc"}],
        }
        records = create_candidate_map_routing_records(state)
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0]["routing_available"])
        self.assertEqual(records[0]["routing_reason"], "map_path_unavailable")
        self.assertFalse(records[0]["preflight"]["available"])

    def test_scheduler_passes_candidate_map_into_model_building_action(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            seq = root / "seq.fasta"
            seq.write_text(">a\nA\n", encoding="utf-8")
            base_map = root / "base_map.mrc"
            base_map.write_text("map", encoding="utf-8")
            m = root / "candidate.mrc"
            m.write_text("map", encoding="utf-8")
            state = SampleState(
                project_id="p",
                project_path=str(root),
                stage="PostProcess",
                status="running",
                map_path=str(base_map),
                sequence_path=str(seq),
                last_execution={"candidate_maps": [_candidate(m, 0)]},
            )
            qc = QCMetrics(
                motioncorr_completed=True,
                ctf_completed=True,
                picking_completed=True,
                class2d_completed=True,
                init3d_completed=True,
                refine3d_completed=True,
                postprocess_completed=True,
                current_resolution=3.5,
            )
            action = Scheduler(project_root=root)._resolve_next_action(state, qc)
            self.assertEqual(action.name, ActionName.RUN_MODELANGELO)
            self.assertEqual(action.parameters.get("branch_name"), "candidate_map_model_building")
            self.assertEqual(action.parameters.get("candidate_index"), 0)
            self.assertEqual(action.parameters.get("map_path"), str(m))
            self.assertIsInstance(action.parameters.get("candidate_map_routing_records"), list)

    def test_report_includes_candidate_map_routing_section(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            seq = root / "seq.fasta"
            seq.write_text(">a\nA\n", encoding="utf-8")
            m = root / "sampled_volumes" / "vol_000.mrc"
            m.parent.mkdir(parents=True, exist_ok=True)
            m.write_text("map", encoding="utf-8")
            (root / "representative_z.txt").write_text("0 0 0\n", encoding="utf-8")
            (root / "abinit").mkdir(parents=True, exist_ok=True)
            (root / "abinit" / "weights.pkl").write_bytes(b"w")
            state = SampleState(
                project_id="p",
                project_path=str(root),
                stage="PostProcess",
                status="running",
                candidate_maps=[_candidate(m, 0)],
                sequence_path=str(seq),
                last_execution={"candidate_maps": [_candidate(m, 0)]},
            )
            report = Reporter().build_report(
                state=state,
                qc=QCMetrics(refine3d_completed=True, postprocess_completed=True, current_resolution=3.5),
                recent_action=None,
                next_action=None,
            )
            md = report.to_markdown()
            self.assertIn("## Candidate Map Routing", md)
            self.assertIn("Candidate count: 1", md)
            self.assertIn(str(m), md)
            self.assertIn("backend: modelangelo", md)
            self.assertIn("weights.pkl", md)
            self.assertIn("representative_z.txt", md)

    def test_no_candidate_maps_keeps_existing_behavior(self) -> None:
        state = SampleState(
            project_id="p",
            project_path="/tmp/project",
            stage="PostProcess",
            status="running",
            map_path="/tmp/map.mrc",
            sequence_path="/tmp/seq.fasta",
        )
        qc = QCMetrics(
            motioncorr_completed=True,
            ctf_completed=True,
            picking_completed=True,
            class2d_completed=True,
            init3d_completed=True,
            refine3d_completed=True,
            postprocess_completed=True,
            current_resolution=3.8,
        )
        action = Scheduler(project_root=Path("/tmp/project"))._resolve_next_action(state, qc)
        self.assertEqual(action.name, ActionName.RUN_MODELANGELO)
        self.assertNotEqual(action.parameters.get("branch_name"), "candidate_map_model_building")


if __name__ == "__main__":
    unittest.main()
