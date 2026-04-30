import json
from pathlib import Path

from agent.closed_loop.evidence_loader import load_evidence_case


def _make_synthetic(tmp_path: Path) -> Path:
    root = tmp_path / "case"
    (root / "rule_based").mkdir(parents=True)
    (root / "llm_supervised").mkdir(parents=True)
    (root / "workflow_summary.json").write_text(json.dumps({
        "dataset_id": "EMPIAR-10017",
        "project_id": "synth",
        "metrics": {
            "n_micrographs": 2, "total_topaz_coords_used": 10,
            "n_extracted_particles": 10, "n_particle_stacks": 2,
            "reconstruction_map_exists": True, "reconstruction_map_size_bytes": 1024,
        },
        "stage_status": {
            "picking": "accepted_for_extraction",
            "extraction": "accepted_for_reconstruction",
            "reconstruction": "accepted_for_map2pdb",
            "map2pdb": "recorded",
        },
        "selected_micrographs": [],
    }))
    (root / "rule_based" / "stage_decisions.jsonl").write_text(
        json.dumps({"stage": "particle_picking", "status": "accepted_for_extraction"}) + "\n" +
        json.dumps({"stage": "reconstruction", "status": "accepted_for_map2pdb"}) + "\n"
    )
    (root / "llm_supervised" / "llm_stagewise_result.json").write_text(json.dumps({
        "stage_assessments": [
            {"stage": "particle_picking", "risk_level": "high", "accept_for_downstream": False},
            {"stage": "reconstruction", "risk_level": "high", "accept_for_downstream": False},
        ],
        "overall_assessment": {
            "accept_backend_execution": True,
            "accept_for_scientific_claim": False,
            "risk_level": "high",
            "summary": "smoke",
        },
    }))
    return root


def test_load_synthetic(tmp_path):
    root = _make_synthetic(tmp_path)
    case = load_evidence_case(root)
    assert case.dataset_id == "EMPIAR-10017"
    assert case.metrics["n_extracted_particles"] == 10
    assert "particle_picking" in case.rule_decisions_by_stage
    assert case.llm_overall["accept_for_scientific_claim"] is False


def test_load_real_fixture():
    root = Path(__file__).resolve().parents[1] / "examples/evidence_cases/empiar10017_full84"
    if not root.exists():
        return
    case = load_evidence_case(root)
    assert case.dataset_id == "EMPIAR-10017"
    assert case.metrics.get("n_micrographs") == 84
    assert case.llm_overall.get("accept_backend_execution") is True
