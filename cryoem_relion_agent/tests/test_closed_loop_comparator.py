from agent.closed_loop.comparator import build_comparison, select_final
from agent.closed_loop.evidence_loader import EvidenceCase


def _baseline() -> EvidenceCase:
    return EvidenceCase(
        case_id="baseline", dataset_id="EMPIAR-10017", run_root="/tmp/x",
        metrics={
            "n_micrographs": 84, "total_topaz_coords_used": 7404,
            "n_extracted_particles": 7404, "n_particle_stacks": 84,
            "reconstruction_map_exists": True, "reconstruction_map_size_bytes": 67109888,
        },
        llm_overall={"accept_backend_execution": True, "accept_for_scientific_claim": False, "risk_level": "high"},
    )


def test_compare_baseline_only():
    rows = build_comparison(_baseline(), [])
    assert len(rows) == 1
    final = select_final(_baseline(), rows, [])
    assert final["accepted_for_scientific_downstream"] is False
    assert final["final_action"] == "STOP_OR_RERUN_REQUIRED"


def test_compare_with_one_candidate():
    base = _baseline()
    rerun = EvidenceCase(
        case_id="topaz_t-6", dataset_id="EMPIAR-10017", run_root="/tmp/y",
        metrics={
            "n_micrographs": 84, "total_topaz_coords_used": 8000,
            "n_extracted_particles": 8000, "n_particle_stacks": 84,
            "reconstruction_map_exists": True, "reconstruction_map_size_bytes": 67109888,
        },
        llm_overall={"accept_backend_execution": True, "accept_for_scientific_claim": False, "risk_level": "high"},
    )
    rows = build_comparison(base, [{"case": rerun, "strategy_id": "topaz_threshold_sweep",
                                    "stage": "particle_picking", "backend": "topaz",
                                    "config": "threshold=-6"}])
    final = select_final(base, rows, [{"case": rerun}])
    assert final["selected_for_backend_execution"] == "topaz_t-6"
    assert final["accepted_for_scientific_downstream"] is False
