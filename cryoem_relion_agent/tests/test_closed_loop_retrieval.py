from agent.closed_loop.evidence_loader import EvidenceCase
from agent.closed_loop.retrieval import retrieve_candidate_strategies


def _case_with_pick_high_risk() -> EvidenceCase:
    return EvidenceCase(
        case_id="c", dataset_id="EMPIAR-10017", run_root="/tmp/x",
        llm_assessments_by_stage={
            "particle_picking": {"risk_level": "high", "accept_for_downstream": False},
            "reconstruction": {"risk_level": "high", "accept_for_downstream": False},
        },
        llm_overall={"accept_backend_execution": True, "accept_for_scientific_claim": False},
    )


def test_returns_topaz_when_picking_high_risk_and_preds_present():
    case = _case_with_pick_high_risk()
    cands = retrieve_candidate_strategies(case, extra_topaz_pred_dirs={"-6": "/tmp/preds"})
    ids = [c["strategy_id"] for c in cands]
    assert "topaz_threshold_sweep" in ids
    topaz = next(c for c in cands if c["strategy_id"] == "topaz_threshold_sweep")
    assert topaz["availability"] == "available"


def test_map2pdb_blocked_when_recon_not_accepted():
    case = _case_with_pick_high_risk()
    cands = retrieve_candidate_strategies(case)
    m = next(c for c in cands if c["strategy_id"] == "map2pdb_real_run")
    assert m["mode"] == "plan-only"
    assert m["availability"] == "unavailable"
