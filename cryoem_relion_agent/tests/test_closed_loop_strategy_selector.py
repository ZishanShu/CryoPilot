from agent.closed_loop.evidence_loader import EvidenceCase
from agent.closed_loop.retrieval import retrieve_candidate_strategies
from agent.closed_loop.strategy_selector import select_rerun_strategy


def test_selector_prefers_topaz_when_picking_high_risk():
    case = EvidenceCase(
        case_id="c", dataset_id="EMPIAR-10017", run_root="/tmp/x",
        llm_assessments_by_stage={
            "particle_picking": {"risk_level": "high", "accept_for_downstream": False},
        },
        llm_overall={"accept_backend_execution": True, "accept_for_scientific_claim": False},
    )
    cands = retrieve_candidate_strategies(case, extra_topaz_pred_dirs={"-6": "/tmp/preds"})
    plan = select_rerun_strategy(case, cands)
    assert plan["selected_strategy_id"] == "topaz_threshold_sweep"
    assert plan["blocked_real_map2pdb"] is True
