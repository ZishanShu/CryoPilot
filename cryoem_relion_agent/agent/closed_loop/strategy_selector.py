"""Select a rerun strategy from retrieved candidates."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .evidence_loader import EvidenceCase


def select_rerun_strategy(
    case: EvidenceCase, candidates: List[Dict[str, Any]]
) -> Dict[str, Any]:
    pick_high = case.llm_assessments_by_stage.get("particle_picking", {}).get("risk_level") == "high"
    recon_high = case.llm_assessments_by_stage.get("reconstruction", {}).get("risk_level") == "high"
    map2pdb_assess = case.llm_assessments_by_stage.get("map2pdb", {})
    overall_accept_downstream = case.llm_overall.get(
        "accept_for_scientific_claim", case.llm_overall.get("accept_for_downstream")
    )

    by_id = {c["strategy_id"]: c for c in candidates}

    selected: Optional[Dict[str, Any]] = None
    rationale: List[str] = []

    if pick_high:
        topaz = by_id.get("topaz_threshold_sweep")
        if topaz and topaz["availability"] in ("available", "partial"):
            selected = topaz
            rationale.append("particle_picking high risk; selecting Topaz threshold sweep.")
        else:
            tpl = by_id.get("template_picker_fallback")
            if tpl and tpl["availability"] == "available":
                selected = tpl
                rationale.append("Topaz unavailable; falling back to template picker.")

    if selected is None and recon_high and not pick_high:
        cd = by_id.get("cryodrgn_ai_reconstruction")
        if cd and cd["availability"] in ("available", "partial"):
            selected = cd
            rationale.append("reconstruction high risk; selecting cryoDRGN-AI rerun.")

    if selected is None:
        # Block real map2pdb if reconstruction not scientifically accepted.
        if overall_accept_downstream is False:
            rationale.append(
                "LLM rejects scientific downstream; real map2pdb is blocked. "
                "Selecting Topaz threshold sweep as default rerun if available."
            )
            topaz = by_id.get("topaz_threshold_sweep")
            if topaz and topaz["availability"] in ("available", "partial"):
                selected = topaz
        if selected is None:
            for cand in sorted(candidates, key=lambda c: c["selection_priority"]):
                if cand["availability"] in ("available", "partial"):
                    selected = cand
                    rationale.append(f"Default to lowest-priority available: {cand['strategy_id']}.")
                    break

    if selected is None:
        selected = candidates[0] if candidates else {}
        rationale.append("No fully-available candidate; returning plan-only highest-priority.")

    plan = {
        "selected_strategy_id": selected.get("strategy_id"),
        "stage": selected.get("stage"),
        "backend": selected.get("backend"),
        "mode": selected.get("mode"),
        "availability": selected.get("availability"),
        "reason": " ".join(rationale) if rationale else selected.get("reason", ""),
        "commands_or_callable": _describe_commands(selected),
        "inputs": selected.get("required_inputs", []),
        "outputs": selected.get("expected_outputs", []),
        "stop_conditions": [
            "rerun produces zero particles",
            "LLM still rejects scientific downstream after rerun",
        ],
        "success_criteria": [
            "extraction yields particles >= baseline",
            "reconstruction map exists and is non-empty",
            "LLM risk_level decreases or QC metrics improve",
        ],
        "fallback_if_failed": _next_fallback(selected, candidates),
        "blocked_real_map2pdb": map2pdb_assess.get("accept_for_downstream") is False
        or overall_accept_downstream is False,
    }
    return plan


def _describe_commands(selected: Dict[str, Any]) -> List[str]:
    sid = selected.get("strategy_id")
    if sid == "topaz_threshold_sweep":
        return [
            "python -m scripts.run_same_dataset_agent_workflow "
            "--topaz-pred-dir <threshold_pred_dir> ...",
        ]
    if sid == "template_picker_fallback":
        return ["scripts/run_template_picker.py (plan)"]
    if sid == "relion_log_autopick_fallback":
        return ["relion_autopick --LoG (plan-only)"]
    if sid == "cryodrgn_ai_reconstruction":
        return ["cryodrgn_ai train ... (plan-only)"]
    if sid == "map2pdb_real_run":
        return ["python -m app.main run-next --project <relion_project>"]
    return []


def _next_fallback(selected: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Optional[str]:
    sid = selected.get("strategy_id")
    ordered = sorted(candidates, key=lambda c: c["selection_priority"])
    for c in ordered:
        if c["strategy_id"] != sid and c["availability"] in ("available", "partial"):
            return c["strategy_id"]
    return None


def write_plan(plan: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rerun_plan.json").write_text(json.dumps(plan, indent=2))
    lines = ["# Rerun Plan", ""]
    for k, v in plan.items():
        lines.append(f"- **{k}**: {v}")
    (out_dir / "rerun_plan.md").write_text("\n".join(lines) + "\n")
