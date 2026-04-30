"""Retrieve candidate fallback strategies from evidence and backend availability."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from .evidence_loader import EvidenceCase


def _high_risk(stage_assess: Dict[str, Any]) -> bool:
    if not stage_assess:
        return False
    if stage_assess.get("risk_level") == "high":
        return True
    if stage_assess.get("accept_for_downstream") is False:
        return True
    return False


def _topaz_pred_dir_available(case: EvidenceCase) -> bool:
    sm = case.workflow_summary_raw.get("selected_micrographs") or []
    for m in sm:
        coord = m.get("topaz_coord")
        if coord and Path(coord).exists():
            return True
    return False


def _have_executable(name: str) -> bool:
    return shutil.which(name) is not None


def _template_picker_script_exists() -> bool:
    here = Path(__file__).resolve().parents[2]
    candidates = [
        here / "scripts" / "run_template_picker.py",
        here / "agent" / "template_picker.py",
        here / "run_autopick",
    ]
    return any(p.exists() for p in candidates)


def retrieve_candidate_strategies(
    case: EvidenceCase,
    backend_availability: Optional[Dict[str, Any]] = None,
    extra_topaz_pred_dirs: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Return list of candidate strategies as dicts."""
    backend_availability = backend_availability or {}
    extra_topaz_pred_dirs = extra_topaz_pred_dirs or {}

    pick_assess = case.llm_assessments_by_stage.get("particle_picking", {})
    recon_assess = case.llm_assessments_by_stage.get("reconstruction", {})
    map2pdb_assess = case.llm_assessments_by_stage.get("map2pdb", {})

    overall_accept_downstream = case.llm_overall.get(
        "accept_for_scientific_claim", case.llm_overall.get("accept_for_downstream")
    )

    candidates: List[Dict[str, Any]] = []

    # A. Topaz threshold sweep.
    topaz_evidence = _topaz_pred_dir_available(case) or bool(extra_topaz_pred_dirs)
    topaz_exec = backend_availability.get("topaz", _have_executable("topaz"))
    topaz_avail = "available" if topaz_evidence else ("partial" if topaz_exec else "unavailable")
    candidates.append({
        "strategy_id": "topaz_threshold_sweep",
        "stage": "particle_picking",
        "backend": "topaz",
        "mode": "real" if topaz_evidence else ("real" if topaz_exec else "plan-only"),
        "availability": topaz_avail,
        "reason": (
            "LLM flagged particle_picking as high risk; sweep Topaz thresholds to find "
            "config with better downstream particle quality."
            if _high_risk(pick_assess) else
            "Available pre-computed Topaz predictions enable threshold sweep evidence reuse."
        ),
        "required_inputs": [
            "pairs_csv or selected_micrographs", "topaz_pred_dirs (per threshold)",
            "relion_bin (optional, for extraction rerun)",
        ],
        "expected_outputs": [
            "reruns/topaz_threshold_<value>/workflow_summary.json",
            "comparison/rerun_comparison.csv",
        ],
        "risk": "low",
        "estimated_cost_level": "low" if topaz_evidence else "medium",
        "selection_priority": 10 if _high_risk(pick_assess) and topaz_evidence else 30,
    })

    # B. Template picker fallback.
    tpl_avail_flag = backend_availability.get("template_picker")
    if tpl_avail_flag is None:
        tpl_avail_flag = _template_picker_script_exists()
    candidates.append({
        "strategy_id": "template_picker_fallback",
        "stage": "particle_picking",
        "backend": "template_picker",
        "mode": "real" if tpl_avail_flag else "plan-only",
        "availability": "available" if tpl_avail_flag else "unavailable",
        "reason": "Fallback picker if Topaz threshold sweep does not improve picks.",
        "required_inputs": ["pairs_csv", "template_stack"],
        "expected_outputs": ["template_picker_coords/"],
        "risk": "medium",
        "estimated_cost_level": "medium",
        "selection_priority": 50 if _high_risk(pick_assess) else 80,
    })

    # C. RELION LoG / autopick fallback.
    relion_bin_env = backend_availability.get("relion_bin", os.environ.get("RELION_BIN"))
    relion_present = bool(relion_bin_env) or _have_executable("relion_autopick")
    candidates.append({
        "strategy_id": "relion_log_autopick_fallback",
        "stage": "particle_picking",
        "backend": "relion_autopick_log",
        "mode": "plan-only",
        "availability": "partial" if relion_present else "unavailable",
        "reason": "Plan-only RELION LoG/autopick fallback; needs CTF + format confirmation.",
        "required_inputs": ["micrographs.star with CTF", "diameter range"],
        "expected_outputs": ["relion_autopick_job/"],
        "risk": "medium",
        "estimated_cost_level": "medium",
        "selection_priority": 70,
    })

    # D. cryoDRGN-AI heterogeneous reconstruction.
    cryodrgn_avail = backend_availability.get(
        "cryodrgn", _have_executable("cryodrgn") or _have_executable("cryodrgn_ai")
    )
    candidates.append({
        "strategy_id": "cryodrgn_ai_reconstruction",
        "stage": "reconstruction",
        "backend": "cryodrgn_ai",
        "mode": "plan-only",
        "availability": "partial" if cryodrgn_avail else "unavailable",
        "reason": "Considered when reconstruction is high risk and heterogeneity is suspected.",
        "required_inputs": ["clean particle stack", "ctf metadata"],
        "expected_outputs": ["cryodrgn_run/"],
        "risk": "high",
        "estimated_cost_level": "high",
        "selection_priority": 60 if _high_risk(recon_assess) else 90,
    })

    # E. Real map2pdb rerun. Blocked unless reconstruction accepted for downstream.
    recon_accepted = (recon_assess.get("accept_for_downstream") is True)
    map2pdb_blocked = not recon_accepted or overall_accept_downstream is False
    candidates.append({
        "strategy_id": "map2pdb_real_run",
        "stage": "map2pdb",
        "backend": "map2pdb",
        "mode": "plan-only" if map2pdb_blocked else "real",
        "availability": "unavailable" if map2pdb_blocked else "available",
        "reason": (
            "Blocked: reconstruction not scientifically accepted for downstream model building."
            if map2pdb_blocked else
            "Reconstruction accepted; real map2pdb execution is allowed."
        ),
        "required_inputs": ["validated reconstruction map", "sequence fasta"],
        "expected_outputs": ["map2pdb/output_model.pdb"],
        "risk": "high",
        "estimated_cost_level": "high",
        "selection_priority": 99,
    })

    candidates.sort(key=lambda c: c["selection_priority"])
    return candidates


def write_candidates(candidates: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "candidate_strategies.json").write_text(json.dumps(candidates, indent=2))
    lines = ["# Candidate Strategies", ""]
    lines.append("| priority | strategy_id | stage | backend | availability | mode | reason |")
    lines.append("|---|---|---|---|---|---|---|")
    for c in candidates:
        lines.append(
            f"| {c['selection_priority']} | {c['strategy_id']} | {c['stage']} | "
            f"{c['backend']} | {c['availability']} | {c['mode']} | {c['reason']} |"
        )
    (out_dir / "candidate_strategies.md").write_text("\n".join(lines) + "\n")
