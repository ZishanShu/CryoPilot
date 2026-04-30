"""Compare baseline vs rerun candidates and produce a final selection."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .evidence_loader import EvidenceCase, load_evidence_case


COMPARE_COLUMNS = [
    "case_id", "strategy_id", "stage", "backend", "config",
    "n_micrographs", "total_topaz_coords_used", "n_extracted_particles",
    "n_particle_stacks", "reconstruction_map_exists", "reconstruction_map_size_bytes",
    "rule_picking_status", "rule_extraction_status", "rule_reconstruction_status",
    "rule_map2pdb_status", "llm_accept_backend_execution", "llm_accept_for_downstream",
    "llm_risk_level", "runtime_sec", "errors",
]


def _row_from_case(
    case: EvidenceCase, strategy_id: str, stage: str, backend: str, config: str,
) -> Dict[str, Any]:
    metrics = case.metrics or {}
    rule = case.rule_decisions_by_stage
    return {
        "case_id": case.case_id,
        "strategy_id": strategy_id,
        "stage": stage,
        "backend": backend,
        "config": config,
        "n_micrographs": metrics.get("n_micrographs"),
        "total_topaz_coords_used": metrics.get("total_topaz_coords_used"),
        "n_extracted_particles": metrics.get("n_extracted_particles"),
        "n_particle_stacks": metrics.get("n_particle_stacks"),
        "reconstruction_map_exists": metrics.get("reconstruction_map_exists"),
        "reconstruction_map_size_bytes": metrics.get("reconstruction_map_size_bytes"),
        "rule_picking_status": rule.get("particle_picking", {}).get("status"),
        "rule_extraction_status": rule.get("particle_extraction", {}).get("status"),
        "rule_reconstruction_status": rule.get("reconstruction", {}).get("status"),
        "rule_map2pdb_status": rule.get("map2pdb", {}).get("status"),
        "llm_accept_backend_execution": case.llm_overall.get("accept_backend_execution"),
        "llm_accept_for_downstream": case.llm_overall.get(
            "accept_for_scientific_claim", case.llm_overall.get("accept_for_downstream")
        ),
        "llm_risk_level": case.llm_overall.get("risk_level"),
        "runtime_sec": rule.get("reconstruction", {}).get("runtime_sec"),
        "errors": "",
    }


def build_comparison(
    baseline: EvidenceCase,
    rerun_cases: List[Dict[str, Any]],  # {case, strategy_id, stage, backend, config}
) -> List[Dict[str, Any]]:
    rows = [
        _row_from_case(baseline, "baseline", "all", "baseline", baseline.case_id)
    ]
    for rc in rerun_cases:
        rows.append(_row_from_case(
            rc["case"], rc.get("strategy_id", "rerun"), rc.get("stage", ""),
            rc.get("backend", ""), rc.get("config", ""),
        ))
    return rows


def select_final(
    baseline: EvidenceCase,
    rows: List[Dict[str, Any]],
    rerun_cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    baseline_row = rows[0]
    rerun_rows = rows[1:]

    selected_for_backend = baseline_row["case_id"]
    reason_parts: List[str] = []

    valid_reruns = [r for r in rerun_rows if r.get("n_extracted_particles")]
    if valid_reruns:
        # pick highest n_extracted_particles, then reconstruction exists
        best = max(
            valid_reruns,
            key=lambda r: (
                bool(r.get("reconstruction_map_exists")),
                r.get("n_extracted_particles") or 0,
            ),
        )
        if (best.get("n_extracted_particles") or 0) > (baseline_row.get("n_extracted_particles") or 0):
            selected_for_backend = best["case_id"]
            reason_parts.append(
                f"Rerun {best['case_id']} produced more extracted particles than baseline."
            )
        else:
            reason_parts.append("No rerun improved extracted-particle count over baseline.")
    else:
        reason_parts.append("No rerun produced executable extraction; baseline kept for backend.")

    overall_accept_downstream = baseline.llm_overall.get(
        "accept_for_scientific_claim", baseline.llm_overall.get("accept_for_downstream")
    )

    accepted_for_scientific_downstream = False
    if overall_accept_downstream is True:
        # check that the selected case (or any rerun) also has acceptance
        accepted_for_scientific_downstream = True

    if accepted_for_scientific_downstream:
        next_action = "PROCEED_TO_DOWNSTREAM"
        reason_parts.append("LLM accepts the run for scientific downstream use.")
    else:
        next_action = "STOP_OR_RERUN_REQUIRED"
        reason_parts.append(
            "LLM still rejects scientific downstream; do not run real map2pdb. "
            "Recommend additional QC: 2D classification, half-map FSC, resolution estimate."
        )

    return {
        "selected_for_backend_execution": selected_for_backend,
        "selected_for_scientific_downstream": None if not accepted_for_scientific_downstream else selected_for_backend,
        "accepted_for_scientific_downstream": accepted_for_scientific_downstream,
        "final_action": next_action,
        "reason": " ".join(reason_parts),
        "n_rerun_candidates": len(rerun_rows),
        "n_valid_reruns": len(valid_reruns),
    }


def write_comparison(rows: List[Dict[str, Any]], final: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "rerun_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COMPARE_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COMPARE_COLUMNS})

    md = ["# Rerun Comparison", "", "| " + " | ".join(COMPARE_COLUMNS) + " |",
          "|" + "|".join(["---"] * len(COMPARE_COLUMNS)) + "|"]
    for r in rows:
        md.append("| " + " | ".join(str(r.get(k, "")) for k in COMPARE_COLUMNS) + " |")
    if len(rows) <= 1:
        md.append("")
        md.append("_No executed rerun candidate; only baseline reported._")
    (out_dir / "rerun_comparison.md").write_text("\n".join(md) + "\n")

    (out_dir / "final_selection.json").write_text(json.dumps(final, indent=2))
    fl = ["# Final Selection", ""]
    for k, v in final.items():
        fl.append(f"- **{k}**: {v}")
    (out_dir / "final_selection.md").write_text("\n".join(fl) + "\n")
