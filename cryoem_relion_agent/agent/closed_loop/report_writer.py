"""Write the closed-loop final markdown report."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .comparator import COMPARE_COLUMNS
from .evidence_loader import EvidenceCase


def write_closed_loop_report(
    out_root: Path,
    baseline: EvidenceCase,
    candidates: List[Dict[str, Any]],
    plan: Dict[str, Any],
    rows: List[Dict[str, Any]],
    final: Dict[str, Any],
) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    report = out_root / "closed_loop_report.md"

    lines: List[str] = []
    lines.append("# CryoAgent Closed-Loop Retrieval/Rerun/Selection Report")
    lines.append("")
    lines.append("## Baseline case")
    lines.append(f"- **dataset_id**: {baseline.dataset_id}")
    lines.append(f"- **baseline root**: {baseline.run_root}")
    lines.append(f"- **case_id**: {baseline.case_id}")
    lines.append("")
    lines.append("### Baseline metrics")
    lines.append("| key | value |")
    lines.append("|---|---|")
    for k, v in (baseline.metrics or {}).items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("### Rule-based status")
    lines.append("| stage | status |")
    lines.append("|---|---|")
    for k, v in (baseline.stage_status or {}).items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("### LLM stop reason")
    overall = baseline.llm_overall or {}
    lines.append(f"- accept_backend_execution: {overall.get('accept_backend_execution')}")
    lines.append(
        f"- accept_for_scientific_claim: "
        f"{overall.get('accept_for_scientific_claim', overall.get('accept_for_downstream'))}"
    )
    lines.append(f"- risk_level: {overall.get('risk_level')}")
    if overall.get("summary"):
        lines.append(f"- summary: {overall.get('summary')}")
    lines.append("")

    lines.append("## Retrieved candidate strategies")
    lines.append("| priority | strategy_id | stage | backend | availability | mode | reason |")
    lines.append("|---|---|---|---|---|---|---|")
    for c in candidates:
        lines.append(
            f"| {c['selection_priority']} | {c['strategy_id']} | {c['stage']} | "
            f"{c['backend']} | {c['availability']} | {c['mode']} | {c['reason']} |"
        )
    lines.append("")

    lines.append("## Selected rerun plan")
    for k, v in plan.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    lines.append("## Rerun results")
    lines.append("| " + " | ".join(COMPARE_COLUMNS) + " |")
    lines.append("|" + "|".join(["---"] * len(COMPARE_COLUMNS)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k in COMPARE_COLUMNS) + " |")
    if len(rows) <= 1:
        lines.append("")
        lines.append("_No executed rerun candidate. Only baseline was evaluated._")
    lines.append("")

    lines.append("## Final selection")
    for k, v in final.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    lines.append("## Interpretation")
    lines.append(
        "Runtime backend success and scientific acceptance are reported separately. "
        "A rerun may improve technical metrics without satisfying the scientific QC gate; "
        "real map2pdb execution remains blocked while the LLM stop-gate rejects downstream use. "
        "This report is an audit trail for the closed-loop controller, not a claim of "
        "completed structure determination."
    )
    lines.append("")
    report.write_text("\n".join(lines))
    return report
