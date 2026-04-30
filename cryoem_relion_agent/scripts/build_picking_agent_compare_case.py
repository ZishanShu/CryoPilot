from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def build_rule_result(summary: Dict[str, Any]) -> Dict[str, Any]:
    dataset_id = summary.get("dataset_id", "unknown_dataset")
    backend = summary.get("backend", "unknown_backend")
    stage = summary.get("stage", "particle_picking")
    runtime_status = summary.get("runtime_status", "unknown")

    f1 = as_float(summary.get("micro_f1"))
    precision = as_float(summary.get("micro_precision"))
    recall = as_float(summary.get("micro_recall"))
    threshold = as_float(summary.get("qc_threshold_f1", 0.2), 0.2)

    runtime_ok = runtime_status in {"success", "completed", "ok"}
    qc_pass = runtime_ok and f1 >= threshold

    if not runtime_ok:
        final_status = "runtime_failed"
        next_action = "fallback_to_alternative_backend_or_fix_input_format"
        reason = f"{backend} did not complete successfully."
    elif qc_pass:
        final_status = "qc_passed"
        next_action = "accept_picks_and_continue_downstream"
        reason = f"F1={f1:.4f} passed threshold={threshold:.4f}."
    else:
        final_status = "qc_failed"
        next_action = "fallback_to_alternative_particle_picker_and_repeat_qc"
        reason = f"F1={f1:.4f} is below threshold={threshold:.4f}."

    return {
        "protocol": "rule_based_agent",
        "dataset": dataset_id,
        "stage": stage,
        "selected_backend": backend,
        "input_state": {
            "dataset_id": dataset_id,
            "stage": stage,
            "n_micrographs": summary.get("n_micrographs"),
            "has_ground_truth": summary.get("total_gt_particles") is not None,
            "backend_runtime": {
                backend: runtime_status
            },
        },
        "decision_trace": [
            {
                "step": "backend_selection",
                "selected_backend": backend,
                "reason": f"Rule policy evaluates {backend} for {stage}."
            },
            {
                "step": "runtime_check",
                "backend": backend,
                "runtime_status": runtime_status,
                "runtime_ok": runtime_ok,
            },
            {
                "step": "scientific_qc",
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "qc_threshold_f1": threshold,
                "qc_pass": qc_pass,
            },
            {
                "step": "next_action",
                "recommended_next_action": next_action,
                "reason": reason,
            },
        ],
        "metrics": summary,
        "final_status": final_status,
        "recommended_next_action": next_action,
    }


def write_rule_markdown(rule: Dict[str, Any], out_md: Path) -> None:
    metrics = rule.get("metrics", {})
    lines = []
    lines.append(f"# Rule-based Agent Result: {rule.get('dataset')}\n\n")
    lines.append("## Backend decision\n\n")
    lines.append(f"- Stage: {rule.get('stage')}\n")
    lines.append(f"- Selected backend: {rule.get('selected_backend')}\n")
    lines.append(f"- Final status: {rule.get('final_status')}\n")
    lines.append(f"- Recommended next action: {rule.get('recommended_next_action')}\n\n")

    lines.append("## Runtime and QC\n\n")
    lines.append(f"- Runtime status: {metrics.get('runtime_status')}\n")
    lines.append(f"- Micrographs: {metrics.get('n_micrographs')}\n")
    lines.append(f"- Total GT particles: {metrics.get('total_gt_particles')}\n")
    lines.append(f"- Total predicted particles: {metrics.get('total_predicted_particles')}\n")
    lines.append(f"- TP / FP / FN: {metrics.get('tp')} / {metrics.get('fp')} / {metrics.get('fn')}\n")
    lines.append(f"- Precision: {as_float(metrics.get('micro_precision')):.4f}\n")
    lines.append(f"- Recall: {as_float(metrics.get('micro_recall')):.4f}\n")
    lines.append(f"- F1: {as_float(metrics.get('micro_f1')):.4f}\n")

    out_md.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-summary", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    args = parser.parse_args()

    summary = read_json(args.eval_summary)
    rule = build_rule_result(summary)

    out_root = args.out_root
    rule_dir = out_root / "rule_based"
    rule_dir.mkdir(parents=True, exist_ok=True)

    out_json = rule_dir / "rule_based_result.json"
    out_md = rule_dir / "rule_based_result.md"

    out_json.write_text(json.dumps(rule, indent=2), encoding="utf-8")
    write_rule_markdown(rule, out_md)

    print(json.dumps({
        "status": "ok",
        "dataset": rule.get("dataset"),
        "backend": rule.get("selected_backend"),
        "out_json": str(out_json),
        "out_md": str(out_md),
    }, indent=2))


if __name__ == "__main__":
    main()
