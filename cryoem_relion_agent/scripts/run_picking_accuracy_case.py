#!/usr/bin/env python3
"""Run a ``picking_accuracy`` case end-to-end.

Reads a case YAML (e.g. ``configs/cases/P1_topaz_empiar10017_picking_accuracy.yaml``),
invokes Topaz on the configured evaluation split, converts Topaz output into
``predicted_coordinates.tsv``, runs the picking-accuracy evaluator at the
primary radius and over the radius-sensitivity sweep, and writes
``benchmark_result.json`` / ``benchmark_result.md`` plus
``picking_accuracy_by_radius.csv`` and ``picking_accuracy_summary.json``.

Outputs are always written, even if Topaz fails -- in that case the result is
"failure-aware": ``execution_type='real'``, ``return_code != 0``, and
``agent_decision='block_execution_failed'`` with no fabricated metrics.

Reuses :mod:`relion.topaz_runner` and :mod:`agent.picking_accuracy` rather
than introducing a parallel execution stack.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml

from agent.picking_accuracy import (
    evaluate_picking_accuracy,
    evaluate_picking_accuracy_radius_sensitivity,
    write_radius_sensitivity_outputs,
)
from agent.picking_preflight import preflight_topaz_pick
from relion.topaz_runner import run_topaz_pick
from scripts.convert_topaz_output_to_pred_tsv import convert_topaz_output_to_pred_tsv


# --------------------------------------------------------------------------- #
# Case loading
# --------------------------------------------------------------------------- #


def load_picking_accuracy_case(case_path: Path) -> Dict[str, Any]:
    p = Path(case_path)
    if not p.is_file():
        raise FileNotFoundError(f"case file not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"case file must be a mapping: {p}")
    case_type = str(data.get("case_type") or "").strip()
    if case_type != "picking_accuracy":
        raise ValueError(
            f"case_type must be 'picking_accuracy' for this runner, got {case_type!r}"
        )
    for key in ("input", "evaluation", "dataset"):
        if key not in data:
            raise ValueError(f"case missing required key: {key}")
    return data


# --------------------------------------------------------------------------- #
# Agent decision
# --------------------------------------------------------------------------- #


def _agent_decision(
    *,
    return_code: Optional[int],
    n_pred: int,
    f1: Optional[float],
) -> str:
    if return_code is None or int(return_code) != 0:
        return "block_execution_failed"
    if int(n_pred) == 0:
        return "block_empty_output"
    if f1 is None:
        return "block_empty_output"
    try:
        if float(f1) < 0.1:
            return "retry_low_f1"
    except (TypeError, ValueError):
        return "retry_low_f1"
    return "continue_to_particle_extraction"


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #


def _format_benchmark_md(result: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"# benchmark_result :: {result.get('case_id')}")
    lines.append("")
    lines.append(f"- dataset: {result.get('dataset', {}).get('name')}")
    lines.append(f"- execution_type: {result.get('execution_type')}")
    lines.append(f"- return_code: {result.get('return_code')}")
    lines.append(f"- runtime_seconds: {result.get('runtime_seconds')}")
    lines.append(f"- agent_decision: **{result.get('agent_decision')}**")
    if result.get("failure_reason"):
        lines.append(f"- failure_reason: {result.get('failure_reason')}")
    lines.append("")
    lines.append("## Topaz command")
    cmd = result.get("topaz_command")
    lines.append(f"`{cmd}`" if cmd else "(no command -- preflight blocked or skipped)")
    lines.append("")

    artifacts = result.get("artifacts") or {}
    if artifacts:
        lines.append("## Artifacts")
        for key, value in artifacts.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")

    summary = result.get("accuracy_summary") or {}
    lines.append("## Accuracy summary")
    primary = summary.get("primary_metrics")
    if primary:
        lines.append(
            f"- primary radius: {summary.get('primary_radius_px')} px"
        )
        lines.append(
            f"  - precision={primary.get('precision')}, recall={primary.get('recall')}, f1={primary.get('f1')}"
        )
        lines.append(
            f"  - n_gt={primary.get('n_gt')}, n_pred={primary.get('n_pred')}, n_matched={primary.get('n_matched')}, empty_output={primary.get('empty_output')}"
        )
    else:
        lines.append("- (no primary metrics available)")
    radii = summary.get("radii") or []
    if radii:
        lines.append("")
        lines.append("### Radius sensitivity")
        lines.append("| radius_px | n_gt | n_pred | n_matched | precision | recall | f1 | empty_output |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for row in radii:
            lines.append(
                "| {r} | {ng} | {np} | {nm} | {p} | {rec} | {f} | {e} |".format(
                    r=row.get("matching_radius_px"),
                    ng=row.get("n_gt"),
                    np=row.get("n_pred"),
                    nm=row.get("n_matched"),
                    p=row.get("precision"),
                    rec=row.get("recall"),
                    f=row.get("f1"),
                    e=row.get("empty_output"),
                )
            )
    return "\n".join(lines) + "\n"


def write_benchmark_outputs(
    result: Mapping[str, Any], output_dir: Path
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmark_result.json"
    md_path = output_dir / "benchmark_result.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_format_benchmark_md(result), encoding="utf-8")
    return {"benchmark_result_json": json_path, "benchmark_result_md": md_path}


# --------------------------------------------------------------------------- #
# Main runner
# --------------------------------------------------------------------------- #


def run_picking_accuracy_case(
    *,
    case: Mapping[str, Any],
    project_root: Path,
    output_dir: Optional[Path] = None,
    dry_run: bool = False,
    topaz_executable: Optional[str] = None,
    topaz_threshold: Optional[float] = None,
    topaz_device: Optional[str] = None,
    particle_diameter_angstrom: Optional[float] = None,
    particle_diameter_px: Optional[float] = None,
) -> Dict[str, Any]:
    case_id = str(case.get("case_id"))
    dataset = dict(case.get("dataset") or {})
    inputs = dict(case.get("input") or {})
    evaluation = dict(case.get("evaluation") or {})
    eval_split = str(evaluation.get("eval_split") or "test")

    primary_radius = float(evaluation.get("matching_radius_px") or 30)
    radii_sweep: Sequence[float] = tuple(
        float(r) for r in (evaluation.get("radius_sensitivity_px") or [primary_radius])
    )

    run_root = Path(case.get("run_root") or output_dir or (project_root / "real_runs" / case_id))
    out_dir = Path(output_dir) if output_dir is not None else (run_root / "output")
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_star_key = f"micrographs_{eval_split}_star"
    eval_star = inputs.get(eval_star_key)
    if not eval_star:
        raise ValueError(f"case input missing key: {eval_star_key}")
    gt_tsv = inputs.get("gt_coordinates")
    if not gt_tsv:
        raise ValueError("case input missing key: gt_coordinates")

    topaz_output_dir = out_dir / "topaz"
    topaz_output_dir.mkdir(parents=True, exist_ok=True)
    picks_star_path = topaz_output_dir / "picks.star"

    params: Dict[str, Any] = {
        "micrographs_star_path": str(eval_star),
        "output_dir": str(topaz_output_dir),
        "picked_particles_star": str(picks_star_path),
    }
    if particle_diameter_angstrom is not None:
        params["particle_diameter_angstrom"] = float(particle_diameter_angstrom)
    if particle_diameter_px is not None:
        params["particle_diameter_px"] = float(particle_diameter_px)
    if topaz_threshold is not None:
        params["topaz_threshold"] = float(topaz_threshold)
    if topaz_device is not None:
        params["topaz_device"] = str(topaz_device)
    if topaz_executable is not None:
        params["topaz_executable"] = str(topaz_executable)

    pre = preflight_topaz_pick(params, project_root=Path(project_root), topaz_config_path=None)

    started = time.time()
    topaz_command: Optional[str] = None
    return_code: Optional[int] = None
    failure_reason: Optional[str] = None
    pred_tsv: Optional[Path] = None
    artifacts: Dict[str, str] = {}

    if not pre.get("available"):
        execution_type = "preflight_blocked"
        failure_reason = str(pre.get("blocked_reason") or "preflight_unavailable")
        topaz_result_dict: Dict[str, Any] = {
            "status": "preflight_blocked",
            "blocked_reason": failure_reason,
        }
    else:
        execution_type = "dry_run" if dry_run else "real"
        topaz_result = run_topaz_pick(
            params=params,
            project_dir=Path(project_root),
            topaz_config_path=None,
            preflight=pre,
            dry_run=bool(dry_run),
        )
        topaz_result_dict = topaz_result.to_dict() if hasattr(topaz_result, "to_dict") else dict(asdict(topaz_result))
        topaz_command = topaz_result_dict.get("command")
        return_code = topaz_result_dict.get("return_code")
        if topaz_result_dict.get("status") == "failed":
            failure_reason = topaz_result_dict.get("error_message") or "topaz_failed"
        if topaz_result_dict.get("generated_paths"):
            artifacts["picks_star"] = str(topaz_result_dict["generated_paths"][0])
        elif picks_star_path.is_file():
            artifacts["picks_star"] = str(picks_star_path)

    pred_tsv = out_dir / "predicted_coordinates.tsv"
    pred_written = False
    if "picks_star" in artifacts and Path(artifacts["picks_star"]).is_file():
        try:
            convert_topaz_output_to_pred_tsv(Path(artifacts["picks_star"]), pred_tsv)
            artifacts["predicted_coordinates_tsv"] = str(pred_tsv)
            pred_written = True
        except Exception as exc:  # noqa: BLE001
            failure_reason = (failure_reason or "") + f"; convert_failed: {exc}"
    if not pred_written:
        # Always emit an empty predicted_coordinates.tsv so downstream tooling
        # can still read it as a real (empty) prediction file.
        pred_tsv.write_text("image_name\tx_coord\ty_coord\tscore\n", encoding="utf-8")
        artifacts["predicted_coordinates_tsv"] = str(pred_tsv)

    finished = time.time()
    runtime_seconds = round(finished - started, 6)

    primary_metrics: Optional[Dict[str, Any]] = None
    radius_rows: List[Dict[str, Any]] = []
    try:
        radius_rows = evaluate_picking_accuracy_radius_sensitivity(
            gt_tsv=Path(gt_tsv),
            pred_tsv=pred_tsv,
            split=eval_split,
            radii_px=tuple(radii_sweep),
        )
        primary_metrics = evaluate_picking_accuracy(
            gt_tsv=Path(gt_tsv),
            pred_tsv=pred_tsv,
            split=eval_split,
            matching_radius_px=primary_radius,
        )
        csv_path, summary_path = write_radius_sensitivity_outputs(
            radius_rows,
            csv_path=out_dir / "picking_accuracy_by_radius.csv",
            summary_json_path=out_dir / "picking_accuracy_summary.json",
            primary_radius_px=primary_radius,
        )
        artifacts["picking_accuracy_by_radius_csv"] = str(csv_path)
        artifacts["picking_accuracy_summary_json"] = str(summary_path)
    except FileNotFoundError as exc:
        failure_reason = (failure_reason or "") + f"; eval_input_missing: {exc}"
    except Exception as exc:  # noqa: BLE001
        failure_reason = (failure_reason or "") + f"; eval_failed: {exc}"

    n_pred = int(primary_metrics.get("n_pred")) if primary_metrics else 0
    f1 = primary_metrics.get("f1") if primary_metrics else None
    decision = _agent_decision(
        return_code=return_code,
        n_pred=n_pred,
        f1=f1,
    )

    accuracy_summary: Dict[str, Any] = {
        "primary_radius_px": primary_radius,
        "primary_metrics": primary_metrics,
        "radii": radius_rows,
        "eval_split": eval_split,
    }

    result: Dict[str, Any] = {
        "case_id": case_id,
        "dataset": dataset,
        "execution_type": execution_type,
        "topaz_command": topaz_command,
        "return_code": return_code,
        "runtime_seconds": runtime_seconds,
        "artifacts": artifacts,
        "prediction_file": str(pred_tsv),
        "gt_file": str(gt_tsv),
        "accuracy_summary": accuracy_summary,
        "failure_reason": failure_reason,
        "agent_decision": decision,
        "topaz_result": topaz_result_dict,
        "preflight": {
            "available": bool(pre.get("available")),
            "blocked_reason": pre.get("blocked_reason"),
            "missing_inputs": pre.get("missing_inputs"),
            "missing_dependencies": pre.get("missing_dependencies"),
        },
        "evaluation": {
            "type": evaluation.get("type"),
            "matching_radius_px": primary_radius,
            "radius_sensitivity_px": list(radii_sweep),
            "eval_split": eval_split,
        },
    }
    paths = write_benchmark_outputs(result, output_dir=out_dir)
    result["artifacts"].update({k: str(v) for k, v in paths.items()})
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--case",
        required=True,
        help="Path to a picking_accuracy case YAML.",
    )
    p.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root used as cwd for Topaz subprocess.",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--topaz-executable", default=None)
    p.add_argument("--topaz-threshold", type=float, default=None)
    p.add_argument("--topaz-device", default=None)
    p.add_argument("--particle-diameter-angstrom", type=float, default=None)
    p.add_argument("--particle-diameter-px", type=float, default=None)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    case = load_picking_accuracy_case(Path(args.case))
    result = run_picking_accuracy_case(
        case=case,
        project_root=Path(args.project_root).resolve(),
        output_dir=args.output_dir,
        dry_run=bool(args.dry_run),
        topaz_executable=args.topaz_executable,
        topaz_threshold=args.topaz_threshold,
        topaz_device=args.topaz_device,
        particle_diameter_angstrom=args.particle_diameter_angstrom,
        particle_diameter_px=args.particle_diameter_px,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    decision = str(result.get("agent_decision") or "")
    if decision == "continue_to_particle_extraction":
        return 0
    if decision in {"block_execution_failed", "block_empty_output"}:
        return 1
    return 2  # retry_low_f1 or other


if __name__ == "__main__":
    raise SystemExit(main())
