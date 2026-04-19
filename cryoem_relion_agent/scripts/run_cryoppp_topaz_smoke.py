#!/usr/bin/env python3
"""Run one real Topaz smoke benchmark on a CryoPPP subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from agent.backend_registry import get_backend_capability_matrix
from agent.benchmark_harness import BenchmarkHarness
from agent.benchmark_persistence import load_latest_benchmark, persist_benchmark_run
from agent.data_prep.cryoppp_smoke_prep import prepare_cryoppp_topaz_smoke
from agent.eval.picking_eval import evaluate_picking_predictions
from agent.llm_supervision_payloads import (
    build_diff_regression_supervisor_input,
    build_evaluation_supervisor_input,
    build_execution_supervisor_input,
    build_preflight_supervisor_input,
)
from app.scheduler import Scheduler


def _build_case(
    manifest: Dict[str, Any],
    *,
    particle_diameter_angstrom: float,
    topaz_executable: str | None,
    topaz_model_path: str | None,
    topaz_threshold: float | None,
    topaz_device: str | None,
) -> Dict[str, Any]:
    inputs = {
        "picking_backend": "topaz",
        "micrographs_star_path": manifest["output_paths"]["micrographs_star_path"],
        "particle_diameter_angstrom": particle_diameter_angstrom,
        "output_dir": manifest["output_paths"]["topaz_output_dir"],
    }
    if topaz_executable:
        inputs["topaz_executable"] = topaz_executable
    if topaz_model_path:
        inputs["topaz_model_path"] = topaz_model_path
    if topaz_threshold is not None:
        inputs["topaz_threshold"] = topaz_threshold
    if topaz_device:
        inputs["topaz_device"] = topaz_device
    return {
        "benchmark_id": f"cryoppp_topaz_smoke_{manifest['dataset_id']}",
        "category": "picking",
        "backend": "topaz",
        "inputs": inputs,
        "expected_status": "completed",
        "notes": "First real CryoPPP subset Topaz smoke run.",
    }


def _evaluate_and_enrich(
    result: Dict[str, Any],
    *,
    manifest: Dict[str, Any],
    particle_diameter_angstrom: float,
    topaz_threshold: float | None,
    match_radius_px: float,
) -> Dict[str, Any]:
    extra = dict(result.get("extra") or {})
    topaz_output = result.get("topaz_output_path")
    gt_files = [Path(p) for p in list(manifest.get("gt_coordinate_files") or [])]
    evaluation = {
        "evaluation_available": False,
        "predicted_count": 0,
        "gt_count": 0,
        "matched_count": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }
    if isinstance(topaz_output, str) and Path(topaz_output).is_file():
        evaluation = evaluate_picking_predictions(
            predicted_coord_file=Path(topaz_output),
            gt_coord_files=gt_files,
            match_radius_px=match_radius_px,
        )
    preflight = result.get("preflight") if isinstance(result.get("preflight"), dict) else {}
    non_empty_output = bool(
        isinstance(topaz_output, str)
        and Path(topaz_output).is_file()
        and Path(topaz_output).stat().st_size > 0
    )
    smoke_success = bool(
        preflight.get("available")
        and str(result.get("status")) in {"completed", "success"}
        and non_empty_output
        and int(evaluation.get("predicted_count") or 0) > 0
        and bool(evaluation.get("evaluation_available"))
    )
    extra.update(
        {
            "dataset_id": manifest.get("dataset_id"),
            "smoke_run_checks": {
                "preflight_ok": bool(preflight.get("available")),
                "execution_ok": str(result.get("status")) in {"completed", "success"},
                "non_empty_output": non_empty_output,
                "evaluation_available": bool(evaluation.get("evaluation_available")),
                "precision": evaluation.get("precision"),
                "recall": evaluation.get("recall"),
                "f1": evaluation.get("f1"),
                "smoke_run_success": smoke_success,
            },
            "picking_evaluation": evaluation,
            "llm_supervision_payloads": {
                "preflight": build_preflight_supervisor_input(
                    selected_backend="topaz",
                    selection_reason=(result.get("selection_decision") or {}).get("selection_reason")
                    if isinstance(result.get("selection_decision"), dict)
                    else None,
                    preflight_result=preflight,
                ),
                "execution": build_execution_supervisor_input(result),
                "evaluation": build_evaluation_supervisor_input(
                    evaluation=evaluation,
                    dataset_id=str(manifest.get("dataset_id") or ""),
                    particle_diameter_angstrom=particle_diameter_angstrom,
                    topaz_threshold=topaz_threshold,
                ),
            },
        }
    )
    result["extra"] = extra
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--run-root", default="real_runs/cryoppp_topaz_smoke")
    p.add_argument("--sample-size", type=int, default=32)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--sampling-strategy", choices=["random", "first_n"], default="random")
    p.add_argument("--particle-diameter-angstrom", type=float, default=None)
    p.add_argument("--pixel-size-angstrom", type=float, default=None)
    p.add_argument("--particle-diameter-px", type=float, default=None)
    p.add_argument("--topaz-executable", default=None)
    p.add_argument("--topaz-model-path", default=None)
    p.add_argument("--topaz-threshold", type=float, default=-6.0)
    p.add_argument("--topaz-device", default="0")
    p.add_argument("--evaluation-match-radius-px", type=float, default=8.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--write-failure-audit",
        action="store_true",
        help="Write failure_audit.json/md next to smoke_run_summary.json from latest benchmark + this summary.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path(args.project_root).resolve()
    run_root = Path(args.run_root).resolve()

    manifest = prepare_cryoppp_topaz_smoke(
        dataset_root=Path(args.dataset_root),
        run_root=run_root,
        dataset_id=str(args.dataset_id),
        sample_size=int(args.sample_size),
        random_seed=int(args.random_seed),
        sampling_strategy=str(args.sampling_strategy),
        use_symlink=True,
        pixel_size_angstrom_override=args.pixel_size_angstrom,
        particle_diameter_px_override=args.particle_diameter_px,
    )
    particle_diameter = args.particle_diameter_angstrom
    if particle_diameter is None:
        particle_diameter = manifest.get("estimated_particle_diameter_angstrom")
    if particle_diameter is None:
        raise ValueError("particle_diameter_angstrom is required (explicit or derivable from manifest)")

    case = _build_case(
        manifest,
        particle_diameter_angstrom=float(particle_diameter),
        topaz_executable=args.topaz_executable,
        topaz_model_path=args.topaz_model_path,
        topaz_threshold=args.topaz_threshold,
        topaz_device=args.topaz_device,
    )

    scheduler = Scheduler(project_root=project_root)
    harness = BenchmarkHarness(executor=scheduler.action_executor, project_root=project_root)
    case_result = harness.run_case(case, dry_run=bool(args.dry_run))
    case_result = _evaluate_and_enrich(
        case_result,
        manifest=manifest,
        particle_diameter_angstrom=float(particle_diameter),
        topaz_threshold=args.topaz_threshold,
        match_radius_px=float(args.evaluation_match_radius_px),
    )
    summary = {
        "case_count": 1,
        "completed": 1 if case_result.get("status") in {"completed", "success"} else 0,
        "blocked": 1 if case_result.get("status") == "preflight_blocked" else 0,
        "failed": 1 if case_result.get("status") in {"failed", "error"} else 0,
        "results": [case_result],
        "markdown": harness.summary_markdown(
            {
                "case_count": 1,
                "completed": 1 if case_result.get("status") in {"completed", "success"} else 0,
                "blocked": 1 if case_result.get("status") == "preflight_blocked" else 0,
                "failed": 1 if case_result.get("status") in {"failed", "error"} else 0,
                "results": [case_result],
            }
        ),
    }
    persist_benchmark_run(
        project_root=project_root,
        summary=summary,
        capability_matrix=get_backend_capability_matrix(),
        dry_run=bool(args.dry_run),
    )
    latest = load_latest_benchmark(project_root) or {}
    llm_diff = build_diff_regression_supervisor_input(latest)
    result_payload = {
        "manifest_path": manifest.get("manifest_path"),
        "benchmark_case": case,
        "case_result": case_result,
        "llm_diff_regression_payload": llm_diff,
        "latest_benchmark_path": str(project_root / ".cryoem_agent" / "benchmarks" / "latest.json"),
    }
    reports_dir = Path(manifest["output_paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_json = reports_dir / "smoke_run_summary.json"
    out_json.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if bool(args.write_failure_audit):
        from agent.failure_audit import (
            attach_audit_sources,
            audit_dict_to_json_bytes,
            build_failure_audit,
            render_failure_audit_markdown,
        )

        audit = build_failure_audit(latest, result_payload)
        attach_audit_sources(
            audit,
            latest_path=str(project_root / ".cryoem_agent" / "benchmarks" / "latest.json"),
            smoke_summary_path=str(out_json),
        )
        (out_json.parent / "failure_audit.json").write_bytes(audit_dict_to_json_bytes(audit))
        (out_json.parent / "failure_audit.md").write_text(
            render_failure_audit_markdown(audit), encoding="utf-8"
        )
    print(json.dumps(result_payload, ensure_ascii=False, indent=2))
    return 0 if case_result.get("status") != "preflight_blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())

