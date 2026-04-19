"""JSON-friendly payload builders for future LLM supervision."""

from __future__ import annotations

from typing import Any, Dict, Optional


def build_preflight_supervisor_input(
    *,
    selected_backend: str,
    selection_reason: Optional[str],
    preflight_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    pf = preflight_result or {}
    return {
        "selected_backend": selected_backend,
        "selection_reason": selection_reason,
        "preflight_result": pf,
        "blocked_reason": pf.get("blocked_reason"),
        "required_inputs_status": {
            "missing_inputs": list(pf.get("missing_inputs") or []),
            "available": bool(pf.get("available")),
        },
        "dependency_status": {
            "missing_dependencies": list(pf.get("missing_dependencies") or []),
            "execution_mode": pf.get("execution_mode"),
        },
    }


def build_execution_supervisor_input(result: Dict[str, Any]) -> Dict[str, Any]:
    stdout = str(result.get("stdout") or "")
    stderr = str(result.get("stderr") or "")
    out = {
        "command": result.get("command"),
        "return_code": result.get("return_code"),
        "stdout_stderr_summary": {
            "stdout_head": stdout[:500],
            "stderr_head": stderr[:500],
        },
        "output_path": result.get("topaz_output_path") or result.get("output_dir"),
        "artifact_summary": {
            "artifacts": list(result.get("artifacts") or []),
            "generated_paths": list(result.get("generated_paths") or []),
        },
        "benchmark_result_status": result.get("status"),
    }
    return out


def build_evaluation_supervisor_input(
    *,
    evaluation: Dict[str, Any],
    dataset_id: str,
    particle_diameter_angstrom: Optional[float],
    topaz_threshold: Optional[float],
) -> Dict[str, Any]:
    return {
        "predicted_count": evaluation.get("predicted_count"),
        "gt_count": evaluation.get("gt_count"),
        "matched_count": evaluation.get("matched_count"),
        "precision": evaluation.get("precision"),
        "recall": evaluation.get("recall"),
        "f1": evaluation.get("f1"),
        "dataset_id": dataset_id,
        "particle_diameter_angstrom": particle_diameter_angstrom,
        "topaz_threshold": topaz_threshold,
    }


def build_diff_regression_supervisor_input(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "regression_summary": payload.get("regression_summary") or {},
        "case_diffs": (payload.get("benchmark_diff") or {}).get("case_diffs") or [],
        "selection_diff": (payload.get("benchmark_diff") or {}).get("selection_diffs") or {},
        "current_run_id": (payload.get("run_metadata") or {}).get("run_id"),
        "previous_run_id": (payload.get("previous_run_metadata") or {}).get("previous_run_id"),
    }

