"""Success audit helper that mirrors ``failure_audit`` for completed cases.

A *success* audit captures:
- run identity
- which protocol + case produced it
- which backend was selected and why
- what evidence supports success (smoke_run_checks, picking_evaluation, etc.)
- reproducibility hints (command, git commit)

It deliberately reuses the same top-level shape as the failure audit so a
single downstream LLM/consumer can handle both.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

SUCCESS_AUDIT_VERSION = "1.0.0"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status_is_success(status: Optional[str]) -> bool:
    return str(status or "").strip().lower() in {"completed", "success"}


def classify_success(
    case_result: Mapping[str, Any],
) -> Dict[str, Any]:
    """Classify the evidence pattern supporting success."""
    extra = dict(case_result.get("extra") or {})
    checks = dict(extra.get("smoke_run_checks") or {})
    evaluation = dict(extra.get("picking_evaluation") or {})

    success_kind = "execution_completed"
    if checks.get("smoke_run_success") is True:
        success_kind = "smoke_run_success"
    elif checks.get("execution_ok") is True and checks.get("non_empty_output") is True:
        success_kind = "execution_completed_with_output"
    elif _status_is_success(case_result.get("status")):
        success_kind = "execution_completed"

    artifacts_raw = case_result.get("artifacts") or case_result.get("generated_paths") or []
    has_artifacts = (
        isinstance(artifacts_raw, (list, tuple))
        and len(artifacts_raw) > 0
    )

    evidence_strength = "medium"
    if (
        checks.get("preflight_ok")
        and checks.get("execution_ok")
        and checks.get("non_empty_output")
        and bool(evaluation.get("evaluation_available"))
    ):
        evidence_strength = "high"
    elif not checks:
        # Fall back to artifact-based evidence: when smoke_run_checks were
        # never populated (model-building backends do not emit them today),
        # a real backend command that produced on-disk artifacts is at least
        # medium-strength evidence of success.  Otherwise low.
        if has_artifacts and _status_is_success(case_result.get("status")):
            evidence_strength = "medium"
        else:
            evidence_strength = "low"

    return {
        "success_kind": success_kind,
        "evidence_strength": evidence_strength,
        "evaluation_available": bool(evaluation.get("evaluation_available")),
        "precision": evaluation.get("precision"),
        "recall": evaluation.get("recall"),
        "f1": evaluation.get("f1"),
    }


def _build_key_signals(
    case_result: Mapping[str, Any],
    checks: Mapping[str, Any],
) -> List[str]:
    signals: List[str] = []
    if checks.get("preflight_ok"):
        signals.append("preflight ok")
    if checks.get("execution_ok"):
        signals.append("execution ok")
    if checks.get("non_empty_output"):
        signals.append("non-empty output")
    if bool(case_result.get("command")) and str(case_result.get("execution_mode")) == "real":
        signals.append("real backend command executed")
    if checks.get("smoke_run_success") is True:
        signals.append("smoke run success flag set")
    artifacts = case_result.get("artifacts") or case_result.get("generated_paths") or []
    if isinstance(artifacts, (list, tuple)) and len(artifacts) > 0:
        signals.append(f"{len(artifacts)} artifact(s) on disk")
    return signals


def build_success_audit(
    latest_payload: Mapping[str, Any],
    smoke_summary: Mapping[str, Any],
    *,
    protocol: Optional[Mapping[str, Any]] = None,
    case: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble a full success audit dict (structure mirrors failure_audit)."""
    case_result = dict(smoke_summary.get("case_result") or {})
    extra = dict(case_result.get("extra") or {})
    checks = dict(extra.get("smoke_run_checks") or {})
    evaluation = dict(extra.get("picking_evaluation") or {})
    run_meta = dict(latest_payload.get("run_metadata") or {})
    prev_meta = dict(latest_payload.get("previous_run_metadata") or {})

    classification = classify_success(case_result)

    run_identity = {
        "run_id": run_meta.get("run_id"),
        "benchmark_id": case_result.get("benchmark_id"),
        "backend": case_result.get("backend"),
        "category": case_result.get("category"),
        "status": case_result.get("status"),
        "execution_mode": case_result.get("execution_mode"),
        "dataset_id": extra.get("dataset_id"),
        "previous_run_id": prev_meta.get("previous_run_id"),
        "protocol_name": (protocol or {}).get("protocol_name"),
        "case_id": (case or {}).get("case_id"),
    }

    execution_context = {
        "selected_backend": case_result.get("backend"),
        "command": case_result.get("command"),
        "return_code": case_result.get("return_code"),
        "runtime_seconds": case_result.get("runtime_seconds"),
        "selection_decision": case_result.get("selection_decision"),
    }

    evidence = {
        "artifact_paths": case_result.get("artifacts") or case_result.get("generated_paths"),
        "smoke_run_checks": checks,
        "picking_evaluation": evaluation,
        "key_signals": _build_key_signals(case_result, checks),
        "regression_summary": latest_payload.get("regression_summary"),
    }

    audit_metadata = {
        "audit_version": SUCCESS_AUDIT_VERSION,
        "audit_kind": "success",
        "generated_at": _utc_now_iso(),
        "source_latest_json": None,
        "source_smoke_summary": None,
    }

    return {
        "audit_metadata": audit_metadata,
        "run_identity": run_identity,
        "execution_context": execution_context,
        "success_classification": classification,
        "evidence": evidence,
    }


def attach_audit_sources(
    audit: Dict[str, Any],
    latest_path: Optional[str] = None,
    smoke_summary_path: Optional[str] = None,
) -> Dict[str, Any]:
    meta = dict(audit.get("audit_metadata") or {})
    meta["source_latest_json"] = latest_path
    meta["source_smoke_summary"] = smoke_summary_path
    audit["audit_metadata"] = meta
    return audit


def render_success_audit_markdown(audit: Mapping[str, Any]) -> str:
    ri = audit.get("run_identity") or {}
    sc = audit.get("success_classification") or {}
    ev = audit.get("evidence") or {}
    ex = audit.get("execution_context") or {}
    chk = ev.get("smoke_run_checks") or {}
    pev = ev.get("picking_evaluation") or {}

    parts: List[str] = [
        "# Smoke Run Success Audit",
        "",
        "## Run Identity",
        "",
        f"- **run_id**: {ri.get('run_id')}",
        f"- **benchmark_id**: {ri.get('benchmark_id')}",
        f"- **backend**: {ri.get('backend')}",
        f"- **status**: {ri.get('status')}",
        f"- **protocol_name**: {ri.get('protocol_name')}",
        f"- **case_id**: {ri.get('case_id')}",
        "",
        "## Success Classification",
        "",
        f"- **success_kind**: {sc.get('success_kind')}",
        f"- **evidence_strength**: {sc.get('evidence_strength')}",
        f"- **evaluation_available**: {sc.get('evaluation_available')}",
        "",
        "## Execution Context",
        "",
        f"- **command**: `{ex.get('command') or ''}`",
        f"- **return_code**: {ex.get('return_code')}",
        f"- **runtime_seconds**: {ex.get('runtime_seconds')}",
        "",
        "## Evidence",
        "",
        "### smoke_run_checks",
        "",
        "```json",
        json.dumps(chk, ensure_ascii=False, indent=2),
        "```",
        "",
        "### picking_evaluation",
        "",
        "```json",
        json.dumps(pev, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    return "\n".join(parts)


def audit_dict_to_json_bytes(audit: Mapping[str, Any]) -> bytes:
    return json.dumps(audit, ensure_ascii=False, indent=2).encode("utf-8")
