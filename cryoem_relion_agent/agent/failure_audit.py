"""Lightweight, rule-based smoke-run failure audit for LLM-ready structured output."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple

AUDIT_VERSION = "1.0.0"

_HEAD_CHARS = 8000

# Signals that suggest image / array API mismatch at runtime (root cause guess only).
_INPUT_FORMAT_HINTS = (
    "pngimagefile",
    "jpegimagefile",
    "jpegimageplugin",
    "webpimagefile",
    "no attribute 'shape'",
    'no attribute "shape"',
    "pil.image",
    "image.open",
)

# Dependency / environment hints.
_MISSING_DEP_HINTS = (
    "modulenotfounderror",
    "no module named",
    "cannot import",
    "no such file or directory",
    "failed to load library",
    "cuda runtime",
    "out of memory",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _find_benchmark_result(latest_payload: Mapping[str, Any], benchmark_id: str) -> Optional[Dict[str, Any]]:
    for row in latest_payload.get("benchmark_results") or []:
        if row.get("benchmark_id") == benchmark_id:
            if isinstance(row, dict):
                return row
    return None


def _truncate(text: Optional[str], max_chars: int = _HEAD_CHARS) -> str:
    if not text:
        return ""
    s = str(text)
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n...[truncated]"


def _norm_blob(*parts: Optional[str]) -> str:
    return " ".join(p for p in parts if p).lower()


def _safe_int(rc: Any) -> Optional[int]:
    if rc is None:
        return None
    try:
        return int(rc)
    except (TypeError, ValueError):
        return None


def _guess_root_cause(combined_lower: str) -> Tuple[str, str]:
    """Return (root_cause_guess, confidence)."""
    if any(h in combined_lower for h in _INPUT_FORMAT_HINTS):
        return "input_format_incompatibility", "high"
    if any(h in combined_lower for h in _MISSING_DEP_HINTS):
        return "missing_model_or_dependency", "medium"
    return "unknown", "low"


def classify_failure(
    case_result: Mapping[str, Any],
    latest_payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Classify failure phase/kind and root-cause guess from a benchmark case_result."""
    benchmark_id = case_result.get("benchmark_id")
    latest_case: Optional[Dict[str, Any]] = None
    if latest_payload is not None and benchmark_id:
        latest_case = _find_benchmark_result(latest_payload, str(benchmark_id))

    preflight = dict(case_result.get("preflight") or {})
    status = str(case_result.get("status") or "").lower()
    extra = dict(case_result.get("extra") or {})
    checks = dict(extra.get("smoke_run_checks") or {})

    stderr = str(case_result.get("stderr") or "")
    stdout = str(case_result.get("stdout") or "")
    blocked = str(case_result.get("blocked_reason") or "")
    if latest_case and not blocked:
        blocked = str(latest_case.get("blocked_reason") or "")

    combined = _norm_blob(stderr, stdout, blocked, preflight.get("blocked_reason"))
    root_guess, root_conf = _guess_root_cause(combined)

    preflight_ok = bool(preflight.get("available", True)) and status != "preflight_blocked"
    if "preflight_ok" in checks:
        preflight_ok = bool(checks["preflight_ok"])

    rc = case_result.get("return_code")
    rc_i = _safe_int(rc)

    failure_phase = "unknown"
    failure_kind = "unknown"

    if status == "preflight_blocked" or not bool(preflight.get("available", True)):
        failure_phase = "preflight"
        failure_kind = "preflight_blocked"
        if root_guess == "unknown":
            root_guess, root_conf = _guess_root_cause(combined)
    elif preflight_ok and rc_i is not None and rc_i != 0:
        failure_phase = "execution"
        failure_kind = "backend_runtime_error"
    elif preflight_ok and checks.get("execution_ok") is False:
        failure_phase = "execution"
        failure_kind = "backend_runtime_error"
    elif preflight_ok and rc_i is not None and rc_i == 0 and checks.get("non_empty_output") is False:
        failure_phase = "execution"
        failure_kind = "empty_output_failure"
    elif (
        preflight_ok
        and checks.get("execution_ok") is True
        and checks.get("smoke_run_success") is False
    ):
        failure_phase = "evaluation"
        failure_kind = "evaluation_failure"
        if root_guess == "unknown":
            root_guess = "evaluation_or_metrics_unavailable"
            root_conf = "low"
    elif preflight_ok and (rc_i == 0 or rc is None) and status in {"failed", "error"}:
        failure_phase = "execution"
        failure_kind = "empty_output_failure"
    else:
        if status in {"failed", "error"} and preflight_ok:
            failure_phase = "execution"
            failure_kind = "backend_runtime_error"

    if failure_kind == "backend_runtime_error" and root_guess == "input_format_incompatibility":
        conf_out = "high" if root_conf == "high" else "medium"
    elif failure_kind == "backend_runtime_error":
        conf_out = "medium"
    else:
        conf_out = root_conf if isinstance(root_conf, str) else "low"

    retryable = False
    requires_data_change = False
    requires_code_change = False
    if failure_kind == "preflight_blocked":
        retryable = "conditionally_true"
        requires_data_change = True
    elif failure_kind == "backend_runtime_error":
        if root_guess == "input_format_incompatibility":
            retryable = "conditionally_true"
            requires_data_change = True
            requires_code_change = False
        elif root_guess == "missing_model_or_dependency":
            retryable = "conditionally_true"
            requires_data_change = True
        else:
            retryable = "conditionally_true"
            requires_data_change = False
    elif failure_kind == "empty_output_failure":
        retryable = "conditionally_true"
        requires_data_change = True
    elif failure_kind == "evaluation_failure":
        retryable = True
        requires_data_change = False

    return {
        "failure_phase": failure_phase,
        "failure_kind": failure_kind,
        "root_cause_guess": root_guess,
        "confidence": conf_out,
        "retryable": retryable,
        "requires_data_change": requires_data_change,
        "requires_code_change": requires_code_change,
    }


def _build_key_signals(
    case_result: Mapping[str, Any],
    checks: Mapping[str, Any],
    classification: Mapping[str, Any],
) -> List[str]:
    signals: List[str] = []
    preflight = case_result.get("preflight") or {}
    if checks.get("preflight_ok") or (preflight.get("available") and str(case_result.get("status")) != "preflight_blocked"):
        signals.append("preflight passed")
    rc = case_result.get("return_code")
    rc_i = _safe_int(rc)
    if rc_i is not None and rc_i != 0:
        signals.append("return_code != 0")
    if str(case_result.get("execution_mode") or "") == "real" and case_result.get("command"):
        signals.append("backend reached real execution")
    stderr = str(case_result.get("stderr") or "")
    if "traceback" in stderr.lower():
        signals.append("python traceback present in stderr")
    if classification.get("root_cause_guess") == "input_format_incompatibility":
        signals.append("image loader / array API mismatch suggested by stderr")
    if checks.get("non_empty_output") is False:
        signals.append("output missing or header-only picks.star")
    if checks.get("non_empty_output") is True and checks.get("execution_ok") is False:
        signals.append("artifact path exists but execution did not succeed")
    return signals


def _default_suggested_actions(classification: Mapping[str, Any]) -> List[Dict[str, Any]]:
    kind = classification.get("failure_kind")
    guess = classification.get("root_cause_guess")
    actions: List[Dict[str, Any]] = []

    if guess == "input_format_incompatibility":
        actions.append(
            {
                "action_id": "use_cryoppp_or_mrc_inputs",
                "priority": "high",
                "title": "改用完整 CryoPPP 流程或 MRC 微图输入",
                "rationale": "Topaz 期望类 numpy 数组；PIL Image 无 shape，说明输入格式与管线假设不一致。",
                "scope": "dataset",
                "actionable_steps": [
                    "使用原始 MRC/或其它可被 loader 转为数组的格式重新导出微图列表",
                    "在 smoke 配置中指向与 Topaz 兼容的 micrographs_star",
                ],
            }
        )
        actions.append(
            {
                "action_id": "convert_lite_to_topaz_friendly",
                "priority": "medium",
                "title": "将 Lite PNG/JPG 批处理为 Topaz 兼容张量路径",
                "rationale": "在独立校验步骤中把图像转为 Topaz 可读取格式后再跑 extract。",
                "scope": "pipeline",
                "actionable_steps": [
                    "编写离线转换步骤（不入当前阶段业务代码）并核对单张可读",
                    "对转换结果做一次最小 extract 试跑验证",
                ],
            }
        )
    if kind == "empty_output_failure" or (
        kind == "backend_runtime_error" and guess == "input_format_incompatibility"
    ):
        actions.append(
            {
                "action_id": "tighten_nonempty_output_check",
                "priority": "low",
                "title": "复核 picks.star 的 non_empty_output 判定",
                "rationale": "若仅写入空壳或 header，检查逻辑应与真实粒子行数一致。",
                "scope": "benchmark",
                "actionable_steps": [
                    "核对 STAR 是否仅有头部或零行 coordinate",
                    "在烟雾测试中增加与粒子数相关的断言",
                ],
            }
        )

    if not actions:
        actions.append(
            {
                "action_id": "inspect_logs_and_repro",
                "priority": "high",
                "title": "基于 stderr 与命令行复现失败",
                "rationale": "需要最小复现以确认根因类别。",
                "scope": "general",
                "actionable_steps": [
                    "在相同环境下手动重跑 command",
                    "收集完整 stdout/stderr 与产物路径",
                ],
            }
        )
    return actions[:3]


def _build_human_summary_cn(
    run_identity: Mapping[str, Any],
    classification: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> str:
    lines = [
        f"本次为后端 {run_identity.get('backend')} 的基准用例 "
        f"{run_identity.get('benchmark_id')}，整体状态为 {run_identity.get('status')}。",
        f"失败阶段判定为 {classification.get('failure_phase')}，"
        f"类别为 {classification.get('failure_kind')}。",
        f"结合 stderr/阻塞信息，最可能的根因猜测为 {classification.get('root_cause_guess')}（置信度 {classification.get('confidence')}）。",
    ]
    chk = evidence.get("smoke_run_checks") or {}
    if chk:
        lines.append(
            f"烟雾检查：preflight_ok={chk.get('preflight_ok')}，execution_ok={chk.get('execution_ok')}，"
            f"return_code 见 execution_context。"
        )
    br = evidence.get("blocked_reason")
    if isinstance(br, str) and br.strip():
        short = br.strip().splitlines()[-1][:200]
        lines.append(f"关键证据摘录：{short}")
    lines.append("建议优先按 structured suggested_actions 中的高优先级项处理数据与环境；若需改代码，仅在确认根因后评估。")
    text = " ".join(lines)
    # 控制篇幅：5–10 句左右
    if len(text) > 1200:
        text = text[:1190] + "…"
    return text


def _as_dict(value: Any) -> Dict[str, Any]:
    """Coerce ``value`` to a dict, returning an empty dict for anything else.

    ``failure_audit`` is called on a very wide set of case_result shapes —
    preflight-blocked topaz, successful-then-empty-output topaz, seq_free
    EModelX where supervision payloads may be absent, dry-runs that never
    populated ``smoke_run_checks``, and so on.  Using ``or {}`` is not enough
    because Python's truthiness returns the value itself when it's a non-dict
    (e.g. a list / False), and downstream ``.get()`` calls would crash.
    """
    return value if isinstance(value, dict) else {}


def build_failure_audit(
    latest_payload: Mapping[str, Any],
    smoke_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build full failure audit dict from benchmark latest.json and smoke_run_summary.json."""
    case_result = _as_dict(smoke_summary.get("case_result") if isinstance(smoke_summary, Mapping) else None)
    case_preflight = _as_dict(case_result.get("preflight"))
    benchmark_id = case_result.get("benchmark_id")
    latest_payload_map = latest_payload if isinstance(latest_payload, Mapping) else {}
    latest_case = (
        _find_benchmark_result(latest_payload_map, str(benchmark_id)) if benchmark_id else None
    )
    latest_case_dict = _as_dict(latest_case)

    classification = classify_failure(case_result, latest_payload_map)

    extra = _as_dict(case_result.get("extra"))
    payloads = _as_dict(extra.get("llm_supervision_payloads"))
    preflight_sup = _as_dict(payloads.get("preflight"))
    run_meta = _as_dict(latest_payload_map.get("run_metadata"))
    prev_meta = _as_dict(latest_payload_map.get("previous_run_metadata"))

    diff_from_smoke = smoke_summary.get("llm_diff_regression_payload") if isinstance(smoke_summary, Mapping) else None
    diff_regression: Any
    if isinstance(diff_from_smoke, dict) and diff_from_smoke:
        diff_regression = diff_from_smoke
    else:
        diff_regression = {
            "regression_summary": latest_payload_map.get("regression_summary"),
            "benchmark_diff": latest_payload_map.get("benchmark_diff"),
            "run_metadata": run_meta,
            "previous_run_metadata": prev_meta,
        }

    dataset_id = extra.get("dataset_id")
    checks = _as_dict(extra.get("smoke_run_checks"))

    prev_rid = prev_meta.get("previous_run_id")
    if not prev_rid and isinstance(diff_from_smoke, dict):
        prev_rid = diff_from_smoke.get("previous_run_id")

    run_identity = {
        "run_id": run_meta.get("run_id"),
        "benchmark_id": case_result.get("benchmark_id"),
        "backend": case_result.get("backend"),
        "category": case_result.get("category"),
        "status": case_result.get("status"),
        "execution_mode": case_result.get("execution_mode"),
        "dataset_id": dataset_id,
        "previous_run_id": prev_rid,
    }

    # pf_result is the authoritative preflight snapshot for this audit:
    # prefer the supervision payload's ``preflight_result``, then the
    # case_result's ``preflight`` block.  Both may be missing or non-dict
    # for non-picking backends (e.g. EModelX seq_free under rule_only_agent)
    # so we coerce at every step.
    pf_result = _as_dict(preflight_sup.get("preflight_result")) or case_preflight
    pf_result = _as_dict(pf_result)
    execution_payload = _as_dict(payloads.get("execution"))
    picking_evaluation = _as_dict(extra.get("picking_evaluation"))

    available_fallback = case_preflight.get("available")
    execution_context = {
        "selected_backend": preflight_sup.get("selected_backend") or case_result.get("backend"),
        "selection_reason": preflight_sup.get("selection_reason"),
        "preflight_available": bool(pf_result.get("available", available_fallback)),
        "preflight_execution_mode": pf_result.get("execution_mode") or case_preflight.get("execution_mode"),
        "command": case_result.get("command") or execution_payload.get("command"),
        "return_code": case_result.get("return_code"),
        "runtime_seconds": case_result.get("runtime_seconds"),
    }

    stderr = str(case_result.get("stderr") or "")
    stdout = str(case_result.get("stdout") or "")
    if latest_case_dict:
        if not stderr and latest_case_dict.get("blocked_reason"):
            stderr = str(latest_case_dict.get("blocked_reason") or "")
        if not case_result.get("command") and latest_case_dict.get("command"):
            execution_context["command"] = latest_case_dict.get("command")
        if case_result.get("return_code") is None and latest_case_dict.get("return_code") is not None:
            execution_context["return_code"] = latest_case_dict.get("return_code")

    evidence = {
        "blocked_reason": case_result.get("blocked_reason") or latest_case_dict.get("blocked_reason"),
        "stderr_head": _truncate(stderr),
        "stdout_head": _truncate(stdout),
        "artifact_paths": case_result.get("artifacts")
        or case_result.get("generated_paths")
        or latest_case_dict.get("artifacts"),
        "smoke_run_checks": checks,
        "picking_evaluation": picking_evaluation or None,
        "regression_summary": latest_payload_map.get("regression_summary"),
        "key_signals": _build_key_signals(case_result, checks, classification),
    }

    suggested = _default_suggested_actions(classification)

    audit_metadata = {
        "audit_version": AUDIT_VERSION,
        "generated_at": _utc_now_iso(),
        "source_latest_json": None,
        "source_smoke_summary": None,
    }

    supervision_inputs = {
        "preflight": payloads.get("preflight"),
        "execution": payloads.get("execution"),
        "evaluation": payloads.get("evaluation"),
        "diff_regression": diff_regression,
    }

    run_identity_for_summary = run_identity
    human_text = _build_human_summary_cn(run_identity_for_summary, classification, evidence)

    return {
        "audit_metadata": audit_metadata,
        "run_identity": run_identity,
        "execution_context": execution_context,
        "failure_classification": classification,
        "evidence": evidence,
        "supervision_inputs": supervision_inputs,
        "suggested_actions": suggested,
        "human_summary": human_text,
    }


def attach_audit_sources(
    audit: Dict[str, Any],
    latest_path: Optional[str] = None,
    smoke_summary_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Fill source_* paths in audit_metadata (mutates and returns audit)."""
    meta = dict(audit.get("audit_metadata") or {})
    meta["source_latest_json"] = latest_path
    meta["source_smoke_summary"] = smoke_summary_path
    audit["audit_metadata"] = meta
    return audit


def render_failure_audit_markdown(audit: Mapping[str, Any]) -> str:
    """Render a short human-readable Markdown report."""
    ri = audit.get("run_identity") or {}
    fc = audit.get("failure_classification") or {}
    ev = audit.get("evidence") or {}
    ex = audit.get("execution_context") or {}
    su = audit.get("human_summary") or ""
    actions = audit.get("suggested_actions") or []

    stderr_prev = ""
    sh = ev.get("stderr_head") or ""
    if isinstance(sh, str) and sh.strip():
        lines = [ln for ln in sh.splitlines() if ln.strip()][:12]
        stderr_prev = "\n".join(lines)
        if len(stderr_prev) > 1200:
            stderr_prev = stderr_prev[:1200] + "\n…"

    chk = ev.get("smoke_run_checks") or {}
    pev = ev.get("picking_evaluation") or {}

    parts: List[str] = [
        "# Smoke Run Failure Audit",
        "",
        "## Run Identity",
        "",
        f"- **run_id**: {ri.get('run_id')}",
        f"- **benchmark_id**: {ri.get('benchmark_id')}",
        f"- **backend**: {ri.get('backend')}",
        f"- **status**: {ri.get('status')}",
        f"- **dataset_id**: {ri.get('dataset_id')}",
        "",
        "## Failure Classification",
        "",
        f"- **failure_phase**: {fc.get('failure_phase')}",
        f"- **failure_kind**: {fc.get('failure_kind')}",
        f"- **root_cause_guess**: {fc.get('root_cause_guess')}",
        f"- **confidence**: {fc.get('confidence')}",
        "",
        "## Key Evidence",
        "",
        f"- **preflight_available** / 通过情况: {ex.get('preflight_available')}",
        f"- **return_code**: {ex.get('return_code')}",
        "",
        "### stderr 摘要",
        "",
        "```text",
        stderr_prev or "(empty)",
        "```",
        "",
        "### Artifacts",
        "",
    ]
    arts = ev.get("artifact_paths") or []
    if isinstance(arts, list):
        for a in arts[:20]:
            parts.append(f"- `{a}`")
    else:
        parts.append(f"- `{arts}`")
    parts.extend(
        [
            "",
            "### smoke_run_checks",
            "",
            "```json",
            json.dumps(chk, ensure_ascii=False, indent=2),
            "```",
            "",
            "### picking_evaluation 摘要",
            "",
            "```json",
            json.dumps(pev, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Suggested Actions",
            "",
        ]
    )
    for item in actions:
        parts.append(
            f"- **{item.get('priority', 'n/a')}** — {item.get('title')} (`{item.get('action_id')}`)"
        )
    parts.extend(["", "## Human Summary", "", su, ""])
    return "\n".join(parts)


def audit_dict_to_json_bytes(audit: Mapping[str, Any]) -> bytes:
    """Serialize audit for writing; ensures UTF-8 and stable formatting."""
    return json.dumps(audit, ensure_ascii=False, indent=2).encode("utf-8")
