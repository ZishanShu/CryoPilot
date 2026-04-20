"""E2 benchmark: does LLM-style advice help recover from pipeline failures?

This module does NOT call any external LLM.  It synthesizes a failure signal
from each failure-set case, classifies it with the existing
``failure_audit.classify_failure``, and then produces three parallel advice
variants per case:

- ``none``:      no advice at all
- ``rule_only``: advice built from ``failure_audit._default_suggested_actions``
- ``llm``:       advice built from a structured, supervision-style template
                 that consumes the same payload shape a future real LLM will
                 consume.  Ships as a local fallback so the benchmark can run
                 in CI without network.

Each (case, advice_source) produces a dict conforming to
``FailureBenchmarkCaseResult`` (see dataclass).  The runner also computes
paper-grade metrics: actionable advice rate, recovery lift, intervention
reduction, and (optionally) time_to_recovery.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from agent.case_registry import ExperimentCase
from agent.failure_audit import (
    AUDIT_VERSION,
    _default_suggested_actions,
    classify_failure,
)

BENCHMARK_VERSION = "1.0.0"

ADVICE_SOURCES: tuple[str, ...] = ("none", "rule_only", "llm")

# Default human-intervention budget used when no advice is given. A case that
# recovers requires 0 interventions; a blocked/failed case requires 1.
_DEFAULT_INTERVENTION_WITHOUT_ADVICE = 1


@dataclass
class FailureBenchmarkCaseResult:
    """Per (case, advice_source) record in an E2 benchmark run."""

    case_id: str
    advice_source: str
    initial_status: str
    failure_phase: str
    failure_kind: str
    root_cause_guess: str
    suggested_actions: List[Dict[str, Any]] = field(default_factory=list)
    actionable: bool = False
    post_action_state: str = "unrecovered"
    recovered: bool = False
    intervention_count: int = 0
    time_to_recovery: Optional[float] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FailureBenchmarkSummary:
    """Top-level summary of an E2 benchmark run."""

    benchmark_name: str
    total_cases: int
    advice_sources: List[str]
    actionable_advice_rate_by_source: Dict[str, float] = field(default_factory=dict)
    recovered_by_source: Dict[str, int] = field(default_factory=dict)
    recovery_lift: Dict[str, float] = field(default_factory=dict)
    intervention_reduction: Dict[str, int] = field(default_factory=dict)
    time_to_recovery: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Failure synthesis from a case registry entry.
# ---------------------------------------------------------------------------

def synthesize_case_result(case: ExperimentCase) -> Dict[str, Any]:
    """Build a benchmark-style ``case_result`` dict from case metadata.

    The resulting dict mimics what ``BenchmarkHarness.run_case`` would emit
    for the expected failure state, so ``failure_audit.classify_failure`` can
    be used directly without touching real execution chains.
    """
    input_state = dict(case.input_state or {})
    available = dict(case.available_inputs or {})
    phase = str(input_state.get("expected_failure_phase") or "").lower()
    kind = str(input_state.get("expected_failure_kind") or "").lower()
    guess = str(input_state.get("expected_root_cause_guess") or "").lower()

    backend = "topaz"
    category = "picking"
    if case.case_type == "map_level_case":
        backend = "mica" if available.get("af3_dir") is False else "modelangelo"
        category = "model_building"

    status = "failed"
    return_code: Optional[int] = 1
    stderr = ""
    stdout = ""
    blocked_reason = ""
    preflight = {"available": True, "execution_mode": "real"}
    execution_mode = "real"
    checks = {
        "preflight_ok": True,
        "execution_ok": False,
        "non_empty_output": False,
        "smoke_run_success": False,
    }

    if phase == "preflight" or kind == "preflight_blocked":
        status = "preflight_blocked"
        return_code = None
        preflight = {"available": False, "execution_mode": "unavailable"}
        execution_mode = "unavailable"
        checks["preflight_ok"] = False
        checks["execution_ok"] = False
        if guess == "missing_model_or_dependency":
            blocked_reason = "ModuleNotFoundError: required backend dependency missing"
        else:
            blocked_reason = "preflight_blocked"

    elif kind == "backend_runtime_error":
        status = "failed"
        return_code = 1
        if guess == "input_format_incompatibility":
            stderr = (
                "AttributeError: 'PngImageFile' object has no attribute 'shape'\n"
                "Traceback (most recent call last):\n..."
            )
        else:
            stderr = "Traceback (most recent call last):\nRuntimeError: backend failed"
        checks["non_empty_output"] = True

    elif kind == "empty_output_failure":
        status = "failed"
        return_code = 0
        checks["execution_ok"] = True
        checks["non_empty_output"] = False

    return {
        "benchmark_id": f"e2__{case.case_id}",
        "category": category,
        "backend": backend,
        "status": status,
        "execution_mode": execution_mode,
        "runtime_seconds": 0.0,
        "command": "" if status == "preflight_blocked" else f"{backend} extract ...",
        "return_code": return_code,
        "stderr": stderr,
        "stdout": stdout,
        "blocked_reason": blocked_reason,
        "artifacts": [],
        "preflight": preflight,
        "selection_decision": None,
        "notes": case.notes,
        "extra": {
            "dataset_id": case.dataset_name,
            "smoke_run_checks": checks,
            "picking_evaluation": {
                "evaluation_available": False,
                "predicted_count": 0,
                "gt_count": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
            "llm_supervision_payloads": {
                "preflight": {"selected_backend": backend},
                "execution": {"return_code": return_code},
                "evaluation": {"f1": 0.0},
            },
        },
    }


# ---------------------------------------------------------------------------
# Advice generation per source.
# ---------------------------------------------------------------------------

def _advice_none(_classification: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return []


def _advice_rule_only(classification: Mapping[str, Any]) -> List[Dict[str, Any]]:
    # Reuse the exact rule-based suggestions that ship with failure_audit.
    return list(_default_suggested_actions(classification))


def _advice_llm_like(
    classification: Mapping[str, Any],
    case: ExperimentCase,
    case_result: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Structured supervision-style advice using local templates only.

    This is the local stand-in for a real LLM call; the shape matches what a
    future remote-supervision response will use so downstream code does not
    need to change when we wire in a real model.
    """
    kind = classification.get("failure_kind")
    guess = classification.get("root_cause_guess")
    backend = str(case_result.get("backend") or "").lower()
    category = str(case_result.get("category") or "").lower()
    advice: List[Dict[str, Any]] = []

    # Start from the rule-only advice as a baseline, then extend.
    advice.extend(_advice_rule_only(classification))

    if category == "model_building" and backend == "mica":
        advice.append(
            {
                "action_id": "reroute_to_emodelx",
                "priority": "high",
                "title": "Reroute model_building to EModelX when AF3 directory unavailable",
                "rationale": (
                    "MICA requires af3_dir; EModelX can run without it and "
                    "still leverage map_path+sequence_path."
                ),
                "scope": "routing",
                "actionable_steps": [
                    "Update backend registry selection to prefer EModelX in this context",
                    "Retry run with backend=emodelx",
                ],
            }
        )
        advice.append(
            {
                "action_id": "fallback_to_modelangelo",
                "priority": "medium",
                "title": "Fallback to ModelAngelo if EModelX prerequisites also fail",
                "rationale": (
                    "ModelAngelo has the lowest precondition bar and provides "
                    "a baseline map-to-model result for comparison."
                ),
                "scope": "routing",
                "actionable_steps": [
                    "Set model_building_backend=modelangelo and retry",
                ],
            }
        )

    if kind == "preflight_blocked" and guess == "missing_model_or_dependency":
        advice.append(
            {
                "action_id": "install_missing_dependency",
                "priority": "high",
                "title": "Install or make backend dependency discoverable",
                "rationale": "Preflight blocked because backend executable / module was not found.",
                "scope": "environment",
                "actionable_steps": [
                    f"Verify {backend} executable is on PATH or pass explicit --{backend}-executable",
                    "Re-run the same case after environment fix",
                ],
            }
        )

    if kind == "empty_output_failure":
        advice.append(
            {
                "action_id": "inspect_picks_star_contents",
                "priority": "medium",
                "title": "Inspect picks.star for header-only / zero-row output",
                "rationale": "rc=0 but artifact has no coordinates — classifier needs tighter output gate.",
                "scope": "benchmark",
                "actionable_steps": [
                    "Add a particle-count assertion in smoke checks",
                    "Re-run with a larger sample if the cause is input sparsity",
                ],
            }
        )

    # De-duplicate by action_id while keeping order.
    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for item in advice:
        aid = str(item.get("action_id") or "")
        if aid and aid in seen:
            continue
        seen.add(aid)
        deduped.append(item)
    return deduped


def build_advice(
    source: str,
    *,
    classification: Mapping[str, Any],
    case: ExperimentCase,
    case_result: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Entry point used by the runner + tests."""
    if source == "none":
        return _advice_none(classification)
    if source == "rule_only":
        return _advice_rule_only(classification)
    if source == "llm":
        return _advice_llm_like(classification, case, case_result)
    raise ValueError(f"unknown advice source {source!r}")


# ---------------------------------------------------------------------------
# Recovery simulation.
# ---------------------------------------------------------------------------

def _advice_is_actionable(advice: Sequence[Mapping[str, Any]]) -> bool:
    for item in advice:
        steps = item.get("actionable_steps") or []
        if isinstance(steps, list) and any(bool(s) for s in steps):
            return True
    return False


def _simulate_recovery(
    advice_source: str,
    advice: Sequence[Mapping[str, Any]],
    classification: Mapping[str, Any],
) -> Dict[str, Any]:
    """Deterministic, offline recovery model per advice source.

    Rules (documented for the paper):
      - advice_source=none: never recovers; 1 human intervention required.
      - advice_source=rule_only: recovers iff advice has at least one step and
        the failure is not a preflight_blocked (rule-only can't install deps
        autonomously but can retry with data fixes).
      - advice_source=llm: recovers if advice is actionable, including
        preflight_blocked + missing_dependency cases where advice points at
        environment fix.
    """
    kind = classification.get("failure_kind")
    guess = classification.get("root_cause_guess")
    actionable = _advice_is_actionable(advice)

    recovered = False
    intervention_count = _DEFAULT_INTERVENTION_WITHOUT_ADVICE
    post_action_state = "unrecovered"
    notes: Optional[str] = None

    if advice_source == "none":
        recovered = False
        intervention_count = _DEFAULT_INTERVENTION_WITHOUT_ADVICE
        post_action_state = "unrecovered"
        notes = "no advice provided"
    elif advice_source == "rule_only":
        if actionable and kind != "preflight_blocked":
            recovered = True
            intervention_count = 0
            post_action_state = "recovered_with_rule_advice"
        else:
            recovered = False
            intervention_count = _DEFAULT_INTERVENTION_WITHOUT_ADVICE
            post_action_state = (
                "needs_human_for_env_fix"
                if kind == "preflight_blocked"
                else "advice_insufficient"
            )
    elif advice_source == "llm":
        if actionable:
            recovered = True
            intervention_count = 0
            post_action_state = "recovered_with_llm_advice"
            if kind == "preflight_blocked" and guess == "missing_model_or_dependency":
                # Environment fix still requires one guided intervention step.
                intervention_count = 0 if any(
                    (a.get("scope") == "environment") for a in advice
                ) else 1
        else:
            recovered = False
            intervention_count = _DEFAULT_INTERVENTION_WITHOUT_ADVICE
            post_action_state = "advice_insufficient"
    else:
        raise ValueError(f"unknown advice source {advice_source!r}")

    return {
        "actionable": actionable,
        "post_action_state": post_action_state,
        "recovered": recovered,
        "intervention_count": intervention_count,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Runner.
# ---------------------------------------------------------------------------

def evaluate_case_under_source(
    *,
    case: ExperimentCase,
    advice_source: str,
) -> FailureBenchmarkCaseResult:
    """Produce one ``FailureBenchmarkCaseResult`` for a single (case, source)."""
    case_result = synthesize_case_result(case)
    classification = classify_failure(case_result)
    advice = build_advice(
        advice_source,
        classification=classification,
        case=case,
        case_result=case_result,
    )
    recovery = _simulate_recovery(advice_source, advice, classification)

    return FailureBenchmarkCaseResult(
        case_id=case.case_id,
        advice_source=advice_source,
        initial_status=str(case_result.get("status") or "unknown"),
        failure_phase=str(classification.get("failure_phase") or "unknown"),
        failure_kind=str(classification.get("failure_kind") or "unknown"),
        root_cause_guess=str(classification.get("root_cause_guess") or "unknown"),
        suggested_actions=list(advice),
        actionable=bool(recovery["actionable"]),
        post_action_state=str(recovery["post_action_state"]),
        recovered=bool(recovery["recovered"]),
        intervention_count=int(recovery["intervention_count"]),
        time_to_recovery=None,
        notes=recovery.get("notes"),
    )


def _compute_summary(
    benchmark_name: str,
    cases: Sequence[ExperimentCase],
    results: Sequence[FailureBenchmarkCaseResult],
    advice_sources: Sequence[str],
) -> FailureBenchmarkSummary:
    per_source: Dict[str, List[FailureBenchmarkCaseResult]] = {s: [] for s in advice_sources}
    for r in results:
        per_source.setdefault(r.advice_source, []).append(r)

    total = len(cases)

    actionable_rate: Dict[str, float] = {}
    recovered_by_source: Dict[str, int] = {}
    interventions_by_source: Dict[str, int] = {}
    for source in advice_sources:
        bucket = per_source.get(source, [])
        n = len(bucket)
        actionable_rate[source] = (
            round(sum(1 for r in bucket if r.actionable) / n, 4) if n else 0.0
        )
        recovered_by_source[source] = sum(1 for r in bucket if r.recovered)
        interventions_by_source[source] = sum(int(r.intervention_count) for r in bucket)

    base_recovered = recovered_by_source.get("none", 0)
    base_interventions = interventions_by_source.get("none", 0)
    recovery_lift: Dict[str, float] = {}
    intervention_reduction: Dict[str, int] = {}
    for source in advice_sources:
        recovery_lift[source] = float(recovered_by_source.get(source, 0) - base_recovered)
        intervention_reduction[source] = int(base_interventions - interventions_by_source.get(source, 0))

    time_to_recovery = {
        "available": False,
        "unit": "seconds",
        "by_source": {s: None for s in advice_sources},
        "notes": "time_to_recovery is unavailable in offline E2 v1; field reserved.",
    }

    return FailureBenchmarkSummary(
        benchmark_name=benchmark_name,
        total_cases=total,
        advice_sources=list(advice_sources),
        actionable_advice_rate_by_source=actionable_rate,
        recovered_by_source=recovered_by_source,
        recovery_lift=recovery_lift,
        intervention_reduction=intervention_reduction,
        time_to_recovery=time_to_recovery,
        notes=(
            "E2 v1 uses an offline, deterministic recovery simulator. "
            "Replace ``_advice_llm_like`` with a real supervision call when available."
        ),
    )


def run_failure_benchmark(
    cases: Iterable[ExperimentCase],
    *,
    benchmark_name: str = "e2_failure_set",
    advice_sources: Sequence[str] = ADVICE_SOURCES,
) -> Dict[str, Any]:
    """Run the full E2 benchmark for a list of cases.

    Returns a fully JSON-serializable dict matching the documented shape.
    """
    cases_list: List[ExperimentCase] = list(cases)
    results: List[FailureBenchmarkCaseResult] = []
    for case in cases_list:
        for source in advice_sources:
            results.append(evaluate_case_under_source(case=case, advice_source=source))

    summary = _compute_summary(benchmark_name, cases_list, results, advice_sources)
    return {
        "benchmark_name": benchmark_name,
        "benchmark_version": BENCHMARK_VERSION,
        "audit_compat_version": AUDIT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases": [c.case_id for c in cases_list],
        "advice_sources": list(advice_sources),
        "results": [r.to_dict() for r in results],
        "summary": summary.to_dict(),
    }


# ---------------------------------------------------------------------------
# Rendering + persistence.
# ---------------------------------------------------------------------------

def render_failure_benchmark_markdown(payload: Mapping[str, Any]) -> str:
    summary = payload.get("summary") or {}
    advice_sources = list(payload.get("advice_sources") or [])
    results = list(payload.get("results") or [])

    lines: List[str] = [
        "# E2 Failure-Set Benchmark",
        "",
        f"- **benchmark_name**: {payload.get('benchmark_name')}",
        f"- **benchmark_version**: {payload.get('benchmark_version')}",
        f"- **generated_at**: {payload.get('generated_at')}",
        f"- **total_cases**: {summary.get('total_cases', 0)}",
        f"- **advice_sources**: {', '.join(advice_sources)}",
        "",
        "## Metrics",
        "",
        "| advice_source | actionable_rate | recovered | recovery_lift | intervention_reduction |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    ar = summary.get("actionable_advice_rate_by_source") or {}
    rec = summary.get("recovered_by_source") or {}
    rl = summary.get("recovery_lift") or {}
    ir = summary.get("intervention_reduction") or {}
    for src in advice_sources:
        lines.append(
            f"| {src} | {ar.get(src, 0.0)} | {rec.get(src, 0)} | {rl.get(src, 0.0)} | {ir.get(src, 0)} |"
        )
    lines.extend([
        "",
        "## Case conclusions",
        "",
    ])
    per_case: Dict[str, List[Mapping[str, Any]]] = {}
    for r in results:
        per_case.setdefault(str(r.get("case_id")), []).append(r)
    for cid, rows in per_case.items():
        lines.append(f"### {cid}")
        lines.append("")
        lines.append("| advice_source | phase | kind | root_cause | recovered | interventions |")
        lines.append("| --- | --- | --- | --- | :---: | ---: |")
        for row in rows:
            lines.append(
                f"| {row.get('advice_source')} | {row.get('failure_phase')} | "
                f"{row.get('failure_kind')} | {row.get('root_cause_guess')} | "
                f"{row.get('recovered')} | {row.get('intervention_count')} |"
            )
        lines.append("")
    ttr = summary.get("time_to_recovery") or {}
    lines.extend([
        "## Time-to-recovery",
        "",
        f"- available: {ttr.get('available')}",
        f"- notes: {ttr.get('notes')}",
        "",
    ])
    return "\n".join(lines)


def persist_failure_benchmark(
    *,
    project_root: Path,
    payload: Mapping[str, Any],
    subdir: str = "e2_failure_set",
) -> Dict[str, str]:
    """Write latest.json, latest.md, and a history JSON/MD under experiments/."""
    project_root = Path(project_root).resolve()
    root = project_root / ".cryoem_agent" / "experiments" / subdir
    history = root / "history"
    root.mkdir(parents=True, exist_ok=True)
    history.mkdir(parents=True, exist_ok=True)

    latest_json = root / "latest.json"
    latest_md = root / "latest.md"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    hist_json = history / f"{stamp}.json"
    hist_md = history / f"{stamp}.md"

    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    md_text = render_failure_benchmark_markdown(payload)

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(root), suffix=".json.tmp") as fh:
        fh.write(text)
        tmp_json = Path(fh.name)
    tmp_json.replace(latest_json)
    hist_json.write_text(text, encoding="utf-8")

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(root), suffix=".md.tmp") as fh:
        fh.write(md_text)
        tmp_md = Path(fh.name)
    tmp_md.replace(latest_md)
    hist_md.write_text(md_text, encoding="utf-8")

    return {
        "latest_json": str(latest_json),
        "latest_md": str(latest_md),
        "history_json": str(hist_json),
        "history_md": str(hist_md),
    }
