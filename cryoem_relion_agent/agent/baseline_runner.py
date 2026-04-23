"""Baseline experiment runner: protocol + case -> unified experiment result.

This layer is *above* the existing scheduler / benchmark_harness / persistence
and does not change them.  It:

1. Picks a backend for a case according to the protocol's routing_mode.
2. Builds a benchmark case dict compatible with ``BenchmarkHarness.run_case``.
3. Executes it (or dry-runs it) via the provided harness.
4. Persists the benchmark via existing ``persist_benchmark_run``.
5. Optionally writes a success_audit.json/md or failure_audit.json/md.
6. Returns a JSON-friendly ``ExperimentResult`` dict with audit/benchmark paths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from agent.backend_registry import get_backend_capability_matrix
from agent.benchmark_harness import BenchmarkHarness
from agent.benchmark_persistence import load_latest_benchmark, persist_benchmark_run
from agent.case_input_mapping import map_case_inputs_to_backend_params
from agent.case_registry import ExperimentCase
from agent.experiment_protocol import ExperimentProtocol

CATEGORY_FOR_TARGET_STAGE: Dict[str, str] = {
    "picking": "picking",
    "extraction": "picking",
    "class2d": "picking",
    "heterogeneity": "heterogeneity",
    "postprocess": "model_building",
    "model_building": "model_building",
}

_REGISTRY_DEFAULTS: Dict[str, str] = {
    "picking": "topaz",
    "heterogeneity": "cryodrgn",
    "model_building": "modelangelo",
}


def _case_wants_seq_free(case: ExperimentCase) -> bool:
    """Return True when a map-level case explicitly lacks a sequence.

    ``modelangelo`` is sequence-guided and always needs ``sequence_path``;
    ``emodelx`` exposes a native ``seq_free`` protocol that builds a Cα-only
    model from density alone.  When the case advertises ``has_sequence=false``
    or pins ``emodelx_protocol=seq_free``, routing must prefer ``emodelx``
    over the sequence-guided registry default.
    """
    state = case.input_state or {}
    if str(state.get("emodelx_protocol") or "").strip().lower() == "seq_free":
        return True
    if state.get("has_sequence") is False:
        return True
    return False


@dataclass
class ExperimentResult:
    """JSON-friendly unified result of a single experiment case run."""

    protocol_name: str
    case_id: str
    selected_backend: Optional[str]
    status: Optional[str]
    stage_reached: Optional[str]
    runtime: Optional[float]
    audit_paths: Dict[str, Optional[str]] = field(default_factory=dict)
    benchmark_paths: Dict[str, Optional[str]] = field(default_factory=dict)
    routing_mode: Optional[str] = None
    fallback_applied: bool = False
    llm_supervision_enabled: bool = False
    supervision_payloads: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None
    case_result: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _select_backend(
    protocol: ExperimentProtocol,
    case: ExperimentCase,
) -> Dict[str, Any]:
    """Pick a backend according to protocol.routing_mode.

    Returns a selection_decision dict with:
      selected_backend, selection_reason, rejected_backends, fallback_applied.
    """
    category = CATEGORY_FOR_TARGET_STAGE.get(case.target_stage, "picking")
    allowed = list(protocol.allowed_backends)

    if protocol.routing_mode == "fixed":
        if len(allowed) != 1:
            raise ValueError(
                "fixed routing_mode requires exactly one allowed_backend"
            )
        return {
            "selected_backend": allowed[0],
            "selection_reason": "protocol.routing_mode=fixed",
            "rejected_backends": [],
            "rejection_reasons": {},
            "fallback_applied": False,
            "category": category,
        }

    # Map-level seq_free routing: prefer emodelx over the sequence-guided
    # modelangelo default when the case explicitly lacks a sequence.
    if (
        category == "model_building"
        and _case_wants_seq_free(case)
        and "emodelx" in allowed
    ):
        rejections: Dict[str, str] = {}
        if "modelangelo" in allowed:
            rejections["modelangelo"] = (
                "sequence_path is required by modelangelo but case has "
                "has_sequence=false / emodelx_protocol=seq_free"
            )
        return {
            "selected_backend": "emodelx",
            "selection_reason": (
                "map_level_seq_free: emodelx preferred over sequence-guided "
                "backends (case has_sequence=false / emodelx_protocol=seq_free)"
            ),
            "rejected_backends": [b for b in allowed if b != "emodelx"],
            "rejection_reasons": rejections,
            "fallback_applied": False,
            "category": category,
        }

    registry_default = _REGISTRY_DEFAULTS.get(category)
    if registry_default and registry_default in allowed:
        return {
            "selected_backend": registry_default,
            "selection_reason": f"registry_default for category={category}",
            "rejected_backends": [b for b in allowed if b != registry_default],
            "rejection_reasons": {},
            "fallback_applied": False,
            "category": category,
        }

    if protocol.fallback_policy == "registry_default" and registry_default:
        return {
            "selected_backend": registry_default,
            "selection_reason": "fallback_policy=registry_default",
            "rejected_backends": allowed,
            "rejection_reasons": {},
            "fallback_applied": True,
            "category": category,
        }

    if not allowed:
        raise ValueError("no allowed_backends available for selection")

    return {
        "selected_backend": allowed[0],
        "selection_reason": "first_allowed_backend",
        "rejected_backends": allowed[1:],
        "rejection_reasons": {},
        "fallback_applied": True,
        "category": category,
    }


def _merge_inputs(
    case: ExperimentCase,
    selection: Mapping[str, Any],
) -> Dict[str, Any]:
    """Merge case.input_state + mapped available_inputs with routing hints.

    ``case.available_inputs`` uses experiment-layer semantic field names (e.g.
    ``micrographs_star``).  ``agent.case_input_mapping`` translates those to the
    backend input contract (e.g. ``micrographs_star_path``) so Topaz preflight
    and executor receive the fields they require.  Same-name contract keys
    already present in ``case.input_state`` win over the mapped values.
    """
    backend = str(selection.get("selected_backend") or "")
    category = str(selection.get("category") or "")
    inputs: Dict[str, Any] = dict(case.input_state or {})

    if backend:
        mapped = map_case_inputs_to_backend_params(
            available_inputs=case.available_inputs or {},
            backend=backend,
        )
        for key, value in mapped.items():
            inputs.setdefault(key, value)

    if category == "picking":
        inputs.setdefault("picking_backend", backend)
    elif category == "heterogeneity":
        inputs.setdefault("heterogeneity_backend", backend)
    elif category == "model_building":
        inputs.setdefault("model_building_backend", backend)
    inputs["picking_selection_decision"] = dict(selection)
    inputs["heterogeneity_selection_decision"] = dict(selection)
    inputs["model_building_selection_decision"] = dict(selection)
    return inputs


def build_benchmark_case(
    protocol: ExperimentProtocol,
    case: ExperimentCase,
    *,
    selection: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Translate protocol + case into the dict shape expected by harness.run_case."""
    sel = dict(selection or _select_backend(protocol, case))
    inputs = _merge_inputs(case, sel)
    benchmark_id = f"{protocol.protocol_name}__{case.case_id}"
    return {
        "benchmark_id": benchmark_id,
        "category": sel.get("category"),
        "backend": sel.get("selected_backend"),
        "inputs": inputs,
        "expected_status": None,
        "expected_artifacts": [],
        "notes": (case.notes or protocol.notes or ""),
        "extra": {
            "protocol_name": protocol.protocol_name,
            "case_id": case.case_id,
            "dataset_name": case.dataset_name,
            "dataset_root": case.dataset_root,
            "target_stage": case.target_stage,
            "evaluation_mode": case.evaluation_mode,
            "selection_decision": sel,
            "materialized_case": case.to_dict(),
        },
    }


def _tail(text: Any, *, max_chars: int = 1200) -> Optional[str]:
    if text is None:
        return None
    s = str(text)
    if len(s) <= max_chars:
        return s
    return s[-max_chars:]


def _default_supervision_payload(
    stop_point: str,
    *,
    case: ExperimentCase,
    case_result: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build a JSON-friendly supervision snapshot for a stop_point.

    Consumers (rule_only + llm advice builders, failure audit, future real LLM
    call) need a single consistent shape regardless of whether the case passed
    preflight or failed at execution.
    """
    preflight = case_result.get("preflight") if isinstance(case_result.get("preflight"), Mapping) else {}
    preflight = dict(preflight or {})
    base = {
        "case_id": case.case_id,
        "backend": case_result.get("backend") or selection.get("selected_backend"),
        "category": case_result.get("category") or selection.get("category"),
        "target_stage": case.target_stage,
    }
    if stop_point == "preflight":
        return {
            **base,
            "available": preflight.get("available"),
            "execution_mode": preflight.get("execution_mode"),
            "blocked_reason": preflight.get("blocked_reason"),
            "missing_inputs": list(preflight.get("missing_inputs") or []),
            "missing_input_paths": list(preflight.get("missing_input_paths") or []),
            "missing_dependencies": list(preflight.get("missing_dependencies") or []),
        }
    if stop_point == "execution":
        return {
            **base,
            "status": case_result.get("status"),
            "execution_mode": case_result.get("execution_mode"),
            "command": case_result.get("command"),
            "return_code": case_result.get("return_code"),
            "stdout_tail": _tail(case_result.get("stdout")),
            "stderr_tail": _tail(case_result.get("stderr")),
        }
    if stop_point == "evaluation":
        return {
            **base,
            "status": case_result.get("status"),
            "artifacts": list(case_result.get("artifacts") or []),
            "expected_artifacts": list(case_result.get("expected_artifacts") or []),
            "generated_paths": list(case_result.get("generated_paths") or []),
        }
    if stop_point == "diff":
        return {
            **base,
            "benchmark_id": case_result.get("benchmark_id"),
            "status": case_result.get("status"),
            "generated_paths": list(case_result.get("generated_paths") or []),
            "expected_artifacts": list(case_result.get("expected_artifacts") or []),
        }
    return {**base, "stop_point": stop_point}


def _derive_stage_reached(
    case_result: Mapping[str, Any],
    case: ExperimentCase,
) -> str:
    status = str(case_result.get("status") or "").lower()
    preflight = case_result.get("preflight") or {}
    if status == "preflight_blocked" or not preflight.get("available", True):
        return "preflight"
    if status in {"completed", "success"}:
        return case.target_stage
    if status in {"failed", "error"}:
        return "execution"
    return "unknown"


def _write_audit(
    *,
    project_root: Path,
    output_dir: Path,
    protocol: ExperimentProtocol,
    case: ExperimentCase,
    case_result: Mapping[str, Any],
    latest_payload: Mapping[str, Any],
) -> Dict[str, Optional[str]]:
    """Write success_audit.* or failure_audit.* under output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    status = str(case_result.get("status") or "").lower()

    smoke_summary = {"case_result": dict(case_result)}

    if status in {"completed", "success"}:
        from agent.success_audit import (
            attach_audit_sources as _attach_success,
            audit_dict_to_json_bytes as _to_bytes,
            build_success_audit,
            render_success_audit_markdown,
        )

        audit = build_success_audit(
            latest_payload,
            smoke_summary,
            protocol=protocol.to_dict(),
            case=case.to_dict(),
        )
        latest_path = str(project_root / ".cryoem_agent" / "benchmarks" / "latest.json")
        _attach_success(audit, latest_path=latest_path, smoke_summary_path=None)
        out_json = output_dir / "success_audit.json"
        out_md = output_dir / "success_audit.md"
        out_json.write_bytes(_to_bytes(audit))
        out_md.write_text(render_success_audit_markdown(audit), encoding="utf-8")
        return {"success_audit_json": str(out_json), "success_audit_md": str(out_md)}

    from agent.failure_audit import (
        attach_audit_sources as _attach_failure,
        audit_dict_to_json_bytes as _to_bytes_fail,
        build_failure_audit,
        render_failure_audit_markdown,
    )

    audit = build_failure_audit(latest_payload, smoke_summary)
    latest_path = str(project_root / ".cryoem_agent" / "benchmarks" / "latest.json")
    _attach_failure(audit, latest_path=latest_path, smoke_summary_path=None)
    out_json = output_dir / "failure_audit.json"
    out_md = output_dir / "failure_audit.md"
    out_json.write_bytes(_to_bytes_fail(audit))
    out_md.write_text(render_failure_audit_markdown(audit), encoding="utf-8")
    return {"failure_audit_json": str(out_json), "failure_audit_md": str(out_md)}


def run_experiment_case(
    *,
    protocol: ExperimentProtocol,
    case: ExperimentCase,
    harness: BenchmarkHarness,
    project_root: Path,
    output_dir: Optional[Path] = None,
    dry_run: bool = True,
    persist: bool = True,
    write_audit: bool = True,
) -> ExperimentResult:
    """Execute a single (protocol, case) pair via the existing benchmark harness.

    This function does NOT reimplement workflow execution; it delegates to the
    same ``BenchmarkHarness`` used by all existing smoke runs.
    """
    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir or (project_root / ".cryoem_agent" / "experiments" / protocol.protocol_name / case.case_id)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selection = _select_backend(protocol, case)
    benchmark_case = build_benchmark_case(protocol, case, selection=selection)

    case_result = harness.run_case(benchmark_case, dry_run=dry_run)

    extra = dict(case_result.get("extra") or {})
    extra.setdefault("protocol_name", protocol.protocol_name)
    extra.setdefault("case_id", case.case_id)
    extra.setdefault("dataset_name", case.dataset_name)
    extra.setdefault("target_stage", case.target_stage)
    extra.setdefault("evaluation_mode", case.evaluation_mode)
    extra.setdefault("selection_decision", selection)
    case_result["extra"] = extra

    benchmark_paths: Dict[str, Optional[str]] = {
        "latest_json": None,
        "latest_md": None,
        "history_json": None,
        "history_md": None,
    }
    latest_payload: Dict[str, Any] = {}

    if persist:
        summary = {
            "case_count": 1,
            "completed": 1 if case_result.get("status") in {"completed", "success"} else 0,
            "blocked": 1 if case_result.get("status") == "preflight_blocked" else 0,
            "failed": 1 if case_result.get("status") in {"failed", "error"} else 0,
            "results": [case_result],
        }
        summary["markdown"] = harness.summary_markdown(summary)
        capability = get_backend_capability_matrix()
        persist_info = persist_benchmark_run(
            project_root=project_root,
            summary=summary,
            capability_matrix=capability,
            dry_run=bool(dry_run),
        )
        benchmark_paths.update(
            {
                "latest_json": persist_info.get("latest_json"),
                "latest_md": persist_info.get("latest_md"),
                "history_json": persist_info.get("history_json"),
                "history_md": persist_info.get("history_md"),
            }
        )
        latest_payload = load_latest_benchmark(project_root) or {}
    else:
        latest_payload = load_latest_benchmark(project_root) or {}

    audit_paths: Dict[str, Optional[str]] = {}
    if write_audit:
        audit_paths = _write_audit(
            project_root=project_root,
            output_dir=output_dir,
            protocol=protocol,
            case=case,
            case_result=case_result,
            latest_payload=latest_payload,
        )

    supervision: Dict[str, Any] = {}
    if protocol.llm_supervision_enabled:
        payloads_from_extra = dict(extra.get("llm_supervision_payloads") or {})
        for sp in protocol.stop_points:
            override = payloads_from_extra.get(sp)
            if override is not None:
                supervision[sp] = override
            else:
                supervision[sp] = _default_supervision_payload(
                    sp,
                    case=case,
                    case_result=case_result,
                    selection=selection,
                )

    result = ExperimentResult(
        protocol_name=protocol.protocol_name,
        case_id=case.case_id,
        selected_backend=selection.get("selected_backend"),
        status=case_result.get("status"),
        stage_reached=_derive_stage_reached(case_result, case),
        runtime=case_result.get("runtime_seconds"),
        audit_paths=audit_paths,
        benchmark_paths=benchmark_paths,
        routing_mode=protocol.routing_mode,
        fallback_applied=bool(selection.get("fallback_applied")),
        llm_supervision_enabled=bool(protocol.llm_supervision_enabled),
        supervision_payloads=supervision,
        notes=(case.notes or protocol.notes),
        case_result=dict(case_result),
    )

    experiment_out = output_dir / "experiment_result.json"
    experiment_out.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def run_experiment_cases(
    *,
    protocol: ExperimentProtocol,
    cases: List[ExperimentCase],
    harness: BenchmarkHarness,
    project_root: Path,
    output_dir: Optional[Path] = None,
    dry_run: bool = True,
    persist: bool = True,
    write_audit: bool = True,
) -> List[ExperimentResult]:
    """Run many cases under a single protocol and return a list of results."""
    results: List[ExperimentResult] = []
    for case in cases:
        case_dir: Optional[Path] = None
        if output_dir is not None:
            case_dir = Path(output_dir) / case.case_id
        results.append(
            run_experiment_case(
                protocol=protocol,
                case=case,
                harness=harness,
                project_root=project_root,
                output_dir=case_dir,
                dry_run=dry_run,
                persist=persist,
                write_audit=write_audit,
            )
        )
    return results
