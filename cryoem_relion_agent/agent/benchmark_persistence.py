"""Helpers for persisting benchmark runs to .cryoem_agent/benchmarks."""

from __future__ import annotations

import json
import os
import platform
import tempfile
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _run_git_command(args: List[str], cwd: Optional[Path]) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git"] + args,
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except (OSError, ValueError):
        return None
    out = (completed.stdout or "").strip()
    return out or None


def collect_environment_snapshot(root: Path) -> Dict[str, Any]:
    """Collect lightweight reproducibility metadata, tolerant to missing git."""
    root = root.resolve()
    commit = _run_git_command(["rev-parse", "HEAD"], cwd=root)
    branch = _run_git_command(["rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
    dirty: Optional[bool] = None
    try:
        status_out = _run_git_command(["status", "--porcelain"], cwd=root)
        if status_out is not None:
            dirty = bool(status_out.strip())
    except Exception:
        dirty = None
    try:
        import sys as _sys
        py_version = _sys.version
    except Exception:
        py_version = "unknown"
    try:
        plat = platform.platform()
    except Exception:
        plat = "unknown"
    try:
        host = socket.gethostname()
    except Exception:
        host = "unknown"

    return {
        "git_commit": commit or "unknown",
        "git_branch": branch or "unknown",
        "git_dirty": dirty,
        "python_version": py_version,
        "platform": plat,
        "hostname": host,
        "enabled_backends": None,
        "test_scope": "cryoem_relion_agent/tests",
    }


def make_run_id(ts: Optional[datetime] = None) -> str:
    """Generate a filesystem-safe benchmark run id with microseconds and a short suffix."""
    ts = ts or datetime.now(timezone.utc)
    base = ts.strftime("bench_%Y%m%dT%H%M%S%f")
    # Derive a short hexadecimal suffix from microsecond-resolution timestamp.
    suffix = hex(int(ts.timestamp() * 1_000_000))[-6:]
    return f"{base}Z_{suffix}"


def _uniquify_history_path(path: Path) -> Path:
    """Append _1, _2, ... until a free path is found."""
    if not path.exists():
        return path
    stem = path.stem
    sufix = ""
    if "_" in stem:
        stem, sufix = stem.rsplit("_", 1)
        # If last part is not numeric, keep entire stem.
        if not sufix.isdigit():
            stem = path.stem
    idx = 1
    parent = path.parent
    suffix = path.suffix
    while True:
        candidate = parent / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def load_previous_benchmark_payload(project_root: Path) -> Optional[Dict[str, Any]]:
    """Load previous latest payload before current benchmark overwrites it."""
    return load_latest_benchmark(project_root)


def _status_rank(status: Optional[str]) -> int:
    s = str(status or "").strip().lower()
    if s in {"completed", "success"}:
        return 3
    if s == "preflight_blocked":
        return 2
    if s in {"failed", "error"}:
        return 1
    return 0


def summarize_case_diff(
    current_case: Optional[Dict[str, Any]],
    previous_case: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    current_case = current_case or {}
    previous_case = previous_case or {}
    benchmark_id = current_case.get("benchmark_id") or previous_case.get("benchmark_id")
    prev_status = previous_case.get("status")
    curr_status = current_case.get("status")
    prev_mode = previous_case.get("execution_mode")
    curr_mode = current_case.get("execution_mode")
    prev_runtime = previous_case.get("runtime_seconds")
    curr_runtime = current_case.get("runtime_seconds")
    runtime_delta = None
    if isinstance(prev_runtime, (int, float)) and isinstance(curr_runtime, (int, float)):
        runtime_delta = round(float(curr_runtime) - float(prev_runtime), 4)
    prev_sel = previous_case.get("selection_decision") if isinstance(previous_case.get("selection_decision"), dict) else {}
    curr_sel = current_case.get("selection_decision") if isinstance(current_case.get("selection_decision"), dict) else {}
    if not previous_case:
        diff_class = "new_case"
    elif not current_case:
        diff_class = "removed_case"
    else:
        prev_rank = _status_rank(prev_status)
        curr_rank = _status_rank(curr_status)
        if curr_rank > prev_rank:
            diff_class = "improved"
        elif curr_rank < prev_rank:
            diff_class = "regressed"
        else:
            diff_class = "unchanged"
    return {
        "benchmark_id": benchmark_id,
        "backend": current_case.get("backend") or previous_case.get("backend"),
        "category": current_case.get("category") or previous_case.get("category"),
        "previous_status": prev_status,
        "current_status": curr_status,
        "status_changed": prev_status != curr_status,
        "previous_execution_mode": prev_mode,
        "current_execution_mode": curr_mode,
        "runtime_seconds_delta": runtime_delta,
        "blocked_reason_changed": previous_case.get("blocked_reason") != current_case.get("blocked_reason"),
        "command_changed": previous_case.get("command") != current_case.get("command"),
        "artifacts_changed": list(previous_case.get("artifacts") or []) != list(current_case.get("artifacts") or []),
        "selection_changed": prev_sel != curr_sel,
        "diff_class": diff_class,
    }


def diff_benchmark_payloads(
    current_payload: Dict[str, Any],
    previous_payload: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate structured run-to-run diff and regression summary."""
    if not isinstance(previous_payload, dict):
        return {
            "previous_run_metadata": {
                "previous_run_id": None,
                "previous_timestamp": None,
                "previous_git_commit": None,
                "previous_git_branch": None,
                "first_run": True,
            },
            "benchmark_diff": {
                "overall_status_change": "first_run",
                "case_diffs": [],
                "summary_diffs": {},
                "capability_diffs": {"added_backends": [], "removed_backends": [], "changed_backends": [], "changed_fields_by_backend": {}},
                "selection_diffs": {
                    "cases_with_selection_change": [],
                    "cases_with_reason_change": [],
                    "cases_with_fallback_change": [],
                },
            },
            "regression_summary": {
                "first_run": True,
                "improved_cases": [],
                "regressed_cases": [],
                "unchanged_cases": [],
                "newly_blocked_cases": [],
                "newly_completed_cases": [],
                "missing_previous_cases": [],
                "new_cases": [],
                "removed_cases": [],
            },
        }

    prev_meta = previous_payload.get("run_metadata") or {}
    prev_env = previous_payload.get("environment_snapshot") or {}
    curr_results = {str(x.get("benchmark_id")): x for x in list(current_payload.get("benchmark_results") or []) if x.get("benchmark_id")}
    prev_results = {str(x.get("benchmark_id")): x for x in list(previous_payload.get("benchmark_results") or []) if x.get("benchmark_id")}
    case_ids = sorted(set(curr_results) | set(prev_results))
    case_diffs = [summarize_case_diff(curr_results.get(cid), prev_results.get(cid)) for cid in case_ids]

    improved_cases = [d["benchmark_id"] for d in case_diffs if d["diff_class"] == "improved"]
    regressed_cases = [d["benchmark_id"] for d in case_diffs if d["diff_class"] == "regressed"]
    unchanged_cases = [d["benchmark_id"] for d in case_diffs if d["diff_class"] == "unchanged"]
    new_cases = [d["benchmark_id"] for d in case_diffs if d["diff_class"] == "new_case"]
    removed_cases = [d["benchmark_id"] for d in case_diffs if d["diff_class"] == "removed_case"]
    newly_blocked_cases = [d["benchmark_id"] for d in case_diffs if d["previous_status"] in {"completed", "success"} and d["current_status"] == "preflight_blocked"]
    newly_completed_cases = [d["benchmark_id"] for d in case_diffs if d["current_status"] in {"completed", "success"} and d["previous_status"] not in {"completed", "success"}]

    curr_summary = current_payload.get("benchmark_summary") or {}
    prev_summary = previous_payload.get("benchmark_summary") or {}
    summary_diffs = {
        "total_cases_delta": int(curr_summary.get("total_cases", 0)) - int(prev_summary.get("total_cases", 0)),
        "completed_delta": int(curr_summary.get("completed", 0)) - int(prev_summary.get("completed", 0)),
        "blocked_delta": int(curr_summary.get("blocked", 0)) - int(prev_summary.get("blocked", 0)),
        "failed_delta": int(curr_summary.get("failed", 0)) - int(prev_summary.get("failed", 0)),
        "backend_set_added": sorted(set(curr_summary.get("backends_covered") or []) - set(prev_summary.get("backends_covered") or [])),
        "backend_set_removed": sorted(set(prev_summary.get("backends_covered") or []) - set(curr_summary.get("backends_covered") or [])),
        "category_set_added": sorted(set(curr_summary.get("categories_covered") or []) - set(prev_summary.get("categories_covered") or [])),
        "category_set_removed": sorted(set(prev_summary.get("categories_covered") or []) - set(curr_summary.get("categories_covered") or [])),
    }

    curr_cap = current_payload.get("capability_snapshot") or {}
    prev_cap = previous_payload.get("capability_snapshot") or {}
    added_backends = sorted(set(curr_cap) - set(prev_cap))
    removed_backends = sorted(set(prev_cap) - set(curr_cap))
    changed_fields_by_backend: Dict[str, List[str]] = {}
    keys_to_compare = [
        "maturity_level",
        "execution_mode_supported",
        "required_inputs",
        "required_dependencies",
        "supports_af3_prior",
        "supports_map_input",
        "supports_sequence_input",
        "supports_particle_input",
        "supports_consensus_pose",
        "fallback_candidates",
    ]
    for backend in sorted(set(curr_cap) & set(prev_cap)):
        changed = [k for k in keys_to_compare if curr_cap.get(backend, {}).get(k) != prev_cap.get(backend, {}).get(k)]
        if changed:
            changed_fields_by_backend[backend] = changed

    curr_sel = current_payload.get("selection_snapshot") or {}
    prev_sel = previous_payload.get("selection_snapshot") or {}
    sel_change: List[str] = []
    reason_change: List[str] = []
    fallback_change: List[str] = []
    for case_id in sorted(set(curr_sel) & set(prev_sel)):
        c = curr_sel.get(case_id) or {}
        p = prev_sel.get(case_id) or {}
        if c.get("selected_backend") != p.get("selected_backend"):
            sel_change.append(case_id)
        if c.get("selection_reason") != p.get("selection_reason"):
            reason_change.append(case_id)
        if bool(c.get("fallback_applied")) != bool(p.get("fallback_applied")):
            fallback_change.append(case_id)

    benchmark_diff = {
        "overall_status_change": "regressed" if regressed_cases else ("improved" if improved_cases else "unchanged"),
        "case_diffs": case_diffs,
        "summary_diffs": summary_diffs,
        "capability_diffs": {
            "added_backends": added_backends,
            "removed_backends": removed_backends,
            "changed_backends": sorted(changed_fields_by_backend.keys()),
            "changed_fields_by_backend": changed_fields_by_backend,
        },
        "selection_diffs": {
            "cases_with_selection_change": sel_change,
            "cases_with_reason_change": reason_change,
            "cases_with_fallback_change": fallback_change,
        },
    }
    regression_summary = {
        "first_run": False,
        "improved_cases": improved_cases,
        "regressed_cases": regressed_cases,
        "unchanged_cases": unchanged_cases,
        "newly_blocked_cases": newly_blocked_cases,
        "newly_completed_cases": newly_completed_cases,
        "missing_previous_cases": new_cases,
        "new_cases": new_cases,
        "removed_cases": removed_cases,
    }
    previous_run_metadata = {
        "previous_run_id": prev_meta.get("run_id"),
        "previous_timestamp": prev_meta.get("timestamp"),
        "previous_git_commit": prev_env.get("git_commit"),
        "previous_git_branch": prev_env.get("git_branch"),
        "first_run": False,
    }
    return {
        "previous_run_metadata": previous_run_metadata,
        "benchmark_diff": benchmark_diff,
        "regression_summary": regression_summary,
    }


def render_benchmark_diff_markdown(payload: Dict[str, Any]) -> str:
    """Render concise markdown including run-to-run diff summary."""
    run_meta = payload.get("run_metadata") or {}
    env = payload.get("environment_snapshot") or {}
    summary = payload.get("benchmark_summary") or {}
    reg = payload.get("regression_summary") or {}
    prev = payload.get("previous_run_metadata") or {}
    lines = [
        "# Backend Benchmark Summary",
        "",
        f"- Run ID: {run_meta.get('run_id') or 'N/A'}",
        f"- Timestamp: {run_meta.get('timestamp') or 'N/A'}",
        f"- Git commit: {env.get('git_commit') or 'unknown'}",
        f"- Cases: {summary.get('total_cases', 0)}",
        f"- Completed: {summary.get('completed', 0)}",
        f"- Blocked: {summary.get('blocked', 0)}",
        f"- Failed: {summary.get('failed', 0)}",
        "",
    ]
    if reg.get("first_run"):
        lines.append("- No previous benchmark run to compare.")
    else:
        lines.extend([
            f"- Compared with previous run: {prev.get('previous_run_id') or 'unknown'}",
            f"- Improved: {len(reg.get('improved_cases') or [])}",
            f"- Regressed: {len(reg.get('regressed_cases') or [])}",
            f"- Unchanged: {len(reg.get('unchanged_cases') or [])}",
            f"- New cases: {len(reg.get('new_cases') or [])}",
            f"- Removed cases: {len(reg.get('removed_cases') or [])}",
        ])
        if reg.get("newly_completed_cases"):
            lines.append(f"- Newly completed: {', '.join(reg.get('newly_completed_cases') or [])}")
        if reg.get("regressed_cases"):
            lines.append(f"- Regressed cases: {', '.join(reg.get('regressed_cases') or [])}")
    lines.extend([
        "",
        "## Results",
    ])
    for item in payload.get("benchmark_results") or []:
        lines.append(
            f"- {item.get('benchmark_id')}: backend={item.get('backend')} "
            f"status={item.get('status')} mode={item.get('execution_mode')}"
        )
    lines.append("")
    return "\n".join(lines)


def build_persistence_payload(
    *,
    project_root: Path,
    run_id: str,
    dry_run: bool,
    summary: Dict[str, Any],
    capability_matrix: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    ts = datetime.now(timezone.utc)
    env = collect_environment_snapshot(project_root)
    results: List[Dict[str, Any]] = list(summary.get("results") or [])
    total_cases = int(summary.get("case_count", len(results)))
    completed = int(summary.get("completed", 0))
    blocked = int(summary.get("blocked", 0))
    failed = int(summary.get("failed", 0))
    backends = sorted({str(r.get("backend") or "").strip() for r in results if r.get("backend")})
    categories = sorted({str(r.get("category") or "").strip() for r in results if r.get("category")})
    selection_snapshot: Dict[str, Any] = {}
    for r in results:
        sd = r.get("selection_decision") or {}
        if not isinstance(sd, dict) or not sd.get("selected_backend"):
            continue
        selection_snapshot[r.get("benchmark_id")] = {
            "selected_backend": sd.get("selected_backend"),
            "selection_reason": sd.get("selection_reason"),
            "rejected_backends": list(sd.get("rejected_backends") or []),
            "rejection_reasons": dict(sd.get("rejection_reasons") or {}),
            "fallback_applied": bool(sd.get("fallback_applied")),
        }

    run_metadata = {
        "run_id": run_id,
        "timestamp": ts.isoformat(),
        "benchmark_count": total_cases,
        "completed_count": completed,
        "blocked_count": blocked,
        "failed_count": failed,
        "dry_run": bool(dry_run),
        "working_directory": os.getcwd(),
        "benchmark_root": str((project_root / ".cryoem_agent" / "benchmarks").resolve()),
    }
    benchmark_summary = {
        "total_cases": total_cases,
        "completed": completed,
        "blocked": blocked,
        "failed": failed,
        "backends_covered": backends,
        "categories_covered": categories,
        "summary_markdown": summary.get("markdown"),
    }
    benchmark_results: List[Dict[str, Any]] = []
    for r in results:
        entry = {
            "benchmark_id": r.get("benchmark_id"),
            "category": r.get("category"),
            "backend": r.get("backend"),
            "status": r.get("status"),
            "execution_mode": r.get("execution_mode"),
            "runtime_seconds": r.get("runtime_seconds"),
            "blocked_reason": r.get("blocked_reason"),
            "command": r.get("command"),
            "artifacts": list(r.get("artifacts") or []),
            "preflight": r.get("preflight"),
            "selection_decision": r.get("selection_decision"),
            "notes": r.get("notes"),
            "generated_paths": list(r.get("generated_paths") or []),
            "topaz_output_path": r.get("topaz_output_path"),
            "return_code": r.get("return_code"),
            "extra": r.get("extra"),
        }
        benchmark_results.append(entry)

    payload = {
        "run_metadata": run_metadata,
        "benchmark_summary": benchmark_summary,
        "benchmark_results": benchmark_results,
        "capability_snapshot": capability_matrix,
        "selection_snapshot": selection_snapshot,
        "environment_snapshot": env,
    }
    return payload


def persist_benchmark_run(
    *,
    project_root: Path,
    summary: Dict[str, Any],
    capability_matrix: Dict[str, Dict[str, Any]],
    dry_run: bool,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Write latest + history JSON/Markdown for one benchmark run."""
    project_root = project_root.resolve()
    bench_root = project_root / ".cryoem_agent" / "benchmarks"
    history_dir = bench_root / "history"
    bench_root.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc)
    effective_run_id = run_id or make_run_id(ts)
    previous_payload = load_previous_benchmark_payload(project_root)
    payload = build_persistence_payload(
        project_root=project_root,
        run_id=effective_run_id,
        dry_run=dry_run,
        summary=summary,
        capability_matrix=capability_matrix,
    )
    payload.update(diff_benchmark_payloads(payload, previous_payload))
    stamp = ts.strftime("%Y%m%dT%H%M%S")
    base_name = f"{stamp}_{effective_run_id}"
    latest_json = bench_root / "latest.json"
    latest_md = bench_root / "latest.md"
    hist_json = _uniquify_history_path(history_dir / f"{base_name}.json")
    hist_md = _uniquify_history_path(history_dir / f"{base_name}.md")
    md_text = render_benchmark_diff_markdown(payload)
    try:
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(bench_root), suffix=".json.tmp") as fh:
            fh.write(text)
            tmp_json = Path(fh.name)
        tmp_json.replace(latest_json)
        hist_json.write_text(text, encoding="utf-8")
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(bench_root), suffix=".md.tmp") as fh:
            fh.write(md_text)
            tmp_md = Path(fh.name)
        tmp_md.replace(latest_md)
        hist_md.write_text(md_text, encoding="utf-8")
    except OSError:
        # Persistence failures must not break main workflow.
        pass
    return {
        "run_id": effective_run_id,
        "latest_json": str(latest_json),
        "latest_md": str(latest_md),
        "history_json": str(hist_json),
        "history_md": str(hist_md),
    }


def load_latest_benchmark(project_root: Path) -> Optional[Dict[str, Any]]:
    bench_root = (project_root / ".cryoem_agent" / "benchmarks").resolve()
    latest = bench_root / "latest.json"
    if not latest.is_file():
        return None
    try:
        return json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

