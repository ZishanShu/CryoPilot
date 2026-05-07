"""Batch runner for the EMPIAR closed-loop picker-to-reconstruction pipeline.

This script reads a manifest CSV describing one or more datasets and, for
each dataset, dispatches to the single-dataset pipeline
``scripts/run_empiar10017_closed_loop_picker_to_reconstruction.py`` in the
requested mode (``plan-only`` / ``smoke`` / ``full`` / ``reuse``).

Design goals
------------
* No duplication of the single-dataset pipeline logic.
* Auditable per-dataset outputs:
    out_root/<dataset_id>/
        dataset_readiness.json
        dataset_plan.md
        run.log
        workflow_summary.json   (when pipeline ran)
        workflow_summary.md     (when pipeline ran)
* Aggregated batch outputs:
    out_root/batch_summary.csv
    out_root/batch_summary.md
    out_root/batch_protocol_report.md
* Resumable (``--skip-existing``) and tolerant (``--continue-on-error``).

Scientific wording
------------------
This is **batch readiness** evidence for closed-loop picker fallback and
downstream-gated workflow execution.  It does NOT claim final autonomous
structure determination.  Map QC is required before any map-to-model
real-run is allowed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


PIPELINE_MODULE = "scripts.run_empiar10017_closed_loop_picker_to_reconstruction"

VALID_MODES = ("plan-only", "smoke", "full", "reuse")

CSV_FIELDS = [
    "dataset_id", "dataset_type", "mode", "input_status", "selected_action",
    "pipeline_status", "stage_reached", "relion_log_f1", "qc_passed",
    "predicted_particles", "extracted_particles",
    "reconstruction_map_exists", "final_readiness", "runtime_sec", "error",
    "out_dir",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _exists(p: Optional[str]) -> bool:
    if not p:
        return False
    try:
        return Path(p).exists()
    except Exception:
        return False


def read_manifest(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cleaned = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            if not cleaned.get("dataset_id"):
                continue
            rows.append(cleaned)
    return rows


# ---------------------------------------------------------------------------
# Per-dataset planning
# ---------------------------------------------------------------------------


def validate_dataset_inputs(
    record: Dict[str, str],
    *,
    mode: str,
    relion_autopick_bin: Optional[Path],
    relion_bin_dir: Optional[Path],
) -> Dict[str, Any]:
    """Return readiness info: input_status, selected_action, blockers."""
    blockers: List[str] = []
    pairs_csv = record.get("pairs_csv") or ""
    reuse_root = record.get("reuse_upstream_root") or ""

    if mode in ("smoke", "full"):
        if not _exists(pairs_csv):
            blockers.append(f"pairs_csv missing: {pairs_csv!r}")
        if relion_autopick_bin is not None and not _exists(str(relion_autopick_bin)):
            blockers.append(f"relion_autopick_bin missing: {relion_autopick_bin}")
        if relion_bin_dir is not None and not _exists(str(relion_bin_dir)):
            blockers.append(f"relion_bin_dir missing: {relion_bin_dir}")

    if mode == "reuse":
        if not _exists(reuse_root):
            blockers.append(f"reuse_upstream_root missing: {reuse_root!r}")
        if relion_bin_dir is not None and not _exists(str(relion_bin_dir)):
            blockers.append(f"relion_bin_dir missing: {relion_bin_dir}")

    if mode == "plan-only":
        # Plan-only does not require RELION, but we still note missing inputs.
        pass

    if blockers:
        return {
            "input_status": "blocked",
            "blockers": blockers,
            "selected_action": "BLOCK_DATASET_INPUT_MISSING",
        }

    if mode == "plan-only":
        action = "PLAN_ONLY_DATASET_READINESS"
    elif mode == "smoke":
        action = "RUN_SMOKE_PIPELINE"
    elif mode == "full":
        action = "RUN_FULL_PIPELINE"
    elif mode == "reuse":
        action = "RUN_DOWNSTREAM_REUSE"
    else:
        action = "UNKNOWN_MODE"

    return {"input_status": "ok", "blockers": [], "selected_action": action}


def write_dataset_plan(
    ds_dir: Path,
    record: Dict[str, str],
    readiness: Dict[str, Any],
    args_dict: Dict[str, Any],
) -> None:
    payload = {
        "dataset_id": record.get("dataset_id"),
        "dataset_type": record.get("dataset_type"),
        "mode": args_dict.get("mode"),
        "input_status": readiness["input_status"],
        "selected_action": readiness["selected_action"],
        "blockers": readiness["blockers"],
        "record": record,
        "args": args_dict,
        "scientific_caveat": (
            "Batch readiness for closed-loop picker fallback and "
            "downstream-gated workflow.  Map QC is required before "
            "map-to-model real-run.  Not final autonomous structure "
            "determination."
        ),
    }
    _write_json(ds_dir / "dataset_readiness.json", payload)
    md = [
        f"# Dataset {record.get('dataset_id')} – batch readiness",
        "",
        f"- mode: {args_dict.get('mode')}",
        f"- dataset_type: {record.get('dataset_type')}",
        f"- input_status: {readiness['input_status']}",
        f"- selected_action: {readiness['selected_action']}",
        "",
    ]
    if readiness["blockers"]:
        md.append("## Blockers")
        md.append("")
        for b in readiness["blockers"]:
            md.append(f"- {b}")
        md.append("")
    md.append("## Record")
    md.append("")
    for k, v in record.items():
        md.append(f"- {k}: {v}")
    md.append("")
    md.append("> Towards autonomous cryo-EM structure determination. "
              "Not final autonomous structure determination.")
    _write_text(ds_dir / "dataset_plan.md", "\n".join(md) + "\n")


# ---------------------------------------------------------------------------
# Per-dataset pipeline invocation
# ---------------------------------------------------------------------------


def build_pipeline_argv(
    record: Dict[str, str],
    ds_dir: Path,
    args: argparse.Namespace,
) -> List[str]:
    """Build the argv for the single-dataset pipeline (without 'python -m ...')."""
    mode = args.mode
    out_root = ds_dir.resolve()
    argv: List[str] = ["--out-root", str(out_root)]

    # Per-dataset diameter / pixel-size overrides if present in manifest.
    diameter = record.get("particle_diameter_angstrom") or args.particle_diameter_angstrom
    pixel = record.get("pixel_size_angstrom") or args.pixel_size_angstrom
    argv += [
        "--particle-diameter-angstrom", str(diameter),
        "--pixel-size-angstrom", str(pixel),
        "--qc-threshold-f1", str(args.qc_threshold_f1),
        "--extract-size", str(args.extract_size),
        "--bg-radius", str(args.bg_radius),
        "--downstream-mode", args.downstream_mode,
        "--class2d-mode", args.class2d_mode,
        "--refine3d-mode", args.refine3d_mode,
        "--postprocess-mode", args.postprocess_mode,
        "--map2pdb-mode", args.map2pdb_mode,
    ]
    if args.relion_gpu_devices:
        argv += ["--relion-gpu-devices", str(args.relion_gpu_devices)]
    if args.relion_autopick_bin:
        argv += ["--relion-autopick-bin", str(args.relion_autopick_bin)]
    if args.relion_bin_dir:
        argv += ["--relion-bin-dir", str(args.relion_bin_dir)]
    sequence = record.get("sequence")
    if sequence:
        argv += ["--sequence", sequence]

    if mode in ("smoke", "full"):
        argv += ["--pairs-csv", record["pairs_csv"]]
        if mode == "smoke":
            argv += ["--limit", str(args.limit)]
        else:
            # Full mode: respect a per-dataset limit if specified, otherwise
            # leave the default in place.
            ds_limit = record.get("limit")
            if ds_limit:
                argv += ["--limit", str(ds_limit)]
    elif mode == "reuse":
        argv += ["--reuse-upstream-root", record["reuse_upstream_root"]]
    if args.dry_run:
        argv += ["--dry-run"]
    return argv


def default_pipeline_runner(
    argv: List[str],
    *,
    log_path: Path,
    timeout_sec: int,
    cwd: Optional[Path] = None,
) -> Dict[str, Any]:
    """Invoke the single-dataset pipeline as a subprocess.

    Streams stdout+stderr to ``log_path`` and returns a result dict.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", PIPELINE_MODULE, *argv]
    t0 = time.monotonic()
    timed_out = False
    returncode: Optional[int]
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write("$ " + shlex.join(cmd) + "\n\n")
        fh.flush()
        try:
            proc = subprocess.run(
                cmd, stdout=fh, stderr=subprocess.STDOUT, text=True,
                timeout=timeout_sec, cwd=str(cwd) if cwd else None,
            )
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = -1
            fh.write("\n[batch] TIMEOUT\n")
    return {
        "cmd": cmd,
        "returncode": returncode,
        "runtime_sec": round(time.monotonic() - t0, 3),
        "log": str(log_path),
        "timed_out": timed_out,
    }


def collect_dataset_result(
    ds_dir: Path,
    *,
    record: Dict[str, str],
    mode: str,
    readiness: Dict[str, Any],
    pipeline_status: str,
    runtime_sec: float,
    error: Optional[str],
) -> Dict[str, Any]:
    """Read workflow_summary.json (if present) and synthesise a row."""
    summary_path = ds_dir / "workflow_summary.json"
    summary: Dict[str, Any] = {}
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            error = (error or "") + f"; summary_parse_error: {exc}"

    relion = (summary.get("relion_log_rerun") or {})
    extraction = (summary.get("extraction") or {})
    reconstruction = (summary.get("reconstruction") or {})
    stage_status = summary.get("stage_status") or {}
    stage_reached = "none"
    for k in sorted(stage_status.keys()):
        s = stage_status[k]
        if s and s != "blocked":
            stage_reached = k

    return {
        "dataset_id": record.get("dataset_id"),
        "dataset_type": record.get("dataset_type"),
        "mode": mode,
        "input_status": readiness["input_status"],
        "selected_action": readiness["selected_action"],
        "pipeline_status": pipeline_status,
        "stage_reached": stage_reached,
        "relion_log_f1": relion.get("f1"),
        "qc_passed": relion.get("qc_passed"),
        "predicted_particles": relion.get("predicted_count"),
        "extracted_particles": extraction.get("n_particles"),
        "reconstruction_map_exists": reconstruction.get("map_exists"),
        "final_readiness": summary.get("final_readiness"),
        "runtime_sec": runtime_sec,
        "error": error or "",
        "out_dir": str(ds_dir),
    }


def run_one_dataset(
    record: Dict[str, str],
    *,
    args: argparse.Namespace,
    out_root: Path,
    pipeline_runner: Callable[..., Dict[str, Any]],
) -> Dict[str, Any]:
    """Plan + (optionally) run the single-dataset pipeline for one record."""
    ds_id = record["dataset_id"]
    ds_dir = out_root / ds_id
    ds_dir.mkdir(parents=True, exist_ok=True)

    readiness = validate_dataset_inputs(
        record,
        mode=args.mode,
        relion_autopick_bin=args.relion_autopick_bin,
        relion_bin_dir=args.relion_bin_dir,
    )
    args_dict = {
        "mode": args.mode,
        "limit": args.limit,
        "qc_threshold_f1": args.qc_threshold_f1,
        "particle_diameter_angstrom": args.particle_diameter_angstrom,
        "pixel_size_angstrom": args.pixel_size_angstrom,
        "extract_size": args.extract_size,
        "bg_radius": args.bg_radius,
        "downstream_mode": args.downstream_mode,
        "class2d_mode": args.class2d_mode,
        "refine3d_mode": args.refine3d_mode,
        "postprocess_mode": args.postprocess_mode,
        "map2pdb_mode": args.map2pdb_mode,
        "relion_gpu_devices": args.relion_gpu_devices,
        "allow_full_real": args.allow_full_real,
        "dry_run": args.dry_run,
    }
    write_dataset_plan(ds_dir, record, readiness, args_dict)

    # Resume support.
    if args.skip_existing and (ds_dir / "workflow_summary.json").is_file():
        return collect_dataset_result(
            ds_dir, record=record, mode=args.mode, readiness=readiness,
            pipeline_status="skipped_existing", runtime_sec=0.0, error=None,
        )

    if readiness["input_status"] == "blocked":
        return collect_dataset_result(
            ds_dir, record=record, mode=args.mode, readiness=readiness,
            pipeline_status="blocked", runtime_sec=0.0,
            error="; ".join(readiness["blockers"]),
        )

    if args.mode == "plan-only":
        return collect_dataset_result(
            ds_dir, record=record, mode=args.mode, readiness=readiness,
            pipeline_status="plan_only", runtime_sec=0.0, error=None,
        )

    if args.mode == "full" and not args.allow_full_real:
        return collect_dataset_result(
            ds_dir, record=record, mode=args.mode, readiness=readiness,
            pipeline_status="blocked", runtime_sec=0.0,
            error="full mode requires --allow-full-real",
        )

    argv = build_pipeline_argv(record, ds_dir, args)
    log_path = ds_dir / "run.log"
    try:
        run_info = pipeline_runner(
            argv,
            log_path=log_path,
            timeout_sec=int(args.timeout_minutes * 60),
            cwd=Path(__file__).resolve().parent.parent,
        )
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return collect_dataset_result(
            ds_dir, record=record, mode=args.mode, readiness=readiness,
            pipeline_status="error", runtime_sec=0.0,
            error=f"pipeline_runner_exception: {exc}",
        )

    pipeline_status = "ok" if run_info.get("returncode") == 0 else (
        "timeout" if run_info.get("timed_out") else "error"
    )
    error_msg = None
    if pipeline_status != "ok":
        error_msg = f"returncode={run_info.get('returncode')}"

    return collect_dataset_result(
        ds_dir, record=record, mode=args.mode, readiness=readiness,
        pipeline_status=pipeline_status,
        runtime_sec=float(run_info.get("runtime_sec", 0.0)),
        error=error_msg,
    )


# ---------------------------------------------------------------------------
# Batch reporting
# ---------------------------------------------------------------------------


def write_batch_csv(out_root: Path, rows: List[Dict[str, Any]]) -> None:
    csv_path = out_root / "batch_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def write_batch_md(out_root: Path, rows: List[Dict[str, Any]],
                   args: argparse.Namespace) -> None:
    md = [
        "# Batch summary",
        "",
        f"- manifest: `{args.manifest}`",
        f"- out_root: `{out_root}`",
        f"- mode: `{args.mode}`",
        f"- datasets: {len(rows)}",
        "",
        "## Per-dataset",
        "",
        "| dataset_id | mode | input_status | pipeline_status | "
        "final_readiness | f1 | extracted | runtime_sec |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r.get('dataset_id')} | {r.get('mode')} | "
            f"{r.get('input_status')} | {r.get('pipeline_status')} | "
            f"{r.get('final_readiness')} | {r.get('relion_log_f1')} | "
            f"{r.get('extracted_particles')} | {r.get('runtime_sec')} |"
        )
    md.append("")
    md.append("## Scientific caveat")
    md.append("")
    md.append("Batch readiness for closed-loop picker fallback and "
              "downstream-gated workflow execution.  Map QC is required "
              "before map-to-model real-run.  Towards autonomous cryo-EM "
              "structure determination — not final autonomous structure "
              "determination.")
    _write_text(out_root / "batch_summary.md", "\n".join(md) + "\n")


def write_batch_protocol_report(out_root: Path, rows: List[Dict[str, Any]],
                                args: argparse.Namespace) -> None:
    counts: Dict[str, int] = {}
    for r in rows:
        s = r.get("pipeline_status", "unknown")
        counts[s] = counts.get(s, 0) + 1
    lines = [
        "# Batch protocol report",
        "",
        f"- manifest: `{args.manifest}`",
        f"- mode: `{args.mode}`",
        f"- max_parallel: {args.max_parallel}",
        f"- timeout_minutes: {args.timeout_minutes}",
        "",
        "## Status counts",
        "",
    ]
    for k, v in sorted(counts.items()):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("This report describes batch readiness for the closed-loop "
                 "picker fallback and downstream-gated workflow.  No claim "
                 "of final autonomous structure determination is made; "
                 "map QC is required before any map-to-model real-run.")
    _write_text(out_root / "batch_protocol_report.md", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Batch runner for the closed-loop EMPIAR pipeline."
    )
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--mode", choices=list(VALID_MODES), default="plan-only")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--qc-threshold-f1", type=float, default=0.2)
    ap.add_argument("--particle-diameter-angstrom", type=float, default=120.0)
    ap.add_argument("--pixel-size-angstrom", type=float, default=1.0)
    ap.add_argument("--extract-size", type=int, default=256)
    ap.add_argument("--bg-radius", type=int, default=100)
    ap.add_argument("--relion-autopick-bin", type=Path, default=None)
    ap.add_argument("--relion-bin-dir", type=Path, default=None)
    ap.add_argument("--downstream-mode", default="plan-only")
    ap.add_argument("--class2d-mode", default="plan-only")
    ap.add_argument("--refine3d-mode", default="plan-only")
    ap.add_argument("--postprocess-mode", default="plan-only")
    ap.add_argument("--map2pdb-mode", default="dry-run")
    ap.add_argument("--relion-gpu-devices", type=str, default=None)
    ap.add_argument("--max-parallel", type=int, default=1)
    ap.add_argument("--timeout-minutes", type=int, default=240)
    ap.add_argument("--skip-existing", action="store_true", default=False)
    ap.add_argument("--allow-full-real", action="store_true", default=False)
    ap.add_argument("--dry-run", action="store_true", default=False)
    ap.add_argument("--continue-on-error", action="store_true", default=True)
    ap.add_argument("--no-continue-on-error", dest="continue_on_error",
                    action="store_false")
    return ap


def run_batch(
    args: argparse.Namespace,
    *,
    pipeline_runner: Optional[Callable[..., Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Programmatic entry point — used by both the CLI and tests."""
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    runner = pipeline_runner or default_pipeline_runner

    records = read_manifest(args.manifest)
    print(f"[batch] {len(records)} datasets from {args.manifest}", flush=True)

    rows: List[Dict[str, Any]] = []
    fatal: Optional[BaseException] = None

    if args.max_parallel <= 1:
        for record in records:
            try:
                rows.append(run_one_dataset(
                    record, args=args, out_root=out_root, pipeline_runner=runner,
                ))
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                rows.append({
                    "dataset_id": record.get("dataset_id"),
                    "dataset_type": record.get("dataset_type"),
                    "mode": args.mode,
                    "input_status": "error",
                    "selected_action": "BATCH_RUN_EXCEPTION",
                    "pipeline_status": "error",
                    "stage_reached": "none",
                    "error": f"{type(exc).__name__}: {exc}",
                    "out_dir": str(out_root / (record.get("dataset_id") or "")),
                    "runtime_sec": 0.0,
                })
                if not args.continue_on_error:
                    fatal = exc
                    break
    else:
        with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
            futures = {
                pool.submit(
                    run_one_dataset, r, args=args, out_root=out_root,
                    pipeline_runner=runner,
                ): r
                for r in records
            }
            for fut in as_completed(futures):
                rec = futures[fut]
                try:
                    rows.append(fut.result())
                except Exception as exc:  # noqa: BLE001
                    traceback.print_exc()
                    rows.append({
                        "dataset_id": rec.get("dataset_id"),
                        "dataset_type": rec.get("dataset_type"),
                        "mode": args.mode,
                        "input_status": "error",
                        "selected_action": "BATCH_RUN_EXCEPTION",
                        "pipeline_status": "error",
                        "stage_reached": "none",
                        "error": f"{type(exc).__name__}: {exc}",
                        "out_dir": str(out_root / (rec.get("dataset_id") or "")),
                        "runtime_sec": 0.0,
                    })
                    if not args.continue_on_error:
                        fatal = exc

    # Stable ordering by manifest input.
    by_id = {r["dataset_id"]: r for r in rows if r.get("dataset_id")}
    ordered = [by_id[r["dataset_id"]] for r in records if r.get("dataset_id") in by_id]

    write_batch_csv(out_root, ordered)
    write_batch_md(out_root, ordered, args)
    write_batch_protocol_report(out_root, ordered, args)

    if fatal is not None:
        raise fatal
    return ordered


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    rows = run_batch(args)
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r.get("pipeline_status", "unknown")] = \
            counts.get(r.get("pipeline_status", "unknown"), 0) + 1
    print(json.dumps({
        "out_root": str(args.out_root.resolve()),
        "n_datasets": len(rows),
        "status_counts": counts,
        "batch_summary_csv": str(args.out_root / "batch_summary.csv"),
        "batch_summary_md": str(args.out_root / "batch_summary.md"),
        "batch_protocol_report_md": str(args.out_root / "batch_protocol_report.md"),
    }, indent=2))


if __name__ == "__main__":
    main()
