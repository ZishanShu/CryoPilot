"""Tests for scripts/run_batch_dataset_protocol.py."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import patch

import pytest

_ROOT_DIR = Path(__file__).parent.parent
_SCRIPTS_DIR = _ROOT_DIR / "scripts"
for _p in (_ROOT_DIR, _SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from run_batch_dataset_protocol import (  # noqa: E402
    _build_arg_parser,
    run_batch,
    validate_dataset_inputs,
    build_pipeline_argv,
    read_manifest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_manifest(path: Path, rows: List[Dict[str, str]]) -> Path:
    fields = [
        "dataset_id", "dataset_type", "pairs_csv", "reuse_upstream_root",
        "sequence", "particle_diameter_angstrom", "pixel_size_angstrom",
        "notes", "limit",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})
    return path


def _make_pairs_csv(tmp_path: Path, name: str = "pairs.csv") -> Path:
    p = tmp_path / name
    p.write_text("micrograph_id,micrograph_path,gt_coord_path\n", encoding="utf-8")
    return p


def _make_argv(out_root: Path, manifest: Path, mode: str = "plan-only",
               extra: Optional[List[str]] = None) -> List[str]:
    base = ["--manifest", str(manifest), "--out-root", str(out_root),
            "--mode", mode]
    if extra:
        base += extra
    return base


def _parse(argv: List[str]):
    return _build_arg_parser().parse_args(argv)


def _spy_runner_factory(write_summary: bool = True, returncode: int = 0,
                        f1: float = 0.65, n_particles: int = 84879):
    """Return a (calls, runner) pair where runner mimics the pipeline."""
    calls: List[Dict[str, Any]] = []

    def runner(argv: List[str], *, log_path: Path, timeout_sec: int,
               cwd: Optional[Path] = None) -> Dict[str, Any]:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("$ pipeline " + " ".join(argv) + "\n", encoding="utf-8")
        out_root_idx = argv.index("--out-root")
        ds_out = Path(argv[out_root_idx + 1])
        if write_summary:
            ds_out.mkdir(parents=True, exist_ok=True)
            (ds_out / "workflow_summary.json").write_text(json.dumps({
                "stage_status": {
                    "01_topaz_evidence": "ok", "02_picker_fallback": "ok",
                    "03_relion_log_rerun": "ok", "04_extract_project": "ok",
                    "05_extraction": "ok", "06_reconstruction": "ok",
                    "07_class2d": "planned", "08_refine3d": "planned",
                    "09_postprocess": "stop_gated",
                    "10_map2pdb_routing": "ok",
                    "11_model_validation": "skipped",
                },
                "relion_log_rerun": {"f1": f1, "qc_passed": True,
                                     "predicted_count": n_particles},
                "extraction": {"n_particles": n_particles},
                "reconstruction": {"map_exists": True,
                                   "map_path": "/tmp/x.mrc"},
                "final_readiness": "MAP_QC_REQUIRED",
            }), encoding="utf-8")
            (ds_out / "workflow_summary.md").write_text("# summary\n",
                                                       encoding="utf-8")
        calls.append({"argv": list(argv), "out_dir": str(ds_out)})
        return {"cmd": ["python", "-m", "fake"], "returncode": returncode,
                "runtime_sec": 0.01, "log": str(log_path), "timed_out": False}

    return calls, runner


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidateInputs:
    def test_smoke_blocks_when_pairs_missing(self, tmp_path: Path) -> None:
        record = {"dataset_id": "ds1", "pairs_csv": str(tmp_path / "missing.csv")}
        r = validate_dataset_inputs(record, mode="smoke",
                                    relion_autopick_bin=None,
                                    relion_bin_dir=None)
        assert r["input_status"] == "blocked"
        assert r["selected_action"] == "BLOCK_DATASET_INPUT_MISSING"
        assert any("pairs_csv missing" in b for b in r["blockers"])

    def test_reuse_blocks_when_root_missing(self, tmp_path: Path) -> None:
        record = {"dataset_id": "ds1",
                  "reuse_upstream_root": str(tmp_path / "missing_root")}
        r = validate_dataset_inputs(record, mode="reuse",
                                    relion_autopick_bin=None,
                                    relion_bin_dir=None)
        assert r["input_status"] == "blocked"

    def test_smoke_ok(self, tmp_path: Path) -> None:
        pairs = _make_pairs_csv(tmp_path)
        r = validate_dataset_inputs({"dataset_id": "ds1",
                                     "pairs_csv": str(pairs)},
                                    mode="smoke",
                                    relion_autopick_bin=None,
                                    relion_bin_dir=None)
        assert r["input_status"] == "ok"
        assert r["selected_action"] == "RUN_SMOKE_PIPELINE"


# ---------------------------------------------------------------------------
# CLI / argv tests
# ---------------------------------------------------------------------------


class TestBuildPipelineArgv:
    def test_smoke_argv_has_pairs_csv_and_limit(self, tmp_path: Path) -> None:
        manifest = _write_manifest(tmp_path / "m.csv", [
            {"dataset_id": "ds1", "pairs_csv": str(tmp_path / "p.csv")},
        ])
        args = _parse(_make_argv(tmp_path / "out", manifest, "smoke",
                                 ["--limit", "5"]))
        argv = build_pipeline_argv({"dataset_id": "ds1",
                                    "pairs_csv": str(tmp_path / "p.csv")},
                                   tmp_path / "ds1", args)
        assert "--pairs-csv" in argv
        assert "--limit" in argv and argv[argv.index("--limit") + 1] == "5"
        assert "--reuse-upstream-root" not in argv

    def test_reuse_argv_has_reuse_root(self, tmp_path: Path) -> None:
        manifest = _write_manifest(tmp_path / "m.csv", [{"dataset_id": "ds1"}])
        args = _parse(_make_argv(tmp_path / "out", manifest, "reuse"))
        argv = build_pipeline_argv({"dataset_id": "ds1",
                                    "reuse_upstream_root": "/upstream/x"},
                                   tmp_path / "ds1", args)
        assert "--reuse-upstream-root" in argv
        assert argv[argv.index("--reuse-upstream-root") + 1] == "/upstream/x"
        assert "--pairs-csv" not in argv


# ---------------------------------------------------------------------------
# End-to-end batch tests with mocked pipeline runner
# ---------------------------------------------------------------------------


class TestBatchRun:
    def test_plan_only_writes_readiness_and_plan(self, tmp_path: Path) -> None:
        manifest = _write_manifest(tmp_path / "m.csv", [
            {"dataset_id": "ds1", "pairs_csv": str(tmp_path / "p.csv")},
            {"dataset_id": "ds2", "pairs_csv": str(tmp_path / "q.csv")},
        ])
        out_root = tmp_path / "out"
        args = _parse(_make_argv(out_root, manifest, "plan-only"))
        rows = run_batch(args)
        assert len(rows) == 2
        for r in rows:
            ds_dir = out_root / r["dataset_id"]
            assert (ds_dir / "dataset_readiness.json").exists()
            assert (ds_dir / "dataset_plan.md").exists()
        assert (out_root / "batch_summary.csv").exists()
        assert (out_root / "batch_summary.md").exists()
        assert (out_root / "batch_protocol_report.md").exists()
        text = (out_root / "batch_summary.md").read_text()
        assert "Batch readiness" in text or "batch readiness" in text.lower()
        assert "not final autonomous structure determination" in text.lower()

    def test_smoke_invokes_pipeline_with_pairs_csv_and_limit(
        self, tmp_path: Path
    ) -> None:
        pairs = _make_pairs_csv(tmp_path)
        manifest = _write_manifest(tmp_path / "m.csv", [
            {"dataset_id": "ds1", "pairs_csv": str(pairs)},
        ])
        out_root = tmp_path / "out"
        calls, runner = _spy_runner_factory()
        args = _parse(_make_argv(out_root, manifest, "smoke", ["--limit", "3"]))
        rows = run_batch(args, pipeline_runner=runner)
        assert len(calls) == 1
        argv = calls[0]["argv"]
        assert "--pairs-csv" in argv
        assert argv[argv.index("--pairs-csv") + 1] == str(pairs)
        assert "--limit" in argv
        assert argv[argv.index("--limit") + 1] == "3"
        assert rows[0]["pipeline_status"] == "ok"
        assert rows[0]["relion_log_f1"] == 0.65
        assert rows[0]["extracted_particles"] == 84879
        assert rows[0]["final_readiness"] == "MAP_QC_REQUIRED"

    def test_reuse_invokes_pipeline_with_reuse_upstream_root(
        self, tmp_path: Path
    ) -> None:
        upstream = tmp_path / "upstream"
        upstream.mkdir()
        (upstream / "workflow_summary.json").write_text("{}", encoding="utf-8")
        manifest = _write_manifest(tmp_path / "m.csv", [
            {"dataset_id": "ds1", "reuse_upstream_root": str(upstream)},
        ])
        out_root = tmp_path / "out"
        calls, runner = _spy_runner_factory()
        args = _parse(_make_argv(out_root, manifest, "reuse"))
        run_batch(args, pipeline_runner=runner)
        assert "--reuse-upstream-root" in calls[0]["argv"]
        assert calls[0]["argv"][calls[0]["argv"].index(
            "--reuse-upstream-root") + 1] == str(upstream)

    def test_full_requires_allow_full_real(self, tmp_path: Path) -> None:
        pairs = _make_pairs_csv(tmp_path)
        manifest = _write_manifest(tmp_path / "m.csv", [
            {"dataset_id": "ds1", "pairs_csv": str(pairs)},
        ])
        out_root = tmp_path / "out"
        calls, runner = _spy_runner_factory()
        # No --allow-full-real → must be blocked.
        args = _parse(_make_argv(out_root, manifest, "full"))
        rows = run_batch(args, pipeline_runner=runner)
        assert rows[0]["pipeline_status"] == "blocked"
        assert calls == []
        # With --allow-full-real → must invoke the pipeline.
        out_root2 = tmp_path / "out2"
        args2 = _parse(_make_argv(out_root2, manifest, "full",
                                  ["--allow-full-real"]))
        rows2 = run_batch(args2, pipeline_runner=runner)
        assert rows2[0]["pipeline_status"] == "ok"
        assert len(calls) == 1

    def test_skip_existing_skips_when_summary_present(
        self, tmp_path: Path
    ) -> None:
        pairs = _make_pairs_csv(tmp_path)
        manifest = _write_manifest(tmp_path / "m.csv", [
            {"dataset_id": "ds1", "pairs_csv": str(pairs)},
        ])
        out_root = tmp_path / "out"
        ds_dir = out_root / "ds1"
        ds_dir.mkdir(parents=True)
        (ds_dir / "workflow_summary.json").write_text(json.dumps({
            "stage_status": {"01_topaz_evidence": "ok"},
            "relion_log_rerun": {"f1": 0.5},
            "extraction": {"n_particles": 100},
            "reconstruction": {"map_exists": True},
            "final_readiness": "MAP_QC_REQUIRED",
        }), encoding="utf-8")
        calls, runner = _spy_runner_factory()
        args = _parse(_make_argv(out_root, manifest, "smoke",
                                 ["--skip-existing"]))
        rows = run_batch(args, pipeline_runner=runner)
        assert rows[0]["pipeline_status"] == "skipped_existing"
        assert calls == [], "no pipeline call when skipping existing"

    def test_continue_on_error_records_failure(self, tmp_path: Path) -> None:
        pairs = _make_pairs_csv(tmp_path)
        manifest = _write_manifest(tmp_path / "m.csv", [
            {"dataset_id": "ds_fail", "pairs_csv": str(pairs)},
            {"dataset_id": "ds_ok", "pairs_csv": str(pairs)},
        ])
        out_root = tmp_path / "out"

        def runner(argv: List[str], *, log_path: Path, timeout_sec: int,
                   cwd: Optional[Path] = None) -> Dict[str, Any]:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("boom\n", encoding="utf-8")
            ds_id = Path(argv[argv.index("--out-root") + 1]).name
            if ds_id == "ds_fail":
                return {"cmd": ["x"], "returncode": 1, "runtime_sec": 0.01,
                        "log": str(log_path), "timed_out": False}
            ds_out = Path(argv[argv.index("--out-root") + 1])
            (ds_out / "workflow_summary.json").write_text(json.dumps({
                "relion_log_rerun": {"f1": 0.7},
                "extraction": {"n_particles": 10},
                "reconstruction": {"map_exists": True},
                "stage_status": {}, "final_readiness": "MAP_QC_REQUIRED",
            }), encoding="utf-8")
            return {"cmd": ["x"], "returncode": 0, "runtime_sec": 0.01,
                    "log": str(log_path), "timed_out": False}

        args = _parse(_make_argv(out_root, manifest, "smoke"))
        rows = run_batch(args, pipeline_runner=runner)
        assert len(rows) == 2
        statuses = {r["dataset_id"]: r["pipeline_status"] for r in rows}
        assert statuses["ds_fail"] == "error"
        assert statuses["ds_ok"] == "ok"
        # Batch outputs still written despite the failure.
        assert (out_root / "batch_summary.csv").exists()
        with (out_root / "batch_summary.csv").open() as fh:
            csv_rows = list(csv.DictReader(fh))
        assert len(csv_rows) == 2

    def test_missing_pairs_csv_is_blocked(self, tmp_path: Path) -> None:
        manifest = _write_manifest(tmp_path / "m.csv", [
            {"dataset_id": "ds1",
             "pairs_csv": str(tmp_path / "does_not_exist.csv")},
        ])
        out_root = tmp_path / "out"
        calls, runner = _spy_runner_factory()
        args = _parse(_make_argv(out_root, manifest, "smoke"))
        rows = run_batch(args, pipeline_runner=runner)
        assert rows[0]["pipeline_status"] == "blocked"
        assert calls == []

    def test_batch_outputs_written(self, tmp_path: Path) -> None:
        pairs = _make_pairs_csv(tmp_path)
        manifest = _write_manifest(tmp_path / "m.csv", [
            {"dataset_id": "ds1", "pairs_csv": str(pairs)},
        ])
        out_root = tmp_path / "out"
        _, runner = _spy_runner_factory()
        args = _parse(_make_argv(out_root, manifest, "smoke"))
        run_batch(args, pipeline_runner=runner)
        assert (out_root / "batch_summary.csv").exists()
        assert (out_root / "batch_summary.md").exists()
        assert (out_root / "batch_protocol_report.md").exists()
        protocol = (out_root / "batch_protocol_report.md").read_text().lower()
        assert "no claim of final autonomous structure determination" in protocol
