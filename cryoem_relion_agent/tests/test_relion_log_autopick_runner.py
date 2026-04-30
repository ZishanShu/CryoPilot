"""Tests for the RELION LoG autopick runner and its integration with picker_fallback.

All tests are offline:
- No real RELION binary or MRC/MRCS files are required.
- Command execution is mocked via RelionLogAutopickConfig.command_runner.
- F1 evaluation uses synthetic coord files.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Tuple

import pytest

from agent.relion_log_autopick_runner import (
    RelionLogAutopickConfig,
    _build_autopick_cmd,
    _parse_relion_coord_star,
    _read_pairs_csv,
    _write_coord_star,
    _write_micrographs_star,
    _write_simple_coord,
    run_relion_log_autopick,
)
from agent.picker_fallback import (
    BackendPaths,
    FINAL_ACTION_RUN,
    FINAL_ACTION_STOP,
    run_picker_fallback_stage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("", encoding="utf-8")
    return p


def _make_gt_coord(path: Path, coords: List[Tuple[float, float]]) -> Path:
    """Write a simple x y ground-truth coordinate file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(f"{x:.2f} {y:.2f}" for x, y in coords) + "\n", encoding="utf-8"
    )
    return path


def _make_pairs_csv(
    path: Path,
    pairs: List[Tuple[str, str]],
) -> Path:
    """Write pairs CSV with (micrograph_path, gt_coord_path) rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["micrograph_path", "gt_coord_path"])
        writer.writeheader()
        for mic, gt in pairs:
            writer.writerow({"micrograph_path": mic, "gt_coord_path": gt})
    return path


def _mock_runner_with_picks(
    star_dir: Path,
    picks_per_mic: dict,
) -> object:
    """Return a command_runner callable that writes autopick STAR files."""

    def runner(cmd: List[str], cwd: Path) -> Tuple[int, str, str]:
        # Extract --odir from cmd
        odir = None
        for i, tok in enumerate(cmd):
            if tok == "--odir" and i + 1 < len(cmd):
                odir = Path(cmd[i + 1])
                break
        if odir is None:
            odir = star_dir
        odir.mkdir(parents=True, exist_ok=True)
        for stem, coords in picks_per_mic.items():
            _write_coord_star(odir / f"{stem}_autopick.star", coords)
        return 0, "ok", ""

    return runner


def _make_relion_paths(tmp_path: Path) -> BackendPaths:
    relion = _touch(tmp_path / "bin" / "relion_autopick")
    return BackendPaths(relion_autopick=relion, allow_system_path=False)


# ---------------------------------------------------------------------------
# Unit tests: STAR writer / parser round-trip
# ---------------------------------------------------------------------------


def test_write_and_parse_coord_star_round_trip(tmp_path):
    coords = [(100.0, 200.0), (300.0, 400.0), (500.0, 600.0)]
    star_path = tmp_path / "test_autopick.star"
    _write_coord_star(star_path, coords)
    parsed = _parse_relion_coord_star(star_path)
    assert len(parsed) == len(coords)
    for (ex, ey), (px, py) in zip(coords, parsed):
        assert abs(ex - px) < 0.01
        assert abs(ey - py) < 0.01


def test_parse_coord_star_missing_file_returns_empty(tmp_path):
    result = _parse_relion_coord_star(tmp_path / "nonexistent.star")
    assert result == []


def test_write_micrographs_star_contains_optics_and_mics(tmp_path):
    star_path = tmp_path / "micrographs.star"
    _write_micrographs_star(
        star_path,
        ["Micrographs/mic001.mrc", "Micrographs/mic002.mrc"],
    )
    text = star_path.read_text()
    assert "data_optics" in text
    assert "data_micrographs" in text
    assert "Micrographs/mic001.mrc" in text
    assert "Micrographs/mic002.mrc" in text
    assert "_rlnMicrographPixelSize" in text
    assert "_rlnVoltage" in text


def test_build_autopick_cmd_contains_log_flags(tmp_path):
    cmd = _build_autopick_cmd(
        Path("relion_autopick"),
        tmp_path / "micrographs.star",
        tmp_path,
        diameter_angstrom=120.0,
        pixel_size=1.0,
    )
    assert "--LoG" in cmd
    assert "--LoG_diam_min" in cmd
    assert "--LoG_diam_max" in cmd
    assert "--odir" in cmd


# ---------------------------------------------------------------------------
# Unit tests: dry-run (no real RELION, no MRC)
# ---------------------------------------------------------------------------


def test_dry_run_returns_zero_predictions_and_f1_zero(tmp_path):
    gt_path = _make_gt_coord(tmp_path / "gt" / "mic001.coord", [(100.0, 200.0)] * 10)
    pairs_csv = _make_pairs_csv(
        tmp_path / "pairs.csv",
        [(str(tmp_path / "Micrographs" / "mic001.mrc"), str(gt_path))],
    )
    cfg = RelionLogAutopickConfig(
        relion_autopick_bin=tmp_path / "relion_autopick",
        output_root=tmp_path / "out",
        pairs_csv=pairs_csv,
        dry_run=True,
    )
    bundle = run_relion_log_autopick(cfg)
    assert bundle["mode"] == "dry_run"
    assert bundle["predicted_count"] == 0
    assert bundle["f1"] == 0.0
    assert bundle["returncode"] is None
    # command is built even in dry-run
    assert "--LoG" in bundle["command"]


def test_dry_run_writes_required_files(tmp_path):
    pairs_csv = _make_pairs_csv(
        tmp_path / "pairs.csv",
        [(str(tmp_path / "Micrographs" / "mic001.mrc"), "")],
    )
    cfg = RelionLogAutopickConfig(
        output_root=tmp_path / "out",
        pairs_csv=pairs_csv,
        dry_run=True,
    )
    run_relion_log_autopick(cfg)
    out = tmp_path / "out"
    assert (out / "micrographs.star").exists()
    assert (out / "autopick_command.txt").exists()
    assert (out / "all_predicted_coords.coord").exists()
    assert (out / "relion_log_autopick_result.json").exists()
    assert (out / "relion_log_autopick_result.md").exists()
    assert (out / "AutoPick" / "job001").is_dir()


def test_dry_run_limit_respected(tmp_path):
    pairs = [(str(tmp_path / f"mic{i:03d}.mrc"), "") for i in range(5)]
    pairs_csv = _make_pairs_csv(tmp_path / "pairs.csv", pairs)
    cfg = RelionLogAutopickConfig(
        output_root=tmp_path / "out",
        pairs_csv=pairs_csv,
        dry_run=True,
        limit=2,
    )
    bundle = run_relion_log_autopick(cfg)
    assert bundle["n_micrographs"] == 2


def test_dry_run_missing_pairs_csv_returns_error(tmp_path):
    cfg = RelionLogAutopickConfig(
        output_root=tmp_path / "out",
        pairs_csv=tmp_path / "nonexistent.csv",
        dry_run=True,
    )
    bundle = run_relion_log_autopick(cfg)
    assert bundle["error"] is not None
    assert bundle["f1"] is None


# ---------------------------------------------------------------------------
# Unit tests: real-run (mocked command execution)
# ---------------------------------------------------------------------------


def _setup_real_run(tmp_path: Path) -> tuple:
    """Create GT coords, pairs CSV, and return (cfg, expected_picks)."""
    gt_coords = [(100.0 + i * 50, 200.0 + i * 50) for i in range(5)]
    gt_path = _make_gt_coord(tmp_path / "gt" / "mic001.coord", gt_coords)

    pairs_csv = _make_pairs_csv(
        tmp_path / "pairs.csv",
        [(str(tmp_path / "Micrographs" / "mic001.mrc"), str(gt_path))],
    )

    # Mock runner: writes a coord STAR for mic001 that perfectly matches GT
    autopick_dir = tmp_path / "out" / "AutoPick" / "job001"
    mock_runner = _mock_runner_with_picks(autopick_dir, {"mic001": gt_coords})

    cfg = RelionLogAutopickConfig(
        relion_autopick_bin=tmp_path / "relion_autopick",
        output_root=tmp_path / "out",
        pairs_csv=pairs_csv,
        dry_run=False,
        match_radius_px=80.0,
        command_runner=mock_runner,
    )
    return cfg, gt_coords


def test_real_run_mocked_perfect_match_gives_f1_1(tmp_path):
    cfg, gt_coords = _setup_real_run(tmp_path)
    bundle = run_relion_log_autopick(cfg)
    assert bundle["mode"] == "real_run"
    assert bundle["returncode"] == 0
    assert bundle["predicted_count"] == len(gt_coords)
    assert bundle["f1"] is not None
    assert bundle["f1"] > 0.9


def test_real_run_mocked_no_picks_gives_f1_zero(tmp_path):
    gt_path = _make_gt_coord(tmp_path / "gt" / "mic001.coord", [(100.0, 200.0)] * 5)
    pairs_csv = _make_pairs_csv(
        tmp_path / "pairs.csv",
        [(str(tmp_path / "Micrographs" / "mic001.mrc"), str(gt_path))],
    )
    autopick_dir = tmp_path / "out" / "AutoPick" / "job001"
    # Runner writes an empty star (no picks)
    mock_runner = _mock_runner_with_picks(autopick_dir, {"mic001": []})

    cfg = RelionLogAutopickConfig(
        relion_autopick_bin=tmp_path / "relion_autopick",
        output_root=tmp_path / "out",
        pairs_csv=pairs_csv,
        dry_run=False,
        command_runner=mock_runner,
    )
    bundle = run_relion_log_autopick(cfg)
    assert bundle["f1"] == 0.0
    assert bundle["predicted_count"] == 0


def test_real_run_failed_command_records_error(tmp_path):
    pairs_csv = _make_pairs_csv(
        tmp_path / "pairs.csv",
        [(str(tmp_path / "Micrographs" / "mic001.mrc"), "")],
    )

    def failing_runner(cmd, cwd):
        return 1, "", "relion_autopick: error"

    cfg = RelionLogAutopickConfig(
        output_root=tmp_path / "out",
        pairs_csv=pairs_csv,
        dry_run=False,
        command_runner=failing_runner,
    )
    bundle = run_relion_log_autopick(cfg)
    assert bundle["returncode"] == 1
    assert bundle["predicted_count"] == 0


def test_real_run_command_runner_exception_does_not_crash(tmp_path):
    pairs_csv = _make_pairs_csv(
        tmp_path / "pairs.csv",
        [(str(tmp_path / "Micrographs" / "mic001.mrc"), "")],
    )

    def exploding_runner(cmd, cwd):
        raise RuntimeError("simulated subprocess failure")

    cfg = RelionLogAutopickConfig(
        output_root=tmp_path / "out",
        pairs_csv=pairs_csv,
        dry_run=False,
        command_runner=exploding_runner,
    )
    bundle = run_relion_log_autopick(cfg)
    assert bundle["error"] is not None
    assert bundle["predicted_count"] == 0


# ---------------------------------------------------------------------------
# Integration tests: picker_fallback with RELION runner
# ---------------------------------------------------------------------------


def _make_failed_evidence(qc_threshold: float = 0.2) -> dict:
    return {
        "selected_for_backend_execution": "topaz_t-4",
        "selection_metric": "picking_micro_f1",
        "selected_micro_f1": 0.031848,
        "qc_threshold_f1": qc_threshold,
        "picking_qc_passed": False,
        "final_action": "FALLBACK_TO_ALTERNATIVE_PICKER_OR_RETRAIN",
    }


def test_fallback_dry_run_relion_stop_preserved(tmp_path):
    """Dry-run yields 0 F1 → STOP is preserved when threshold not met."""
    gt_path = _make_gt_coord(
        tmp_path / "gt" / "mic001.coord", [(100.0, 200.0)] * 5
    )
    pairs_csv = _make_pairs_csv(
        tmp_path / "pairs.csv",
        [(str(tmp_path / "Micrographs" / "mic001.mrc"), str(gt_path))],
    )
    relion_bin = _touch(tmp_path / "bin" / "relion_autopick")
    paths = BackendPaths(relion_autopick=relion_bin, allow_system_path=False)

    bundle = run_picker_fallback_stage(
        _make_failed_evidence(),
        paths,
        tmp_path / "out",
        relion_run_config={
            "pairs_csv": pairs_csv,
            "dry_run": True,
        },
    )
    assert bundle["relion_rerun"] is not None
    assert bundle["relion_rerun"]["mode"] == "dry_run"
    assert bundle["final"]["final_action"] == FINAL_ACTION_STOP
    # dry-run produces no predictions → F1=0.0 (GT exists, predicted=0)
    assert bundle["final"]["relion_log_f1"] == 0.0


def test_fallback_real_run_mocked_below_threshold_stops(tmp_path):
    """Even with a real run, if F1 < threshold → STOP is preserved."""
    gt_coords = [(100.0, 200.0)]
    gt_path = _make_gt_coord(tmp_path / "gt" / "mic001.coord", gt_coords)
    pairs_csv = _make_pairs_csv(
        tmp_path / "pairs.csv",
        [(str(tmp_path / "Micrographs" / "mic001.mrc"), str(gt_path))],
    )
    autopick_dir = tmp_path / "out" / "reruns" / "relion_log_autopick" / "AutoPick" / "job001"
    # Zero picks → F1 = 0
    mock_runner = _mock_runner_with_picks(autopick_dir, {"mic001": []})
    relion_bin = _touch(tmp_path / "bin" / "relion_autopick")
    paths = BackendPaths(relion_autopick=relion_bin, allow_system_path=False)

    bundle = run_picker_fallback_stage(
        _make_failed_evidence(qc_threshold=0.2),
        paths,
        tmp_path / "out",
        relion_run_config={
            "pairs_csv": pairs_csv,
            "dry_run": False,
            "command_runner": mock_runner,
        },
    )
    assert bundle["final"]["final_action"] == FINAL_ACTION_STOP
    assert bundle["final"]["relion_log_f1"] == 0.0


def test_fallback_real_run_mocked_above_threshold_proceeds(tmp_path):
    """If RELION F1 >= threshold → final_action is RUN (proceed)."""
    gt_coords = [(100.0 + i * 60, 200.0 + i * 60) for i in range(10)]
    gt_path = _make_gt_coord(tmp_path / "gt" / "mic001.coord", gt_coords)
    pairs_csv = _make_pairs_csv(
        tmp_path / "pairs.csv",
        [(str(tmp_path / "Micrographs" / "mic001.mrc"), str(gt_path))],
    )
    autopick_dir = tmp_path / "out" / "reruns" / "relion_log_autopick" / "AutoPick" / "job001"
    mock_runner = _mock_runner_with_picks(autopick_dir, {"mic001": gt_coords})
    relion_bin = _touch(tmp_path / "bin" / "relion_autopick")
    paths = BackendPaths(relion_autopick=relion_bin, allow_system_path=False)

    bundle = run_picker_fallback_stage(
        _make_failed_evidence(qc_threshold=0.2),
        paths,
        tmp_path / "out",
        relion_run_config={
            "pairs_csv": pairs_csv,
            "dry_run": False,
            "command_runner": mock_runner,
            "match_radius_px": 80.0,
        },
    )
    assert bundle["final"]["final_action"] == FINAL_ACTION_RUN
    f1 = bundle["final"]["relion_log_f1"]
    assert f1 is not None and f1 >= 0.2


def test_fallback_all_artifacts_written_with_relion_rerun(tmp_path):
    """All required output artifacts are present after a dry-run fallback."""
    pairs_csv = _make_pairs_csv(
        tmp_path / "pairs.csv",
        [(str(tmp_path / "Micrographs" / "mic001.mrc"), "")],
    )
    relion_bin = _touch(tmp_path / "bin" / "relion_autopick")
    paths = BackendPaths(relion_autopick=relion_bin, allow_system_path=False)
    out = tmp_path / "out"

    run_picker_fallback_stage(
        _make_failed_evidence(),
        paths,
        out,
        relion_run_config={"pairs_csv": pairs_csv, "dry_run": True},
    )

    expected = [
        out / "retrieval" / "alternative_picker_candidates.json",
        out / "retrieval" / "alternative_picker_candidates.md",
        out / "selection" / "alternative_picker_rerun_plan.json",
        out / "selection" / "alternative_picker_rerun_plan.md",
        out / "comparison" / "picker_model_selection_comparison.csv",
        out / "comparison" / "picker_model_selection_comparison.md",
        out / "comparison" / "final_picker_selection.json",
        out / "comparison" / "final_picker_selection.md",
        out / "picker_model_selection_report.md",
        out / "reruns" / "relion_log_autopick" / "relion_log_autopick_result.json",
        out / "reruns" / "relion_log_autopick" / "relion_log_autopick_result.md",
        out / "reruns" / "relion_log_autopick" / "micrographs.star",
        out / "reruns" / "relion_log_autopick" / "autopick_command.txt",
    ]
    for p in expected:
        assert p.exists(), f"missing artifact: {p}"


def test_fallback_comparison_csv_contains_relion_f1(tmp_path):
    """After dry-run, relion_log_autopick row in comparison CSV has f1=0.0."""
    gt_path = _make_gt_coord(
        tmp_path / "gt" / "mic001.coord", [(100.0, 200.0)] * 5
    )
    pairs_csv = _make_pairs_csv(
        tmp_path / "pairs.csv",
        [(str(tmp_path / "Micrographs" / "mic001.mrc"), str(gt_path))],
    )
    relion_bin = _touch(tmp_path / "bin" / "relion_autopick")
    paths = BackendPaths(relion_autopick=relion_bin, allow_system_path=False)
    out = tmp_path / "out"

    run_picker_fallback_stage(
        _make_failed_evidence(),
        paths,
        out,
        relion_run_config={"pairs_csv": pairs_csv, "dry_run": True},
    )

    rows = list(csv.DictReader((out / "comparison" / "picker_model_selection_comparison.csv").open()))
    relion_row = next((r for r in rows if r["model"] == "relion_log_autopick"), None)
    assert relion_row is not None
    assert relion_row["best_f1"] == "0.0"


def test_no_relion_runner_called_when_no_relion_available(tmp_path):
    """When relion_log_autopick is unavailable, runner is not called and STOP is preserved."""
    bundle = run_picker_fallback_stage(
        _make_failed_evidence(),
        BackendPaths(allow_system_path=False),
        tmp_path / "out",
    )
    assert bundle["relion_rerun"] is None
    assert bundle["final"]["final_action"] == FINAL_ACTION_STOP


# ---------------------------------------------------------------------------
# Flexible pairs CSV parser
# ---------------------------------------------------------------------------


def _write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return path


def test_pairs_csv_parses_empiar_columns(tmp_path):
    """Real EMPIAR-10017 column layout must be parsed without dropping rows."""
    csv_path = _write_csv(
        tmp_path / "pairs.csv",
        [
            "micrograph_id", "mrc_path", "coord_path",
            "local_mrc_path", "local_coord_path",
            "n_gt_particles", "mrc_size_bytes", "coord_exists",
        ],
        [
            {
                "micrograph_id": "Falcon_2012_06_12-14_33_35_0",
                "mrc_path": "/data/empiar/Falcon_2012_06_12-14_33_35_0.mrc",
                "coord_path": "/data/empiar/Falcon_2012_06_12-14_33_35_0.coord",
                "local_mrc_path": "/runs/input/Falcon_2012_06_12-14_33_35_0.mrc",
                "local_coord_path": "/runs/input/Falcon_2012_06_12-14_33_35_0.coord",
                "n_gt_particles": "350",
                "mrc_size_bytes": "1234",
                "coord_exists": "True",
            }
        ],
    )
    pairs, diag = _read_pairs_csv(csv_path)
    assert diag["n_rows_raw"] == 1
    assert diag["n_rows_usable"] == 1
    assert len(pairs) == 1
    p = pairs[0]
    assert p["micrograph_id"] == "Falcon_2012_06_12-14_33_35_0"
    # local_mrc_path takes priority over mrc_path
    assert p["micrograph_path"].endswith("/runs/input/Falcon_2012_06_12-14_33_35_0.mrc")
    # local_coord_path takes priority over coord_path
    assert p["gt_coord_path"].endswith("/runs/input/Falcon_2012_06_12-14_33_35_0.coord")


def test_pairs_csv_parses_image_path_gt_path_aliases(tmp_path):
    """Alias-style CSV: image_path + gt_path must also parse."""
    csv_path = _write_csv(
        tmp_path / "pairs.csv",
        ["image_path", "gt_path"],
        [
            {"image_path": "/runs/input/foo.mrc", "gt_path": "/runs/input/foo.coord"},
            {"image_path": "/runs/input/bar.mrc", "gt_path": ""},
        ],
    )
    pairs, diag = _read_pairs_csv(csv_path)
    assert diag["n_rows_raw"] == 2
    assert diag["n_rows_usable"] == 2
    # micrograph_id is synthesised from path stem when absent
    assert pairs[0]["micrograph_id"] == "foo"
    assert pairs[0]["micrograph_path"] == "/runs/input/foo.mrc"
    assert pairs[0]["gt_coord_path"] == "/runs/input/foo.coord"
    # gt is optional
    assert pairs[1]["gt_coord_path"] == ""


def test_pairs_csv_diagnostics_json_is_written(tmp_path):
    """run_relion_log_autopick must persist pairs_csv_diagnostics.json."""
    csv_path = _write_csv(
        tmp_path / "pairs.csv",
        ["micrograph_id", "local_mrc_path", "local_coord_path"],
        [
            {
                "micrograph_id": "mic001",
                "local_mrc_path": str(tmp_path / "mic001.mrc"),
                "local_coord_path": "",
            }
        ],
    )
    cfg = RelionLogAutopickConfig(
        output_root=tmp_path / "out",
        pairs_csv=csv_path,
        dry_run=True,
    )
    bundle = run_relion_log_autopick(cfg)
    diag_path = tmp_path / "out" / "pairs_csv_diagnostics.json"
    assert diag_path.exists(), "diagnostics file not written"
    diag = json.loads(diag_path.read_text())
    assert diag["n_rows_raw"] == 1
    assert diag["n_rows_usable"] == 1
    assert "micrograph_id" in diag["fieldnames"]
    assert diag["first_raw_row"]["micrograph_id"] == "mic001"
    assert diag["first_usable_row"]["micrograph_id"] == "mic001"
    # Bundle should not error and the project layout should exist
    assert bundle["error"] is None
    assert bundle["n_micrographs"] == 1


def test_pairs_csv_empty_returns_empty_error(tmp_path):
    csv_path = _write_csv(tmp_path / "pairs.csv", ["micrograph_id", "mrc_path"], [])
    cfg = RelionLogAutopickConfig(
        output_root=tmp_path / "out",
        pairs_csv=csv_path,
        dry_run=True,
    )
    bundle = run_relion_log_autopick(cfg)
    assert bundle["error"] == "pairs_csv is empty"


def test_pairs_csv_no_usable_rows_distinct_error(tmp_path):
    """Rows present but no recognised path column → distinct error message."""
    csv_path = _write_csv(
        tmp_path / "pairs.csv",
        ["unrelated_col", "another"],
        [{"unrelated_col": "x", "another": "y"}],
    )
    cfg = RelionLogAutopickConfig(
        output_root=tmp_path / "out",
        pairs_csv=csv_path,
        dry_run=True,
    )
    bundle = run_relion_log_autopick(cfg)
    assert bundle["error"] == "pairs_csv has rows but no usable micrograph paths"
    diag = json.loads((tmp_path / "out" / "pairs_csv_diagnostics.json").read_text())
    assert diag["n_rows_raw"] == 1
    assert diag["n_rows_usable"] == 0
    assert diag["dropped_reasons"]
