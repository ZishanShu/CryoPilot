"""Tests for scripts/run_empiar10017_closed_loop_picker_to_reconstruction.py.

All RELION binary calls are mocked; no real MRC/MRCS data is required.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

# Ensure the cryoem_relion_agent root and scripts directories are importable.
_ROOT_DIR = Path(__file__).parent.parent
_SCRIPTS_DIR = _ROOT_DIR / "scripts"
for _p in (_ROOT_DIR, _SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from run_empiar10017_closed_loop_picker_to_reconstruction import (  # noqa: E402
    main,
    stage01_record_topaz_evidence,
    stage02_picker_fallback_selection,
    stage03_relion_log_rerun,
    stage04_materialise_extract_project,
    stage05_extract,
    stage06_reconstruct,
    stage07_write_summary,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _make_pairs_csv(tmp_path: Path, n: int = 3) -> Path:
    """Write a minimal pairs CSV with synthetic file paths."""
    pairs_csv = tmp_path / "pairs.csv"
    rows = [
        {
            "micrograph_id": f"mic_{i:04d}",
            "micrograph_path": str(tmp_path / "micrographs" / f"mic_{i:04d}.mrc"),
            "gt_coord_path": str(tmp_path / "gt_coords" / f"mic_{i:04d}.coord"),
        }
        for i in range(n)
    ]
    with pairs_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["micrograph_id", "micrograph_path", "gt_coord_path"])
        writer.writeheader()
        writer.writerows(rows)
    return pairs_csv


def _make_gt_coords(tmp_path: Path, n: int = 3) -> None:
    """Write synthetic GT coordinate files."""
    coords_dir = tmp_path / "gt_coords"
    coords_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        coord_file = coords_dir / f"mic_{i:04d}.coord"
        lines = [f"{100 + j * 10:.2f} {200 + j * 10:.2f}" for j in range(5)]
        coord_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_converted_coords(relion_log_out: Path, n: int = 3) -> None:
    """Write synthetic Stage03 converted_coords/*.coord files."""
    cdir = relion_log_out / "converted_coords"
    cdir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        coord = cdir / f"mic_{i:04d}.coord"
        coord.write_text(
            "image_name x_coord y_coord score\n"
            "mic 100.0 200.0 0.9\n"
            "mic 150.0 250.0 0.8\n",
            encoding="utf-8",
        )


def _make_autopick_stars(tmp_path: Path, autopick_dir: Path, n: int = 3) -> None:
    """Write synthetic per-micrograph autopick STAR files."""
    autopick_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        star = autopick_dir / f"mic_{i:04d}_autopick.star"
        star.write_text(
            "data_\n\nloop_\n"
            "_rlnCoordinateX #1\n"
            "_rlnCoordinateY #2\n"
            "_rlnAutopickFigureOfMerit #3\n"
            "100.00 200.00 0.500000\n"
            "150.00 250.00 0.600000\n",
            encoding="utf-8",
        )


def _mock_runner_success(cmd: List[str], cwd: Path) -> Tuple[int, str, str]:
    """Fake RELION command runner that succeeds with a plausible stdout."""
    return (0, "Total number of particles from 3 micrographs = 6\n", "")


def _mock_runner_fail(cmd: List[str], cwd: Path) -> Tuple[int, str, str]:
    return (1, "", "ERROR: something went wrong\n")


# ---------------------------------------------------------------------------
# Stage 01 tests
# ---------------------------------------------------------------------------


class TestStage01RecordTopazEvidence:
    def test_constructs_canonical_evidence_when_no_json(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        evidence = stage01_record_topaz_evidence(out, None, qc_threshold_f1=0.2)
        assert evidence["picking_qc_passed"] is False
        assert evidence["qc_threshold_f1"] == 0.2
        assert (out / "topaz_failed_evidence.json").exists()

    def test_loads_existing_evidence_json(self, tmp_path: Path) -> None:
        ev_path = tmp_path / "ev.json"
        ev_path.write_text(
            json.dumps({"picking_qc_passed": False, "selected_micro_f1": 0.05}),
            encoding="utf-8",
        )
        out = tmp_path / "out"
        evidence = stage01_record_topaz_evidence(out, ev_path, qc_threshold_f1=0.2)
        assert evidence["selected_micro_f1"] == 0.05
        assert (out / "topaz_failed_evidence.json").exists()

    def test_persists_json_file(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        stage01_record_topaz_evidence(out, None, qc_threshold_f1=0.2)
        payload = json.loads((out / "topaz_failed_evidence.json").read_text())
        assert "picking_qc_passed" in payload


# ---------------------------------------------------------------------------
# Stage 02 tests
# ---------------------------------------------------------------------------


class TestStage02PickerFallbackSelection:
    def _failed_evidence(self) -> Dict[str, Any]:
        return {
            "selected_for_backend_execution": "topaz_t-4",
            "selection_metric": "picking_micro_f1",
            "selected_micro_f1": 0.031848,
            "qc_threshold_f1": 0.2,
            "picking_qc_passed": False,
        }

    def test_selects_relion_when_binary_present(self, tmp_path: Path) -> None:
        # Create a fake relion_autopick binary.
        fake_bin = tmp_path / "relion_autopick"
        fake_bin.write_text("#!/bin/sh\n", encoding="utf-8")
        fake_bin.chmod(0o755)

        out = tmp_path / "selection"
        result = stage02_picker_fallback_selection(
            evidence=self._failed_evidence(),
            out_dir=out,
            relion_autopick_bin=fake_bin,
        )
        assert result.get("skipped") is not True
        selected = (result.get("final") or {}).get("selected")
        assert selected == "relion_log_autopick"

    def test_skips_when_qc_passed(self, tmp_path: Path) -> None:
        evidence = self._failed_evidence()
        evidence["picking_qc_passed"] = True
        out = tmp_path / "selection"
        result = stage02_picker_fallback_selection(
            evidence=evidence,
            out_dir=out,
            relion_autopick_bin=None,
        )
        assert result.get("skipped") is True

    def test_writes_artifacts(self, tmp_path: Path) -> None:
        fake_bin = tmp_path / "relion_autopick"
        fake_bin.write_text("#!/bin/sh\n", encoding="utf-8")
        fake_bin.chmod(0o755)

        out = tmp_path / "selection"
        stage02_picker_fallback_selection(
            evidence=self._failed_evidence(),
            out_dir=out,
            relion_autopick_bin=fake_bin,
        )
        assert (out / "comparison" / "final_picker_selection.json").exists()
        assert (out / "picker_model_selection_report.md").exists()


# ---------------------------------------------------------------------------
# Stage 03 tests
# ---------------------------------------------------------------------------


class TestStage03RelionLogRerun:
    def test_dry_run_returns_bundle(self, tmp_path: Path) -> None:
        pairs_csv = _make_pairs_csv(tmp_path)
        out = tmp_path / "rerun"
        result = stage03_relion_log_rerun(
            pairs_csv=pairs_csv,
            out_dir=out,
            relion_autopick_bin=None,
            limit=3,
            particle_diameter_angstrom=120.0,
            pixel_size_angstrom=1.0,
            qc_threshold_f1=0.2,
            dry_run=True,
        )
        assert result["mode"] == "dry_run"
        assert result["qc_threshold_f1"] == 0.2
        assert "qc_passed" in result
        assert (out / "relion_log_autopick_result.json").exists()

    def test_real_run_with_mock_runner_and_autopick_stars(self, tmp_path: Path) -> None:
        pairs_csv = _make_pairs_csv(tmp_path, n=3)
        _make_gt_coords(tmp_path, n=3)
        out = tmp_path / "rerun"

        autopick_dir = out / "AutoPick" / "job001" / "Micrographs"
        _make_autopick_stars(tmp_path, autopick_dir, n=3)

        result = stage03_relion_log_rerun(
            pairs_csv=pairs_csv,
            out_dir=out,
            relion_autopick_bin=tmp_path / "fake_relion_autopick",
            limit=3,
            particle_diameter_angstrom=120.0,
            pixel_size_angstrom=1.0,
            qc_threshold_f1=0.2,
            dry_run=False,
            command_runner=_mock_runner_success,
        )
        assert result["mode"] == "real_run"
        assert result["returncode"] == 0

    def test_qc_passed_when_f1_above_threshold(self, tmp_path: Path) -> None:
        pairs_csv = _make_pairs_csv(tmp_path, n=2)
        _make_gt_coords(tmp_path, n=2)
        out = tmp_path / "rerun"

        autopick_dir = out / "AutoPick" / "job001" / "Micrographs"
        _make_autopick_stars(tmp_path, autopick_dir, n=2)

        result = stage03_relion_log_rerun(
            pairs_csv=pairs_csv,
            out_dir=out,
            relion_autopick_bin=None,
            limit=2,
            particle_diameter_angstrom=120.0,
            pixel_size_angstrom=1.0,
            qc_threshold_f1=0.0,  # threshold=0 → always pass if any particles matched
            dry_run=False,
            command_runner=_mock_runner_success,
        )
        # With threshold=0, any non-None F1 ≥ 0 passes.
        assert result["qc_passed"] is True or result["f1"] is None

    def test_fail_runner_returns_error(self, tmp_path: Path) -> None:
        pairs_csv = _make_pairs_csv(tmp_path, n=2)
        out = tmp_path / "rerun"
        result = stage03_relion_log_rerun(
            pairs_csv=pairs_csv,
            out_dir=out,
            relion_autopick_bin=None,
            limit=2,
            particle_diameter_angstrom=120.0,
            pixel_size_angstrom=1.0,
            qc_threshold_f1=0.2,
            dry_run=False,
            command_runner=_mock_runner_fail,
        )
        assert result["returncode"] == 1


# ---------------------------------------------------------------------------
# Stage 04 tests
# ---------------------------------------------------------------------------


class TestStage04MaterialiseExtractProject:
    def test_creates_project_structure(self, tmp_path: Path) -> None:
        pairs_csv = _make_pairs_csv(tmp_path, n=3)
        relion_log_out = tmp_path / "rerun"
        _make_converted_coords(relion_log_out, n=3)

        out = tmp_path / "extract"
        manifest = stage04_materialise_extract_project(
            pairs_csv=pairs_csv,
            relion_log_out=relion_log_out,
            out_dir=out,
            limit=3,
            pixel_size_angstrom=1.0,
            dry_run=True,
            extract_size=256,
        )
        project = out / "relion_project"
        assert (project / "micrographs.star").exists()
        assert (project / "Micrographs").is_dir()
        assert (project / "Coords").is_dir()
        assert (project / "Coords" / "Micrographs").is_dir()
        assert manifest["n_micrographs"] == 3

    def test_converts_coords_to_micrographs_box(self, tmp_path: Path) -> None:
        pairs_csv = _make_pairs_csv(tmp_path, n=2)
        relion_log_out = tmp_path / "rerun"
        _make_converted_coords(relion_log_out, n=2)

        out = tmp_path / "extract"
        manifest = stage04_materialise_extract_project(
            pairs_csv=pairs_csv,
            relion_log_out=relion_log_out,
            out_dir=out,
            limit=2,
            pixel_size_angstrom=1.0,
            dry_run=True,
            extract_size=200,
        )
        coord_dir = out / "relion_project" / "Coords" / "Micrographs"
        box_files = sorted(coord_dir.glob("*.box"))
        assert len(box_files) == 2
        text = box_files[0].read_text(encoding="utf-8")
        assert "0.000000 100.000000 200 200" in text
        assert manifest["n_converted_coord_files"] == 2
        assert manifest["n_box_files"] == 4
        assert manifest["coord_source"] == "relion_log_converted_coords"

    def test_manifest_written(self, tmp_path: Path) -> None:
        pairs_csv = _make_pairs_csv(tmp_path, n=2)
        out = tmp_path / "extract"
        stage04_materialise_extract_project(
            pairs_csv=pairs_csv,
            relion_log_out=tmp_path / "rerun",
            out_dir=out,
            limit=2,
            pixel_size_angstrom=1.0,
            dry_run=True,
            extract_size=256,
        )
        assert (out / "extract_project_manifest.json").exists()

    def test_respects_limit(self, tmp_path: Path) -> None:
        pairs_csv = _make_pairs_csv(tmp_path, n=5)
        relion_log_out = tmp_path / "rerun"
        _make_converted_coords(relion_log_out, n=5)
        out = tmp_path / "extract"
        manifest = stage04_materialise_extract_project(
            pairs_csv=pairs_csv,
            relion_log_out=relion_log_out,
            out_dir=out,
            limit=2,
            pixel_size_angstrom=1.0,
            dry_run=True,
            extract_size=256,
        )
        assert manifest["n_micrographs"] == 2


# ---------------------------------------------------------------------------
# Stage 05 tests
# ---------------------------------------------------------------------------


class TestStage05Extract:
    def test_dry_run_skips_execution(self, tmp_path: Path) -> None:
        extract_dir = tmp_path / "extract"
        result = stage05_extract(
            extract_project_dir=extract_dir,
            relion_bin_dir=tmp_path / "relion_bin",
            extract_size=256,
            bg_radius=100,
            dry_run=True,
        )
        assert result["dry_run"] is True
        assert result["skipped"] is True
        assert result["returncode"] is None
        assert (extract_dir / "extract_command.json").exists()

    def test_real_run_calls_relion_preprocess(self, tmp_path: Path) -> None:
        extract_dir = tmp_path / "extract"
        project = extract_dir / "relion_project"
        project.mkdir(parents=True)
        (project / "micrographs.star").write_text("", encoding="utf-8")

        called_cmds: List[List[str]] = []

        def fake_run(cmd: List[str], log_path: Path, cwd=None, env=None) -> Dict[str, Any]:
            called_cmds.append(cmd)
            # Simulate relion_preprocess writing a particles.star.
            ps = project / "Extract" / "job002_log" / "particles.star"
            ps.parent.mkdir(parents=True, exist_ok=True)
            ps.write_text(
                "data_\n\nloop_\n_rlnImageName #1\n001@Extract/job002_log/Particles/mic.mrcs\n",
                encoding="utf-8",
            )
            return {"returncode": 0, "runtime_sec": 0.1, "log": str(log_path), "has_error_pattern": False}

        with patch(
            "run_empiar10017_closed_loop_picker_to_reconstruction._run_cmd",
            side_effect=fake_run,
        ):
            result = stage05_extract(
                extract_project_dir=extract_dir,
                relion_bin_dir=tmp_path / "relion_bin",
                extract_size=256,
                bg_radius=100,
                dry_run=False,
            )

        assert result["returncode"] == 0
        assert result["n_particles"] == 1
        assert len(called_cmds) == 1
        assert "relion_preprocess" in called_cmds[0][0]
        assert "--coord_dir" in called_cmds[0]
        assert called_cmds[0][called_cmds[0].index("--coord_dir") + 1] == "Coords"
        assert "--coord_suffix" in called_cmds[0]
        assert called_cmds[0][called_cmds[0].index("--coord_suffix") + 1] == ".box"


# ---------------------------------------------------------------------------
# Stage 06 tests
# ---------------------------------------------------------------------------


class TestStage06Reconstruct:
    def test_dry_run_skips_execution(self, tmp_path: Path) -> None:
        result = stage06_reconstruct(
            extract_project_dir=tmp_path / "extract",
            recon_out_dir=tmp_path / "recon",
            relion_bin_dir=tmp_path / "relion_bin",
            n_particles=100,
            dry_run=True,
        )
        assert result["dry_run"] is True
        assert result["skipped"] is True
        assert result["returncode"] is None

    def test_skips_when_no_particles(self, tmp_path: Path) -> None:
        result = stage06_reconstruct(
            extract_project_dir=tmp_path / "extract",
            recon_out_dir=tmp_path / "recon",
            relion_bin_dir=tmp_path / "relion_bin",
            n_particles=0,
            dry_run=False,
        )
        assert result["skipped"] is True
        assert result["map_exists"] is False

    def test_real_run_calls_relion_reconstruct(self, tmp_path: Path) -> None:
        extract_dir = tmp_path / "extract"
        project = extract_dir / "relion_project"
        project.mkdir(parents=True)
        recon_dir = tmp_path / "recon"

        called: List[List[str]] = []

        def fake_run(cmd: List[str], log_path: Path, cwd=None, env=None) -> Dict[str, Any]:
            called.append(cmd)
            # Simulate map file creation.
            map_path = Path(cmd[cmd.index("--o") + 1])
            map_path.parent.mkdir(parents=True, exist_ok=True)
            map_path.write_bytes(b"\x00" * 1024)
            return {"returncode": 0, "runtime_sec": 0.1, "log": str(log_path), "has_error_pattern": False}

        with patch(
            "run_empiar10017_closed_loop_picker_to_reconstruction._run_cmd",
            side_effect=fake_run,
        ):
            result = stage06_reconstruct(
                extract_project_dir=extract_dir,
                recon_out_dir=recon_dir,
                relion_bin_dir=tmp_path / "relion_bin",
                n_particles=50,
                dry_run=False,
            )

        assert result["returncode"] == 0
        assert result["map_exists"] is True
        assert len(called) == 1
        assert "relion_reconstruct" in called[0][0]


# ---------------------------------------------------------------------------
# Stage 07 tests
# ---------------------------------------------------------------------------


class TestStage07WriteSummary:
    def _make_results(self) -> Dict[str, Any]:
        return {
            "evidence": {
                "selected_micro_f1": 0.031848,
                "qc_threshold_f1": 0.2,
                "picking_qc_passed": False,
            },
            "selection_result": {
                "final": {"selected": "relion_log_autopick", "final_action": "RUN_ALTERNATIVE_PICKER"},
            },
            "relion_result": {
                "mode": "dry_run",
                "f1": None,
                "precision": None,
                "recall": None,
                "predicted_count": 0,
                "gt_count": 0,
                "matched_count": 0,
                "qc_passed": False,
                "qc_threshold_f1": 0.2,
                "returncode": None,
                "skipped": True,
            },
            "extract_manifest": {"n_micrographs": 3},
            "extract_result": {"n_particles": 0, "returncode": None, "dry_run": True, "skipped": True},
            "recon_result": {"map_exists": False, "map_path": "/tmp/out.mrc", "dry_run": True, "skipped": True},
        }

    def test_writes_json_and_md(self, tmp_path: Path) -> None:
        r = self._make_results()
        summary = stage07_write_summary(
            out_root=tmp_path,
            evidence=r["evidence"],
            selection_result=r["selection_result"],
            relion_result=r["relion_result"],
            extract_manifest=r["extract_manifest"],
            extract_result=r["extract_result"],
            recon_result=r["recon_result"],
            pairs_csv=tmp_path / "pairs.csv",
            args_dict={"dry_run": True},
        )
        assert (tmp_path / "workflow_summary.json").exists()
        assert (tmp_path / "workflow_summary.md").exists()
        assert "scientific_caveat" in summary

    def test_summary_contains_stage_status(self, tmp_path: Path) -> None:
        r = self._make_results()
        summary = stage07_write_summary(
            out_root=tmp_path,
            evidence=r["evidence"],
            selection_result=r["selection_result"],
            relion_result=r["relion_result"],
            extract_manifest=r["extract_manifest"],
            extract_result=r["extract_result"],
            recon_result=r["recon_result"],
            pairs_csv=tmp_path / "pairs.csv",
            args_dict={"dry_run": True},
        )
        assert "stage_status" in summary
        assert len(summary["stage_status"]) == 6

    def test_summary_prefers_stage03_relion_when_qc_passed(self, tmp_path: Path) -> None:
        r = self._make_results()
        r["selection_result"] = {
            "final": {
                "selected": "STOP_PLAN_ONLY_NO_VIABLE_PICKER",
                "final_action": "STOP_PLAN_ONLY_NO_VIABLE_PICKER",
            }
        }
        r["relion_result"]["qc_passed"] = True
        r["relion_result"]["f1"] = 0.7

        summary = stage07_write_summary(
            out_root=tmp_path,
            evidence=r["evidence"],
            selection_result=r["selection_result"],
            relion_result=r["relion_result"],
            extract_manifest=r["extract_manifest"],
            extract_result=r["extract_result"],
            recon_result=r["recon_result"],
            pairs_csv=tmp_path / "pairs.csv",
            args_dict={"dry_run": False},
        )

        assert summary["picker_selection"]["selected_picker"] == "relion_log_autopick"
        assert summary["picker_selection"]["stage02_selected_picker"] == "STOP_PLAN_ONLY_NO_VIABLE_PICKER"

    def test_scientific_caveat_in_md(self, tmp_path: Path) -> None:
        r = self._make_results()
        stage07_write_summary(
            out_root=tmp_path,
            evidence=r["evidence"],
            selection_result=r["selection_result"],
            relion_result=r["relion_result"],
            extract_manifest=r["extract_manifest"],
            extract_result=r["extract_result"],
            recon_result=r["recon_result"],
            pairs_csv=tmp_path / "pairs.csv",
            args_dict={"dry_run": True},
        )
        md = (tmp_path / "workflow_summary.md").read_text()
        assert "Scientific caveat" in md
        assert "not final structure determination" in md.lower() or "not claim" in md.lower()


# ---------------------------------------------------------------------------
# Full pipeline integration test (dry-run, mocked RELION)
# ---------------------------------------------------------------------------


class TestFullPipelineDryRun:
    def test_dry_run_end_to_end(self, tmp_path: Path) -> None:
        """Run the full pipeline in dry-run mode; all stages must complete."""
        pairs_csv = _make_pairs_csv(tmp_path, n=3)
        _make_gt_coords(tmp_path, n=3)

        out_root = tmp_path / "pipeline_out"
        fake_bin = tmp_path / "relion_autopick"
        fake_bin.write_text("#!/bin/sh\n", encoding="utf-8")
        fake_bin.chmod(0o755)
        _make_converted_coords(out_root / "02_relion_log_rerun", n=3)

        main([
            "--pairs-csv", str(pairs_csv),
            "--relion-autopick-bin", str(fake_bin),
            "--relion-bin-dir", str(tmp_path),
            "--out-root", str(out_root),
            "--limit", "3",
            "--dry-run",
        ])

        # Required output structure exists.
        for subdir in ("01_picker_selection", "02_relion_log_rerun", "03_extract_project",
                       "04_reconstruction", "reports"):
            assert (out_root / subdir).is_dir(), f"{subdir} missing"

        assert (out_root / "workflow_summary.json").exists()
        assert (out_root / "workflow_summary.md").exists()

        summary = json.loads((out_root / "workflow_summary.json").read_text())
        assert summary["scientific_caveat"] != ""
        assert "stage_status" in summary

    def test_dry_run_produces_extract_project_layout(self, tmp_path: Path) -> None:
        """RELION extraction project directory must be created in dry-run."""
        pairs_csv = _make_pairs_csv(tmp_path, n=2)
        out_root = tmp_path / "pipeline_out"
        _make_converted_coords(out_root / "02_relion_log_rerun", n=2)

        main([
            "--pairs-csv", str(pairs_csv),
            "--out-root", str(out_root),
            "--limit", "2",
            "--dry-run",
        ])

        project = out_root / "03_extract_project" / "relion_project"
        assert (project / "micrographs.star").exists()
        assert (project / "Micrographs").is_dir()
        assert (project / "Coords" / "Micrographs").is_dir()
        assert (out_root / "03_extract_project" / "extract_command.json").exists()

    def test_workflow_summary_has_scientific_caveat(self, tmp_path: Path) -> None:
        pairs_csv = _make_pairs_csv(tmp_path, n=2)
        out_root = tmp_path / "pipeline_out"

        main([
            "--pairs-csv", str(pairs_csv),
            "--out-root", str(out_root),
            "--limit", "2",
            "--dry-run",
        ])

        summary = json.loads((out_root / "workflow_summary.json").read_text())
        assert "picker-stage backend selection" in summary["scientific_caveat"]
        assert "not final structure determination" in summary["scientific_caveat"]

    def test_topaz_evidence_recorded(self, tmp_path: Path) -> None:
        pairs_csv = _make_pairs_csv(tmp_path, n=2)
        out_root = tmp_path / "pipeline_out"

        main([
            "--pairs-csv", str(pairs_csv),
            "--out-root", str(out_root),
            "--limit", "2",
            "--dry-run",
        ])

        ev = json.loads(
            (out_root / "01_picker_selection" / "topaz_failed_evidence.json").read_text()
        )
        assert ev["picking_qc_passed"] is False

    def test_custom_topaz_evidence_loaded(self, tmp_path: Path) -> None:
        pairs_csv = _make_pairs_csv(tmp_path, n=2)
        ev_path = tmp_path / "topaz_ev.json"
        ev_path.write_text(
            json.dumps({
                "picking_qc_passed": False,
                "selected_micro_f1": 0.01,
                "qc_threshold_f1": 0.2,
            }),
            encoding="utf-8",
        )
        out_root = tmp_path / "pipeline_out"

        main([
            "--pairs-csv", str(pairs_csv),
            "--topaz-failed-evidence-json", str(ev_path),
            "--out-root", str(out_root),
            "--limit", "2",
            "--dry-run",
        ])

        ev_out = json.loads(
            (out_root / "01_picker_selection" / "topaz_failed_evidence.json").read_text()
        )
        assert ev_out["selected_micro_f1"] == 0.01

    def test_mocked_non_dry_run_reconstructs_when_particles_present(self, tmp_path: Path) -> None:
        pairs_csv = _make_pairs_csv(tmp_path, n=2)
        out_root = tmp_path / "pipeline_out"
        relion_bin = tmp_path / "relion_bin"
        relion_bin.mkdir(parents=True, exist_ok=True)

        def fake_stage03(**kwargs: Any) -> Dict[str, Any]:
            out_dir = kwargs["out_dir"]
            _make_converted_coords(out_dir, n=2)
            return {
                "mode": "real_run",
                "returncode": 0,
                "f1": 0.7,
                "precision": 0.7,
                "recall": 0.7,
                "predicted_count": 20,
                "gt_count": 20,
                "matched_count": 14,
                "qc_passed": True,
            }

        def fake_run(cmd: List[str], log_path: Path, cwd=None, env=None) -> Dict[str, Any]:
            if "relion_preprocess" in cmd[0]:
                ps = Path(cwd) / "Extract" / "job002_log" / "particles.star"
                ps.parent.mkdir(parents=True, exist_ok=True)
                ps.write_text(
                    "data_\n\nloop_\n_rlnImageName #1\n001@Extract/job002_log/Particles/mic.mrcs\n",
                    encoding="utf-8",
                )
            if "relion_reconstruct" in cmd[0]:
                map_path = Path(cmd[cmd.index("--o") + 1])
                map_path.parent.mkdir(parents=True, exist_ok=True)
                map_path.write_bytes(b"\x00" * 1024)
            return {"returncode": 0, "runtime_sec": 0.1, "log": str(log_path), "has_error_pattern": False}

        with patch(
            "run_empiar10017_closed_loop_picker_to_reconstruction.stage03_relion_log_rerun",
            side_effect=fake_stage03,
        ), patch(
            "run_empiar10017_closed_loop_picker_to_reconstruction._run_cmd",
            side_effect=fake_run,
        ):
            main([
                "--pairs-csv", str(pairs_csv),
                "--relion-bin-dir", str(relion_bin),
                "--out-root", str(out_root),
                "--limit", "2",
            ])

        summary = json.loads((out_root / "workflow_summary.json").read_text())
        assert summary["extraction"]["n_particles"] > 0
        assert summary["reconstruction"]["map_exists"] is True
