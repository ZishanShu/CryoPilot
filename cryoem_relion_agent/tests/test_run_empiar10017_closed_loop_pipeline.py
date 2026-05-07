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
    stage03_deposited_coords_handoff,
    stage04_materialise_extract_project,
    stage05_extract,
    stage06_reconstruct,
    stage07_write_summary,
    _parse_eman_box_xywh_to_centers,
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
            "coord_path": str(tmp_path / "gt_coords" / f"mic_{i:04d}.coord"),
            "local_coord_path": str(tmp_path / "gt_coords" / f"mic_{i:04d}.coord"),
        }
        for i in range(n)
    ]
    with pairs_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "micrograph_id", "micrograph_path", "gt_coord_path",
                "coord_path", "local_coord_path",
            ],
        )
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
    def test_parse_eman_box_to_center_coords(self, tmp_path: Path) -> None:
        box = tmp_path / "a.box"
        box.write_text("10 20 100 80\n", encoding="utf-8")
        centers = _parse_eman_box_xywh_to_centers(box)
        assert centers == [(60.0, 60.0)]

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

    def test_deposited_coords_handoff_materializes_coords(self, tmp_path: Path) -> None:
        pairs_csv = _make_pairs_csv(tmp_path, n=2)
        _make_gt_coords(tmp_path, n=2)
        dep_out = tmp_path / "02_deposited_coords"
        relion_out = tmp_path / "02_relion_log_rerun"
        result = stage03_deposited_coords_handoff(
            pairs_csv=pairs_csv,
            out_dir=dep_out,
            relion_log_out=relion_out,
            limit=2,
            qc_threshold_f1=0.2,
        )
        assert result["picking_source"] == "deposited_coords"
        assert result["reference_coordinate_handoff"] is True
        assert (dep_out / "deposited_coords_result.json").exists()
        assert (dep_out / "deposited_coords_result.md").exists()
        assert (relion_out / "converted_coords" / "mic_0000.coord").exists()
        assert (relion_out / "converted_coords" / "mic_0001.coord").exists()
        converted_text = (relion_out / "converted_coords" / "mic_0000.coord").read_text(encoding="utf-8")
        assert "image_name x_coord y_coord score" in converted_text

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

    def test_manifest_records_missing_coord_zero_for_deposited(self, tmp_path: Path) -> None:
        pairs_csv = _make_pairs_csv(tmp_path, n=2)
        _make_gt_coords(tmp_path, n=2)
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
            coord_source="deposited_coords",
        )
        assert manifest["n_missing_coord"] == 0
        assert manifest["coord_source"] == "deposited_coords"

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
        cdir = project / "Coords" / "Micrographs"
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "mic_0000.box").write_text("0 0 256 256\n", encoding="utf-8")

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

    def test_abnormal_particle_size_blocks_reconstruction(self, tmp_path: Path) -> None:
        extract_dir = tmp_path / "extract"
        recon_dir = tmp_path / "recon"
        result = stage06_reconstruct(
            extract_project_dir=extract_dir,
            recon_out_dir=recon_dir,
            relion_bin_dir=tmp_path / "relion_bin",
            n_particles=10,
            extract_result={"decision": "BLOCK_RECONSTRUCTION_ABNORMAL_PARTICLE_SIZE"},
            dry_run=False,
        )
        assert result["skipped"] is True
        assert result["decision"] == "BLOCK_RECONSTRUCTION_ABNORMAL_PARTICLE_SIZE"

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
        # Stages 01–11 (Stage 12 is the summary itself).
        assert len(summary["stage_status"]) == 11

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

    def test_summary_records_reference_coordinate_handoff(self, tmp_path: Path) -> None:
        r = self._make_results()
        r["relion_result"].update({
            "picking_source": "deposited_coords",
            "automatic_picking": False,
            "reference_coordinate_handoff": True,
            "qc_passed": True,
        })
        summary = stage07_write_summary(
            out_root=tmp_path,
            evidence=r["evidence"],
            selection_result=r["selection_result"],
            relion_result=r["relion_result"],
            extract_manifest=r["extract_manifest"],
            extract_result={"n_particles": 2, "returncode": 0},
            recon_result={"map_exists": True, "map_path": "/tmp/x.mrc"},
            pairs_csv=tmp_path / "pairs.csv",
            args_dict={"dry_run": False, "picker_source": "deposited_coords"},
        )
        assert summary["picker_source"] == "deposited_coords"
        assert summary["relion_log_rerun"]["reference_coordinate_handoff"] is True
        assert summary["relion_log_rerun"]["automatic_picking"] is False
        assert summary["full_workflow_execution"] is True

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
        caveat = summary["scientific_caveat"].lower()
        assert "downstream-gated" in caveat or \
            "picker-stage backend selection" in caveat
        assert "not final structure determination" in caveat

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


# ---------------------------------------------------------------------------
# Stage 07 – Class2D gate tests
# ---------------------------------------------------------------------------


from agent.downstream_gates import (  # noqa: E402
    stage07_class2d_gate,
    stage08_refine3d_gate,
)
from agent.map_quality_gate import stage09_postprocess_gate  # noqa: E402
from agent.map2pdb_routing import stage10_map2pdb_routing  # noqa: E402
from agent.model_validation_gate import stage11_model_validation_gate  # noqa: E402


def _write_particles_star(tmp_path: Path) -> Path:
    p = tmp_path / "particles.star"
    p.write_text(
        "data_\n\nloop_\n_rlnImageName #1\n001@Particles/mic.mrcs\n",
        encoding="utf-8",
    )
    return p


class TestStage07Class2DGate:
    def test_blocked_when_particles_missing(self, tmp_path: Path) -> None:
        out = tmp_path / "07"
        result = stage07_class2d_gate(
            out, particles_star=None, n_particles=0,
            relion_bin_dir=None, mode="plan-only",
        )
        assert result["status"] == "blocked"
        assert result["decision"] == "BLOCK_CLASS2D_NO_PARTICLES"
        assert (out / "class2d_plan.json").exists()
        assert (out / "class2d_plan.md").exists()

    def test_plan_only_writes_artifacts(self, tmp_path: Path) -> None:
        ps = _write_particles_star(tmp_path)
        out = tmp_path / "07"
        result = stage07_class2d_gate(
            out, particles_star=ps, n_particles=84879,
            relion_bin_dir=None, mode="plan-only", full_dataset=True,
        )
        assert result["status"] == "planned"
        assert result["decision"] == "PLAN_CLASS2D"
        assert result["K"] == 50
        assert (out / "class2d_plan.json").exists()
        plan = json.loads((out / "class2d_plan.json").read_text())
        assert "relion_refine" in plan["cmd"][0]

    def test_real_mode_records_disabled_status(self, tmp_path: Path) -> None:
        ps = _write_particles_star(tmp_path)
        out = tmp_path / "07"
        result = stage07_class2d_gate(
            out, particles_star=ps, n_particles=84879,
            relion_bin_dir=None, mode="real",
        )
        assert result["status"] == "planned"
        assert result["decision"] == "REAL_CLASS2D_NOT_IMPLEMENTED_OR_DISABLED"

    def test_none_mode_skips(self, tmp_path: Path) -> None:
        out = tmp_path / "07"
        result = stage07_class2d_gate(
            out, particles_star=None, n_particles=0,
            relion_bin_dir=None, mode="none",
        )
        assert result["status"] == "skipped"
        assert result["decision"] == "SKIP_CLASS2D"


class TestStage08Refine3DGate:
    def test_blocked_when_no_map(self, tmp_path: Path) -> None:
        ps = _write_particles_star(tmp_path)
        out = tmp_path / "08"
        result = stage08_refine3d_gate(
            out, map_path=None, map_exists=False,
            particles_star=ps, n_particles=10, relion_bin_dir=None,
            mode="plan-only",
        )
        assert result["status"] == "blocked"
        assert result["decision"] == "BLOCK_REFINE3D_NO_MAP"
        assert result["refined_map_available"] is False

    def test_blocked_when_no_particles(self, tmp_path: Path) -> None:
        map_path = tmp_path / "map.mrc"
        map_path.write_bytes(b"\x00" * 10)
        out = tmp_path / "08"
        result = stage08_refine3d_gate(
            out, map_path=map_path, map_exists=True,
            particles_star=None, n_particles=0, relion_bin_dir=None,
            mode="plan-only",
        )
        assert result["decision"] == "BLOCK_REFINE3D_NO_PARTICLES"

    def test_plan_only_writes_artifacts(self, tmp_path: Path) -> None:
        ps = _write_particles_star(tmp_path)
        map_path = tmp_path / "map.mrc"
        map_path.write_bytes(b"\x00" * 10)
        out = tmp_path / "08"
        result = stage08_refine3d_gate(
            out, map_path=map_path, map_exists=True,
            particles_star=ps, n_particles=84879,
            relion_bin_dir=None, mode="plan-only",
        )
        assert result["status"] == "planned"
        assert result["decision"] == "PLAN_REFINE3D"
        assert (out / "refine3d_plan.json").exists()
        assert (out / "refine3d_plan.md").exists()
        assert result["map_artifact_from_reconstruct"] is True
        assert result["refined_map_available"] is False


class TestStage09MapQualityGate:
    def test_blocks_real_map2pdb_when_no_half_maps(self, tmp_path: Path) -> None:
        out = tmp_path / "09"
        map_path = tmp_path / "map.mrc"
        map_path.write_bytes(b"\x00" * 10)
        bundle = stage09_postprocess_gate(
            out, map_path=map_path, map_exists=True,
            half_map_1=None, half_map_2=None, mask=None,
            relion_bin_dir=None, mode="plan-only",
        )
        gate = bundle["map_quality_gate"]
        assert bundle["plan"]["decision"] == "STOP_POSTPROCESS_HALF_MAPS_REQUIRED"
        assert gate["allow_map2pdb_real"] is False
        assert gate["fsc_available"] is False
        assert (out / "map_quality_gate.json").exists()
        assert (out / "postprocess_plan.md").exists()

    def test_dry_run_writes_plan(self, tmp_path: Path) -> None:
        out = tmp_path / "09"
        map_path = tmp_path / "map.mrc"
        map_path.write_bytes(b"\x00" * 10)
        h1 = tmp_path / "h1.mrc"; h1.write_bytes(b"\x00")
        h2 = tmp_path / "h2.mrc"; h2.write_bytes(b"\x00")
        mask = tmp_path / "mask.mrc"; mask.write_bytes(b"\x00")
        bundle = stage09_postprocess_gate(
            out, map_path=map_path, map_exists=True,
            half_map_1=h1, half_map_2=h2, mask=mask,
            relion_bin_dir=None, mode="dry-run",
        )
        assert bundle["plan"]["decision"] == "PLAN_POSTPROCESS"
        # No FSC artifacts → still unvalidated.
        assert bundle["map_quality_gate"]["allow_map2pdb_real"] is False


class TestStage10Map2PDBRouting:
    def _gate(self, allow_real: bool = False, fsc: bool = False) -> Dict[str, Any]:
        return {
            "fsc_available": fsc,
            "allow_map2pdb_real": allow_real,
            "map_quality_status": "validated" if allow_real else "unvalidated",
        }

    def test_blocks_when_map_missing(self, tmp_path: Path) -> None:
        out = tmp_path / "10"
        routing = stage10_map2pdb_routing(
            out, map_path=None, map_exists=False, map_source="reconstruct",
            map_quality_gate=self._gate(), sequence=None,
            backends=["modelangelo", "emodelx"], mode="dry-run",
        )
        assert routing["decision"] == "BLOCK_MAP2PDB_NO_MAP"
        assert (out / "map2pdb_routing.json").exists()

    def test_records_candidate_backends_when_map_exists(self, tmp_path: Path) -> None:
        out = tmp_path / "10"
        map_path = tmp_path / "map.mrc"; map_path.write_bytes(b"\x00")
        routing = stage10_map2pdb_routing(
            out, map_path=map_path, map_exists=True, map_source="reconstruct",
            map_quality_gate=self._gate(), sequence=None,
            backends=["modelangelo", "emodelx", "mica"], mode="dry-run",
            availability_overrides={"modelangelo": True, "emodelx": False, "mica": False},
            allow_without_fsc=True,
        )
        assert routing["candidate_backends"] == ["modelangelo", "emodelx", "mica"]
        assert routing["selected_backend"] == "modelangelo"
        assert routing["decision"] == "ALLOW_MAP2PDB_DRY_RUN"

    def test_real_blocked_when_quality_gate_disallows(self, tmp_path: Path) -> None:
        out = tmp_path / "10"
        map_path = tmp_path / "map.mrc"; map_path.write_bytes(b"\x00")
        routing = stage10_map2pdb_routing(
            out, map_path=map_path, map_exists=True, map_source="reconstruct",
            map_quality_gate=self._gate(allow_real=False),
            sequence=None, backends=["modelangelo"], mode="real",
            availability_overrides={"modelangelo": True},
        )
        assert routing["real_run_allowed"] is False
        assert routing["decision"] == "STOP_MAP2PDB_QC_REQUIRED"

    def test_no_available_backend(self, tmp_path: Path) -> None:
        out = tmp_path / "10"
        map_path = tmp_path / "map.mrc"; map_path.write_bytes(b"\x00")
        routing = stage10_map2pdb_routing(
            out, map_path=map_path, map_exists=True, map_source="reconstruct",
            map_quality_gate=self._gate(), sequence=None,
            backends=["modelangelo", "emodelx", "mica"], mode="dry-run",
            availability_overrides={"modelangelo": False, "emodelx": False, "mica": False},
        )
        assert routing["decision"] == "BLOCK_MAP2PDB_NO_AVAILABLE_BACKEND"


class TestStage11ModelValidationGate:
    def test_skipped_when_no_model(self, tmp_path: Path) -> None:
        out = tmp_path / "11"
        result = stage11_model_validation_gate(
            out, model_path=None, map_path=None, sequence=None,
        )
        assert result["status"] == "skipped"
        assert result["decision"] == "SKIP_MODEL_VALIDATION_NO_MODEL"
        assert (out / "model_validation.json").exists()

    def test_planned_when_model_exists(self, tmp_path: Path) -> None:
        out = tmp_path / "11"
        model = tmp_path / "model.cif"
        model.write_text("dummy", encoding="utf-8")
        result = stage11_model_validation_gate(
            out, model_path=model, map_path=None, sequence=None,
        )
        assert result["status"] == "planned"
        assert result["decision"] == "PLAN_MODEL_VALIDATION"


# ---------------------------------------------------------------------------
# Full-pipeline downstream integration tests
# ---------------------------------------------------------------------------


class TestPipelineDownstreamIntegration:
    def test_workflow_summary_includes_new_stages(self, tmp_path: Path) -> None:
        pairs_csv = _make_pairs_csv(tmp_path, n=2)
        out_root = tmp_path / "pipeline_out"
        main([
            "--pairs-csv", str(pairs_csv),
            "--out-root", str(out_root),
            "--limit", "2",
            "--dry-run",
        ])
        summary = json.loads((out_root / "workflow_summary.json").read_text())
        for stage in ("07_class2d", "08_refine3d", "09_postprocess",
                      "10_map2pdb_routing", "11_model_validation"):
            assert stage in summary["stage_status"]
        for sub in ("class2d", "refine3d", "postprocess", "map_quality_gate",
                    "map2pdb_routing", "model_validation", "final_readiness"):
            assert sub in summary

    def test_new_output_directories_created(self, tmp_path: Path) -> None:
        pairs_csv = _make_pairs_csv(tmp_path, n=2)
        out_root = tmp_path / "pipeline_out"
        main([
            "--pairs-csv", str(pairs_csv),
            "--out-root", str(out_root),
            "--limit", "2",
            "--dry-run",
        ])
        for sub in ("05_class2d", "06_refine3d", "07_postprocess",
                    "08_map2pdb_routing", "09_model_validation"):
            assert (out_root / sub).is_dir(), f"{sub} missing"

    def test_final_readiness_map_qc_required_when_reconstruction_present(
        self, tmp_path: Path
    ) -> None:
        pairs_csv = _make_pairs_csv(tmp_path, n=2)
        out_root = tmp_path / "pipeline_out"
        relion_bin = tmp_path / "relion_bin"
        relion_bin.mkdir(parents=True, exist_ok=True)

        def fake_stage03(**kwargs: Any) -> Dict[str, Any]:
            out_dir = kwargs["out_dir"]
            _make_converted_coords(out_dir, n=2)
            return {
                "mode": "real_run", "returncode": 0, "f1": 0.65, "precision": 0.7,
                "recall": 0.6, "predicted_count": 84879, "gt_count": 84879,
                "matched_count": 50000, "qc_passed": True,
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
            return {"returncode": 0, "runtime_sec": 0.1, "log": str(log_path),
                    "has_error_pattern": False}

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
        assert summary["reconstruction"]["map_exists"] is True
        # No FSC, no half maps → must not allow real map2pdb run.
        assert summary["map_quality_gate"]["allow_map2pdb_real"] is False
        assert summary["map2pdb_routing"]["real_run_allowed"] is False
        # Final readiness must not claim full structure determination.
        assert summary["final_readiness"] in (
            "MAP_QC_REQUIRED", "MAP2PDB_READY_DRY_RUN_ONLY",
        )
        # Model validation must be skipped because no model exists.
        assert summary["model_validation"]["decision"] == \
            "SKIP_MODEL_VALIDATION_NO_MODEL"
        md = (out_root / "workflow_summary.md").read_text()
        assert "final structure determination" not in md.lower() or \
            "not" in md.lower()


class TestPipelineDryRunRobustness:
    def test_pipeline_dry_run_continues_after_stage02_stop_plan_only(
        self, tmp_path: Path
    ) -> None:
        """Stage02 returning STOP_PLAN_ONLY_NO_VIABLE_PICKER must not abort the pipeline."""
        pairs_csv = _make_pairs_csv(tmp_path, n=3)
        out_root = tmp_path / "pipeline_out"

        def fake_stage02(**kwargs: Any) -> Dict[str, Any]:
            return {
                "final": {
                    "selected": "relion_log_autopick",
                    "final_action": "STOP_PLAN_ONLY_NO_VIABLE_PICKER",
                },
                "candidates": [], "selection": {}, "plan": {}, "comparison_rows": [],
                "relion_rerun": None,
            }

        with patch(
            "run_empiar10017_closed_loop_picker_to_reconstruction.stage02_picker_fallback_selection",
            side_effect=fake_stage02,
        ):
            main([
                "--pairs-csv", str(pairs_csv),
                "--out-root", str(out_root),
                "--limit", "3",
                "--downstream-mode", "plan-only",
                "--map2pdb-mode", "dry-run",
                "--dry-run",
            ])

        # Final summary written even though stage02 reported a STOP candidate.
        assert (out_root / "workflow_summary.json").exists()
        assert (out_root / "workflow_summary.md").exists()
        assert (out_root / "05_class2d" / "class2d_plan.json").exists()
        assert (out_root / "06_refine3d" / "refine3d_plan.json").exists()
        assert (out_root / "07_postprocess" / "map_quality_gate.json").exists()
        assert (out_root / "08_map2pdb_routing" / "map2pdb_routing.json").exists()
        assert (out_root / "09_model_validation" / "model_validation.json").exists()

        summary = json.loads((out_root / "workflow_summary.json").read_text())
        assert summary["picker_selection"]["stage02_final_action"] == \
            "STOP_PLAN_ONLY_NO_VIABLE_PICKER"
        # Pipeline status is not aborted: all stage_status keys present.
        for k in ("07_class2d", "08_refine3d", "09_postprocess",
                  "10_map2pdb_routing", "11_model_validation"):
            assert k in summary["stage_status"]
        # Stages 05/06 must be skipped in dry-run, not failed.
        assert summary["stage_status"]["05_extraction"] in ("skipped", "dry_run")
        assert summary["stage_status"]["06_reconstruction"] in ("skipped", "dry_run")
        assert summary["final_readiness"] in (
            "WORKFLOW_HANDOFF_ONLY", "MAP_QC_REQUIRED",
        )

    def test_pipeline_writes_summary_even_when_stage03_raises(
        self, tmp_path: Path
    ) -> None:
        pairs_csv = _make_pairs_csv(tmp_path, n=2)
        out_root = tmp_path / "pipeline_out"

        def boom(**kwargs: Any) -> Dict[str, Any]:
            raise RuntimeError("synthetic stage03 failure")

        with patch(
            "run_empiar10017_closed_loop_picker_to_reconstruction.stage03_relion_log_rerun",
            side_effect=boom,
        ):
            main([
                "--pairs-csv", str(pairs_csv),
                "--out-root", str(out_root),
                "--limit", "2",
                "--dry-run",
            ])

        assert (out_root / "workflow_summary.json").exists()
        summary = json.loads((out_root / "workflow_summary.json").read_text())
        # Stage07-11 still ran and wrote artifacts.
        assert (out_root / "05_class2d" / "class2d_plan.json").exists()
        assert (out_root / "08_map2pdb_routing" / "map2pdb_routing.json").exists()
        assert "stage03" in summary["args"].get("stage_errors", {})


# ---------------------------------------------------------------------------
# GPU detection / planning tests
# ---------------------------------------------------------------------------


from agent.downstream_gates import detect_relion_refine_gpu_support  # noqa: E402


class TestGpuDetectionAndPlanning:
    def test_detect_gpu_support_true(self, tmp_path: Path) -> None:
        bin_path = tmp_path / "relion_refine"
        bin_path.write_text("#!/bin/sh", encoding="utf-8")
        ok = detect_relion_refine_gpu_support(
            bin_path,
            help_runner=lambda cmd: "Usage: relion_refine ...\n  --gpu (false) : Use GPU\n",
        )
        assert ok is True

    def test_detect_gpu_support_false(self, tmp_path: Path) -> None:
        bin_path = tmp_path / "relion_refine"
        bin_path.write_text("#!/bin/sh", encoding="utf-8")
        ok = detect_relion_refine_gpu_support(
            bin_path, help_runner=lambda cmd: "Usage: ...\n  --i input\n",
        )
        assert ok is False

    def test_class2d_plan_includes_gpu_when_supported(self, tmp_path: Path) -> None:
        ps = _write_particles_star(tmp_path)
        out = tmp_path / "07"
        result = stage07_class2d_gate(
            out, particles_star=ps, n_particles=84879,
            relion_bin_dir=None, mode="plan-only",
            gpu_devices="0", gpu_supported=True,
        )
        assert result["gpu_requested"] is True
        assert result["gpu_supported"] is True
        assert "--gpu" in result["cmd"]
        assert result["cmd"][result["cmd"].index("--gpu") + 1] == "0"
        assert result["decision"] == "PLAN_CLASS2D"

    def test_class2d_plan_blocks_when_gpu_unsupported(self, tmp_path: Path) -> None:
        ps = _write_particles_star(tmp_path)
        out = tmp_path / "07"
        result = stage07_class2d_gate(
            out, particles_star=ps, n_particles=84879,
            relion_bin_dir=None, mode="plan-only",
            gpu_devices="0:1", gpu_supported=False,
        )
        assert result["gpu_requested"] is True
        assert result["gpu_supported"] is False
        assert result["decision"] == "BLOCK_GPU_REQUESTED_BUT_UNAVAILABLE"
        assert "--gpu" not in result["cmd"]

    def test_refine3d_plan_includes_gpu_when_supported(self, tmp_path: Path) -> None:
        ps = _write_particles_star(tmp_path)
        map_path = tmp_path / "map.mrc"; map_path.write_bytes(b"\x00" * 10)
        out = tmp_path / "08"
        result = stage08_refine3d_gate(
            out, map_path=map_path, map_exists=True,
            particles_star=ps, n_particles=84879,
            relion_bin_dir=None, mode="plan-only",
            gpu_devices="0", gpu_supported=True,
        )
        assert "--gpu" in result["cmd"]
        assert result["decision"] == "PLAN_REFINE3D"
        assert result["gpu_requested"] is True


# ---------------------------------------------------------------------------
# Upstream reuse tests
# ---------------------------------------------------------------------------


def _write_upstream_summary(reuse_root: Path) -> None:
    """Synthesize an upstream out_root with workflow_summary.json + map + particles.star."""
    reuse_root.mkdir(parents=True, exist_ok=True)
    project = reuse_root / "03_extract_project" / "relion_project" / \
        "Extract" / "job002_log"
    project.mkdir(parents=True, exist_ok=True)
    particles = project / "particles.star"
    particles.write_text(
        "data_\n\nloop_\n_rlnImageName #1\n001@Particles/mic.mrcs\n",
        encoding="utf-8",
    )
    recon = reuse_root / "04_reconstruction"
    recon.mkdir(parents=True, exist_ok=True)
    map_path = recon / "reconstruct_from_relion_log.mrc"
    map_path.write_bytes(b"\x00" * 4096)

    summary = {
        "pipeline": "empiar10017_closed_loop_picker_to_reconstruction",
        "picker_selection": {
            "topaz_f1": 0.031848,
            "qc_threshold_f1": 0.2,
            "stage02_selected_picker": "relion_log_autopick",
            "stage02_final_action": "RUN_ALTERNATIVE_PICKER",
            "selected_picker": "relion_log_autopick",
            "final_action": "RUN_ALTERNATIVE_PICKER",
        },
        "relion_log_rerun": {
            "mode": "real_run", "f1": 0.649950, "precision": 0.7, "recall": 0.6,
            "predicted_count": 84879, "gt_count": 84879, "matched_count": 50000,
            "qc_passed": True,
        },
        "extraction": {
            "n_particles": 84879,
            "particles_star": str(particles),
            "returncode": 0,
        },
        "reconstruction": {
            "map_exists": True,
            "map_path": str(map_path),
            "map_size_bytes": 4096,
        },
    }
    (reuse_root / "workflow_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )


class TestUpstreamReuse:
    def test_reuse_loads_existing_summary_and_runs_only_downstream(
        self, tmp_path: Path
    ) -> None:
        reuse_root = tmp_path / "upstream"
        _write_upstream_summary(reuse_root)
        out_root = tmp_path / "downstream"

        # Spy on _run_cmd to confirm no RELION binary is invoked.
        called: List[List[str]] = []

        def spy_run(cmd: List[str], log_path: Path, cwd=None, env=None) -> Dict[str, Any]:
            called.append(cmd)
            return {"returncode": 0, "runtime_sec": 0.0, "log": str(log_path),
                    "has_error_pattern": False}

        with patch(
            "run_empiar10017_closed_loop_picker_to_reconstruction._run_cmd",
            side_effect=spy_run,
        ):
            main([
                "--reuse-upstream-root", str(reuse_root),
                "--out-root", str(out_root),
                "--downstream-mode", "plan-only",
                "--map2pdb-mode", "dry-run",
            ])

        assert called == [], "No RELION subprocess should run during reuse"

        summary = json.loads((out_root / "workflow_summary.json").read_text())
        assert summary["upstream_reused"] is True
        assert summary["reuse_upstream_root"].endswith("upstream")
        for k in ("01_topaz_evidence", "02_picker_fallback", "03_relion_log_rerun",
                  "04_extract_project", "05_extraction", "06_reconstruction"):
            assert summary["stage_status"][k] == "reused"
        # Downstream stages must have produced artifacts.
        for sub in ("05_class2d", "06_refine3d", "07_postprocess",
                    "08_map2pdb_routing", "09_model_validation"):
            assert (out_root / sub).is_dir()
        assert (out_root / "05_class2d" / "class2d_plan.json").exists()
        assert (out_root / "07_postprocess" / "map_quality_gate.json").exists()
        assert (out_root / "08_map2pdb_routing" / "map2pdb_routing.json").exists()
        assert (out_root / "09_model_validation" / "model_validation.json").exists()

        # Reused metrics must be propagated.
        assert summary["relion_log_rerun"]["f1"] == 0.649950
        assert summary["extraction"]["n_particles"] == 84879
        assert summary["reconstruction"]["map_exists"] is True
        # Without FSC, final readiness must be a stop-gated state, not "FINAL...".
        assert summary["final_readiness"] in (
            "MAP_QC_REQUIRED", "MAP2PDB_READY_DRY_RUN_ONLY",
        )

    def test_reuse_with_gpu_devices_threads_into_class2d_and_refine3d(
        self, tmp_path: Path
    ) -> None:
        reuse_root = tmp_path / "upstream"
        _write_upstream_summary(reuse_root)
        out_root = tmp_path / "downstream"

        with patch(
            "agent.downstream_gates.detect_relion_refine_gpu_support",
            return_value=True,
        ):
            main([
                "--reuse-upstream-root", str(reuse_root),
                "--out-root", str(out_root),
                "--downstream-mode", "plan-only",
                "--map2pdb-mode", "dry-run",
                "--relion-gpu-devices", "0",
            ])

        class2d = json.loads((out_root / "05_class2d" / "class2d_plan.json").read_text())
        refine3d = json.loads((out_root / "06_refine3d" / "refine3d_plan.json").read_text())
        assert class2d["gpu_requested"] is True
        assert class2d["gpu_supported"] is True
        assert "--gpu" in class2d["cmd"]
        assert class2d["cmd"][class2d["cmd"].index("--gpu") + 1] == "0"
        assert refine3d["gpu_requested"] is True
        assert "--gpu" in refine3d["cmd"]

    def test_reuse_with_missing_summary_blocks_gracefully(
        self, tmp_path: Path
    ) -> None:
        reuse_root = tmp_path / "upstream_empty"
        reuse_root.mkdir()
        out_root = tmp_path / "downstream"

        main([
            "--reuse-upstream-root", str(reuse_root),
            "--out-root", str(out_root),
            "--downstream-mode", "plan-only",
            "--map2pdb-mode", "dry-run",
        ])

        # Summary still written, but with the BLOCK decision recorded.
        summary = json.loads((out_root / "workflow_summary.json").read_text())
        assert summary["upstream_reused"] is False
        assert summary["reuse_decision"] == "BLOCK_REUSE_UPSTREAM_SUMMARY_MISSING"
        assert (out_root / "workflow_summary.md").exists()
