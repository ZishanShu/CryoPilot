"""Reproducible closed-loop pipeline: picker fallback → RELION extraction → reconstruction.

Stages
------
01  Record failed Topaz evidence
02  Picker fallback selection  (agent.picker_fallback)
03  RELION LoG rerun + F1 evaluation  (agent.relion_log_autopick_runner)
04  Materialise RELION extraction project
05  relion_preprocess extraction
06  relion_reconstruct reconstruction
07  Write workflow_summary.json + workflow_summary.md

Scientific caveat
-----------------
This pipeline implements picker-stage backend selection and downstream
handoff.  It does NOT claim final structure determination.  The
reconstruction map produced here is a technical handoff artefact for
downstream QC, not a validated cryo-EM structure.

Usage example
-------------
python scripts/run_empiar10017_closed_loop_picker_to_reconstruction.py \\
    --pairs-csv /data/empiar10017_full84/pairs.csv \\
    --relion-autopick-bin /opt/relion/bin/relion_autopick \\
    --relion-bin-dir  /opt/relion/bin \\
    --out-root /scratch/empiar10017_closed_loop \\
    --limit 84 \\
    --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _run_cmd(
    cmd: List[str],
    log_path: Path,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Run a subprocess, tee stdout+stderr to a log file, return a result dict."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write("$ " + shlex.join(cmd) + "\n\n")
        fh.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = round(time.monotonic() - t0, 3)
    text = log_path.read_text(errors="ignore")
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "runtime_sec": elapsed,
        "log": str(log_path),
        "has_error_pattern": bool(
            re.search(r"ERROR|Traceback|Cannot|failed|not found|Exception|segmentation", text, re.I)
        ),
    }


def _count_particle_rows(particles_star: Path) -> int:
    """Count particle rows in a particles.star (lines containing .mrcs or .mrc)."""
    if not particles_star.is_file():
        return 0
    return sum(
        1 for line in particles_star.read_text(errors="ignore").splitlines()
        if ".mrcs" in line or ".mrc" in line
    )


# ---------------------------------------------------------------------------
# Stage 01 – record failed Topaz evidence
# ---------------------------------------------------------------------------


def stage01_record_topaz_evidence(
    out_dir: Path,
    topaz_evidence_json: Optional[Path],
    qc_threshold_f1: float,
) -> Dict[str, Any]:
    """Load or construct the failed-Topaz evidence dict and persist it."""
    if topaz_evidence_json is not None and topaz_evidence_json.exists():
        evidence = json.loads(topaz_evidence_json.read_text(encoding="utf-8"))
    else:
        # Construct a minimal canonical evidence dict matching the full84 run.
        evidence = {
            "selected_for_backend_execution": "topaz_t-4",
            "selection_metric": "picking_micro_f1",
            "selected_micro_f1": 0.031848,
            "qc_threshold_f1": qc_threshold_f1,
            "picking_qc_passed": False,
            "final_action": "FALLBACK_TO_ALTERNATIVE_PICKER_OR_RETRAIN",
            "source": "canonical_empiar10017_full84_topaz_failure",
        }

    # Ensure picking_qc_passed and qc_threshold_f1 are set consistently.
    evidence.setdefault("picking_qc_passed", False)
    evidence.setdefault("qc_threshold_f1", qc_threshold_f1)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "topaz_failed_evidence.json", evidence)
    print(f"[stage01] Topaz evidence recorded → {out_dir / 'topaz_failed_evidence.json'}", flush=True)
    return evidence


# ---------------------------------------------------------------------------
# Stage 02 – picker fallback selection
# ---------------------------------------------------------------------------


def stage02_picker_fallback_selection(
    evidence: Dict[str, Any],
    out_dir: Path,
    relion_autopick_bin: Optional[Path],
) -> Dict[str, Any]:
    """Run the picker fallback selection stage (dry-run for the RELION sub-run).

    Stage 03 will execute the actual RELION LoG run; here we only need to
    determine which picker is selected.
    """
    from agent.picker_fallback import BackendPaths, run_picker_fallback_stage

    paths = BackendPaths(
        relion_autopick=relion_autopick_bin,
        allow_system_path=(relion_autopick_bin is None),
    )

    # Run with dry_run=True so Stage 03 owns the real RELION execution.
    result = run_picker_fallback_stage(
        evidence=evidence,
        paths=paths,
        output_dir=out_dir,
        relion_run_config={"dry_run": True},
    )

    selected = (result.get("final") or {}).get("selected")
    final_action = (result.get("final") or {}).get("final_action")
    print(f"[stage02] Picker fallback selection: {selected} / {final_action}", flush=True)
    return result


# ---------------------------------------------------------------------------
# Stage 03 – RELION LoG rerun + F1 evaluation
# ---------------------------------------------------------------------------


def stage03_relion_log_rerun(
    pairs_csv: Path,
    out_dir: Path,
    relion_autopick_bin: Optional[Path],
    *,
    limit: Optional[int],
    particle_diameter_angstrom: float,
    pixel_size_angstrom: float,
    qc_threshold_f1: float,
    dry_run: bool,
    command_runner: Any = None,
) -> Dict[str, Any]:
    """Run RELION LoG autopick and evaluate F1 against ground-truth coordinates."""
    from agent.relion_log_autopick_runner import RelionLogAutopickConfig, run_relion_log_autopick

    cfg_kwargs: Dict[str, Any] = {
        "output_root": out_dir,
        "pairs_csv": pairs_csv,
        "dry_run": dry_run,
        "limit": limit,
        "particle_diameter_angstrom": particle_diameter_angstrom,
        "pixel_size_angstrom": pixel_size_angstrom,
    }
    if relion_autopick_bin is not None:
        cfg_kwargs["relion_autopick_bin"] = relion_autopick_bin
    if command_runner is not None:
        cfg_kwargs["command_runner"] = command_runner

    cfg = RelionLogAutopickConfig(**cfg_kwargs)
    result = run_relion_log_autopick(cfg)

    f1 = result.get("f1")
    predicted = result.get("predicted_count", 0)
    qc_passed = f1 is not None and f1 >= qc_threshold_f1
    result["qc_passed"] = qc_passed
    result["qc_threshold_f1"] = qc_threshold_f1

    # Persist enriched result.
    _write_json(out_dir / "relion_log_autopick_result.json", result)

    mode = result.get("mode", "?")
    print(
        f"[stage03] RELION LoG {mode}: F1={f1}, predicted={predicted}, "
        f"qc_passed={qc_passed}",
        flush=True,
    )
    return result


# ---------------------------------------------------------------------------
# Stage 04 – materialise RELION extraction project
# ---------------------------------------------------------------------------


def _write_micrographs_star_for_extraction(
    path: Path,
    mic_entries: List[Dict[str, str]],
    *,
    pixel_size: float = 1.0,
    voltage: float = 300.0,
    cs: float = 2.7,
    ac: float = 0.1,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "data_optics\n\n"
        "loop_\n"
        "_rlnOpticsGroupName #1\n"
        "_rlnOpticsGroup #2\n"
        "_rlnMicrographPixelSize #3\n"
        "_rlnVoltage #4\n"
        "_rlnSphericalAberration #5\n"
        "_rlnAmplitudeContrast #6\n"
        f"opticsGroup1 1 {pixel_size} {voltage} {cs} {ac}\n\n"
        "data_micrographs\n\n"
        "loop_\n"
        "_rlnOpticsGroup #1\n"
        "_rlnMicrographName #2\n"
    )
    rows = "\n".join(f"1 {e['rel_path']}" for e in mic_entries)
    path.write_text(header + rows + "\n", encoding="utf-8")


def _parse_coord_xy(coord_path: Path) -> List[Tuple[float, float]]:
    """Parse center coordinates from a converted_coords .coord file."""
    centers: List[Tuple[float, float]] = []
    for raw in coord_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue

        x_val: Optional[float] = None
        y_val: Optional[float] = None

        # Format: image_name x_coord y_coord score
        if len(parts) >= 3:
            try:
                x_val = float(parts[1])
                y_val = float(parts[2])
            except ValueError:
                x_val = None
                y_val = None

        # Generic whitespace/tab format with numeric x/y in first two columns.
        if x_val is None or y_val is None:
            try:
                x_val = float(parts[0])
                y_val = float(parts[1])
            except ValueError:
                continue

        centers.append((x_val, y_val))

    return centers


def _coord_to_box_rows(centers: List[Tuple[float, float]], extract_size: int) -> str:
    half = extract_size / 2.0
    rows: List[str] = []
    for x_center, y_center in centers:
        x_left = x_center - half
        y_top = y_center - half
        rows.append(f"{x_left:.6f} {y_top:.6f} {extract_size} {extract_size}")
    return "\n".join(rows) + ("\n" if rows else "")


def stage04_materialise_extract_project(
    pairs_csv: Path,
    relion_log_out: Path,
    out_dir: Path,
    *,
    limit: Optional[int],
    pixel_size_angstrom: float,
    dry_run: bool,
    extract_size: int,
) -> Dict[str, Any]:
    """Create a minimal RELION project tree ready for relion_preprocess.

    Micrograph MRC files are symlinked (not copied).
    Stage03 converted coordinates are converted to RELION .box files under:
    - Coords/<micrograph_id>.box
    - Coords/Micrographs/<micrograph_id>.box
    """
    import csv

    project = out_dir / "relion_project"
    mics_dir = project / "Micrographs"
    coords_dir = project / "Coords"
    coords_mics_dir = coords_dir / "Micrographs"
    mics_dir.mkdir(parents=True, exist_ok=True)
    coords_mics_dir.mkdir(parents=True, exist_ok=True)

    pairs: List[Dict[str, str]] = []
    with pairs_csv.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            mic_path = (
                row.get("local_mrc_path") or row.get("mrc_path")
                or row.get("micrograph_path") or row.get("image_path") or ""
            ).strip()
            if not mic_path:
                continue
            pairs.append({"micrograph_path": mic_path})
            if limit is not None and len(pairs) >= limit:
                break

    converted_coords_dir = relion_log_out / "converted_coords"

    mic_entries: List[Dict[str, str]] = []
    missing_mrc: List[str] = []
    missing_coord: List[str] = []
    n_converted_coord_files = 0
    n_box_files = 0

    for p in pairs:
        src_mrc = Path(p["micrograph_path"])
        stem = src_mrc.stem
        dst_mrc = mics_dir / src_mrc.name
        rel_path = f"Micrographs/{src_mrc.name}"

        if src_mrc.exists():
            if not dst_mrc.exists() and not dst_mrc.is_symlink():
                dst_mrc.symlink_to(src_mrc.resolve())
        else:
            missing_mrc.append(str(src_mrc))
            if not dst_mrc.exists():
                dst_mrc.write_text("placeholder\n", encoding="utf-8")

        coord_src = converted_coords_dir / f"{stem}.coord"
        box_text = ""
        if coord_src.exists():
            n_converted_coord_files += 1
            centers = _parse_coord_xy(coord_src)
            box_text = _coord_to_box_rows(centers, extract_size)
            (coords_dir / f"{stem}.box").write_text(box_text, encoding="utf-8")
            (coords_mics_dir / f"{stem}.box").write_text(box_text, encoding="utf-8")
            n_box_files += 2
        else:
            missing_coord.append(str(coord_src))

        mic_entries.append({"rel_path": rel_path, "stem": stem})

    micrographs_star = project / "micrographs.star"
    _write_micrographs_star_for_extraction(
        micrographs_star,
        mic_entries,
        pixel_size=pixel_size_angstrom,
    )

    manifest = {
        "project": str(project),
        "micrographs_star": str(micrographs_star),
        "n_micrographs": len(mic_entries),
        "n_converted_coord_files": n_converted_coord_files,
        "n_box_files": n_box_files,
        "n_missing_mrc": len(missing_mrc),
        "n_missing_coord": len(missing_coord),
        "coord_source": "relion_log_converted_coords",
        "dry_run": dry_run,
    }
    _write_json(out_dir / "extract_project_manifest.json", manifest)

    print(
        f"[stage04] Extraction project materialised: {len(mic_entries)} micrographs, "
        f"{len(missing_mrc)} missing MRC, {len(missing_coord)} missing COORD",
        flush=True,
    )
    return manifest


# ---------------------------------------------------------------------------
# Stage 05 – relion_preprocess extraction
# ---------------------------------------------------------------------------


def stage05_extract(
    extract_project_dir: Path,
    relion_bin_dir: Path,
    *,
    extract_size: int,
    bg_radius: int,
    dry_run: bool,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Run relion_preprocess to extract particles from the materialised project."""
    project = extract_project_dir / "relion_project"
    particles_star = project / "Extract" / "job002_log" / "particles.star"
    particles_star.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(relion_bin_dir / "relion_preprocess"),
        "--i", "micrographs.star",
        "--coord_dir", "Coords",
        "--coord_suffix", ".box",
        "--part_dir", "Extract/job002_log/Particles/",
        "--part_star", "Extract/job002_log/particles.star",
        "--extract",
        "--extract_size", str(extract_size),
        "--bg_radius", str(bg_radius),
        "--norm",
        "--white_dust", "-1",
        "--black_dust", "-1",
    ]

    log_path = extract_project_dir / "logs" / "relion_preprocess.log"
    _write_json(extract_project_dir / "extract_command.json", {"cmd": cmd})

    if dry_run:
        print(f"[stage05] DRY-RUN extraction command recorded → {log_path}", flush=True)
        return {
            "dry_run": True,
            "cmd": cmd,
            "returncode": None,
            "n_particles": 0,
            "particles_star": str(particles_star),
            "skipped": True,
        }

    result = _run_cmd(cmd, log_path, cwd=project, env=env)
    n_particles = _count_particle_rows(particles_star)
    stacks = sorted(particles_star.parent.rglob("*.mrcs"))

    extract_result = {
        "dry_run": False,
        "cmd": cmd,
        "returncode": result["returncode"],
        "runtime_sec": result["runtime_sec"],
        "n_particles": n_particles,
        "n_particle_stacks": len(stacks),
        "particles_star": str(particles_star),
        "log": result["log"],
    }
    _write_json(extract_project_dir / "extract_result.json", extract_result)
    print(
        f"[stage05] Extraction: returncode={result['returncode']}, "
        f"particles={n_particles}, stacks={len(stacks)}",
        flush=True,
    )
    return extract_result


# ---------------------------------------------------------------------------
# Stage 06 – relion_reconstruct reconstruction
# ---------------------------------------------------------------------------


def stage06_reconstruct(
    extract_project_dir: Path,
    recon_out_dir: Path,
    relion_bin_dir: Path,
    *,
    n_particles: int,
    dry_run: bool,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Run relion_reconstruct on the extracted particles."""
    project = extract_project_dir / "relion_project"
    recon_out_dir.mkdir(parents=True, exist_ok=True)
    recon_map = recon_out_dir / "reconstruct_from_relion_log.mrc"

    cmd = [
        str(relion_bin_dir / "relion_reconstruct"),
        "--i", "Extract/job002_log/particles.star",
        "--o", str(recon_map),
    ]

    log_path = recon_out_dir / "relion_reconstruct.log"
    _write_json(recon_out_dir / "reconstruct_command.json", {"cmd": cmd})

    if dry_run:
        print(f"[stage06] DRY-RUN reconstruction command recorded → {log_path}", flush=True)
        return {
            "dry_run": True,
            "cmd": cmd,
            "returncode": None,
            "map_exists": False,
            "map_path": str(recon_map),
            "skipped": True,
        }

    if n_particles == 0:
        reason = "No extracted particles; reconstruction skipped."
        print(f"[stage06] {reason}", flush=True)
        return {
            "dry_run": False,
            "skipped": True,
            "reason": reason,
            "map_path": str(recon_map),
            "map_exists": False,
        }

    result = _run_cmd(cmd, log_path, cwd=project, env=env)
    map_exists = recon_map.is_file() and recon_map.stat().st_size > 0

    recon_result = {
        "dry_run": False,
        "cmd": cmd,
        "returncode": result["returncode"],
        "runtime_sec": result["runtime_sec"],
        "map_path": str(recon_map),
        "map_exists": map_exists,
        "map_size_bytes": recon_map.stat().st_size if recon_map.is_file() else 0,
        "log": result["log"],
    }
    _write_json(recon_out_dir / "reconstruct_result.json", recon_result)
    print(
        f"[stage06] Reconstruction: returncode={result['returncode']}, "
        f"map_exists={map_exists}",
        flush=True,
    )
    return recon_result


# ---------------------------------------------------------------------------
# Stages 07–11 – downstream scientific gates
# ---------------------------------------------------------------------------


def _stage07_class2d(
    out_dir: Path,
    *,
    extract_result: Dict[str, Any],
    relion_bin_dir: Optional[Path],
    mode: str,
    particle_diameter_angstrom: float,
    full_dataset: bool,
    skip_large: bool,
) -> Dict[str, Any]:
    from agent.downstream_gates import stage07_class2d_gate
    effective_mode = mode
    if mode == "real" and skip_large:
        effective_mode = "plan-only"
    particles_star = extract_result.get("particles_star")
    return stage07_class2d_gate(
        out_dir,
        particles_star=Path(particles_star) if particles_star else None,
        n_particles=int(extract_result.get("n_particles", 0) or 0),
        relion_bin_dir=relion_bin_dir,
        mode=effective_mode,
        particle_diameter_angstrom=particle_diameter_angstrom,
        full_dataset=full_dataset,
    )


def _stage08_refine3d(
    out_dir: Path,
    *,
    extract_result: Dict[str, Any],
    recon_result: Dict[str, Any],
    relion_bin_dir: Optional[Path],
    mode: str,
    particle_diameter_angstrom: float,
    skip_large: bool,
) -> Dict[str, Any]:
    from agent.downstream_gates import stage08_refine3d_gate
    effective_mode = mode
    if mode == "real" and skip_large:
        effective_mode = "plan-only"
    map_path = recon_result.get("map_path")
    particles_star = extract_result.get("particles_star")
    return stage08_refine3d_gate(
        out_dir,
        map_path=Path(map_path) if map_path else None,
        map_exists=bool(recon_result.get("map_exists")),
        particles_star=Path(particles_star) if particles_star else None,
        n_particles=int(extract_result.get("n_particles", 0) or 0),
        relion_bin_dir=relion_bin_dir,
        mode=effective_mode,
        particle_diameter_angstrom=particle_diameter_angstrom,
    )


def _stage09_postprocess(
    out_dir: Path,
    *,
    recon_result: Dict[str, Any],
    half_map_1: Optional[Path],
    half_map_2: Optional[Path],
    mask: Optional[Path],
    relion_bin_dir: Optional[Path],
    mode: str,
    resolution_threshold: float,
) -> Dict[str, Any]:
    from agent.map_quality_gate import stage09_postprocess_gate
    map_path = recon_result.get("map_path")
    return stage09_postprocess_gate(
        out_dir,
        map_path=Path(map_path) if map_path else None,
        map_exists=bool(recon_result.get("map_exists")),
        half_map_1=half_map_1,
        half_map_2=half_map_2,
        mask=mask,
        relion_bin_dir=relion_bin_dir,
        mode=mode,
        resolution_threshold=resolution_threshold,
    )


def _stage10_map2pdb(
    out_dir: Path,
    *,
    recon_result: Dict[str, Any],
    map_quality_gate: Dict[str, Any],
    sequence: Optional[Path],
    backends: List[str],
    mode: str,
    require_fsc: bool,
    allow_without_fsc: bool,
) -> Dict[str, Any]:
    from agent.map2pdb_routing import stage10_map2pdb_routing
    pp_map = map_quality_gate.get("postprocess_map_path")
    if pp_map:
        map_path: Optional[Path] = Path(pp_map)
        map_source = "postprocess"
        map_exists = True
    else:
        raw = recon_result.get("map_path")
        map_path = Path(raw) if raw else None
        map_source = "reconstruct"
        map_exists = bool(recon_result.get("map_exists"))
    return stage10_map2pdb_routing(
        out_dir,
        map_path=map_path,
        map_exists=map_exists,
        map_source=map_source,
        map_quality_gate=map_quality_gate,
        sequence=sequence,
        backends=backends,
        mode=mode,
        require_fsc=require_fsc,
        allow_without_fsc=allow_without_fsc,
    )


def _stage11_model_validation(
    out_dir: Path,
    *,
    routing: Dict[str, Any],
    recon_result: Dict[str, Any],
    sequence: Optional[Path],
) -> Dict[str, Any]:
    from agent.model_validation_gate import stage11_model_validation_gate
    model_path = routing.get("model_output_path")
    map_path = recon_result.get("map_path")
    return stage11_model_validation_gate(
        out_dir,
        model_path=Path(model_path) if model_path else None,
        map_path=Path(map_path) if map_path else None,
        sequence=sequence,
    )


def _compute_final_readiness(
    *,
    recon_result: Dict[str, Any],
    map_quality_gate: Dict[str, Any],
    routing: Dict[str, Any],
    model_validation: Dict[str, Any],
) -> str:
    """Conservative scientific-readiness verdict for the workflow."""
    if not recon_result.get("map_exists"):
        return "WORKFLOW_HANDOFF_ONLY"
    if model_validation.get("status") == "planned" and \
            model_validation.get("model_exists"):
        return "MODEL_VALIDATION_REQUIRED"
    if routing.get("real_run_allowed"):
        return "MAP2PDB_REAL_ALLOWED"
    if routing.get("decision") == "ALLOW_MAP2PDB_DRY_RUN":
        return "MAP2PDB_READY_DRY_RUN_ONLY"
    if map_quality_gate.get("map_quality_status") == "unvalidated":
        return "MAP_QC_REQUIRED"
    return "WORKFLOW_HANDOFF_ONLY"


# ---------------------------------------------------------------------------
# Stage 12 – workflow summary
# ---------------------------------------------------------------------------


_SUMMARY_MD_TEMPLATE = """\
# EMPIAR-10017 Closed-Loop Picker → Reconstruction Workflow

## Scientific caveat

This pipeline implements **autonomous picker-to-reconstruction workflow**
with **closed-loop picker fallback and downstream handoff**, towards
autonomous cryo-EM structure determination.  It does NOT claim final
structure determination.  The reconstruction map produced here is a
technical handoff artefact; map-to-model routing is stop-gated when map
QC evidence (FSC / postprocess) is missing.

## Stage status

| stage | name | status |
|---|---|---|
| 01 | Record Topaz failure evidence | {s01} |
| 02 | Picker fallback selection | {s02} |
| 03 | RELION LoG rerun + F1 | {s03} |
| 04 | Materialise extraction project | {s04} |
| 05 | relion_preprocess extraction | {s05} |
| 06 | relion_reconstruct reconstruction | {s06} |
| 07 | Class2D plan / gate | {s07} |
| 08 | Refine3D plan / gate | {s08} |
| 09 | Postprocess / FSC gate | {s09} |
| 10 | Map-to-PDB routing | {s10} |
| 11 | Model validation | {s11} |

## Final readiness

**{final_readiness}**

## Key metrics

| metric | value |
|---|---|
| Topaz F1 (failed) | {topaz_f1} |
| RELION LoG F1 | {relion_f1} |
| QC threshold F1 | {qc_threshold} |
| QC passed | {qc_passed} |
| Predicted particles | {predicted} |
| GT particles | {gt_count} |
| Matched particles | {matched} |
| Extracted particles | {extracted} |
| Reconstruction map | {map_exists} |

## Output locations

- Picker selection: `01_picker_selection/`
- RELION LoG rerun: `02_relion_log_rerun/`
- Extraction project: `03_extract_project/`
- Reconstruction: `04_reconstruction/`
- Class2D plan: `05_class2d/`
- Refine3D plan: `06_refine3d/`
- Postprocess / FSC: `07_postprocess/`
- Map-to-PDB routing: `08_map2pdb_routing/`
- Model validation: `09_model_validation/`
- Reports: `reports/`
"""


def stage07_write_summary(
    out_root: Path,
    *,
    evidence: Dict[str, Any],
    selection_result: Dict[str, Any],
    relion_result: Dict[str, Any],
    extract_manifest: Dict[str, Any],
    extract_result: Dict[str, Any],
    recon_result: Dict[str, Any],
    pairs_csv: Path,
    args_dict: Dict[str, Any],
    class2d_result: Optional[Dict[str, Any]] = None,
    refine3d_result: Optional[Dict[str, Any]] = None,
    postprocess_result: Optional[Dict[str, Any]] = None,
    map_quality_gate: Optional[Dict[str, Any]] = None,
    map2pdb_routing: Optional[Dict[str, Any]] = None,
    model_validation: Optional[Dict[str, Any]] = None,
    final_readiness: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist workflow_summary.json and workflow_summary.md."""

    def _stage_status(result: Dict[str, Any], key: str = "returncode") -> str:
        if result.get("skipped"):
            return "skipped"
        if result.get("dry_run"):
            return "dry_run"
        rc = result.get(key)
        if rc is None:
            return "ok"
        return "ok" if rc == 0 else f"failed(rc={rc})"

    def _gate_status(g: Optional[Dict[str, Any]]) -> str:
        if not g:
            return "skipped"
        return str(g.get("status") or "ok")

    s01 = "ok"
    s02 = "skipped" if selection_result.get("skipped") else "ok"
    s03 = _stage_status(relion_result)
    s04 = "ok"
    s05 = _stage_status(extract_result)
    s06 = _stage_status(recon_result)
    s07 = _gate_status(class2d_result)
    s08 = _gate_status(refine3d_result)
    if postprocess_result and isinstance(postprocess_result.get("plan"), dict):
        s09 = _gate_status(postprocess_result["plan"])
    else:
        s09 = _gate_status(postprocess_result)
    s10 = _gate_status(map2pdb_routing) if map2pdb_routing else "skipped"
    s11 = _gate_status(model_validation)

    stage02_selected = (selection_result.get("final") or {}).get("selected")
    stage02_final_action = (selection_result.get("final") or {}).get("final_action")
    relion_qc_passed = bool(relion_result.get("qc_passed"))
    final_selected_picker = "relion_log_autopick" if relion_qc_passed else stage02_selected
    final_action = "RUN_ALTERNATIVE_PICKER" if relion_qc_passed else stage02_final_action

    summary: Dict[str, Any] = {
        "pipeline": "empiar10017_closed_loop_picker_to_reconstruction",
        "args": args_dict,
        "stage_status": {
            "01_topaz_evidence": s01,
            "02_picker_fallback": s02,
            "03_relion_log_rerun": s03,
            "04_extract_project": s04,
            "05_extraction": s05,
            "06_reconstruction": s06,
            "07_class2d": s07,
            "08_refine3d": s08,
            "09_postprocess": s09,
            "10_map2pdb_routing": s10,
            "11_model_validation": s11,
        },
        "picker_selection": {
            "topaz_f1": evidence.get("selected_micro_f1"),
            "qc_threshold_f1": evidence.get("qc_threshold_f1"),
            "stage02_selected_picker": stage02_selected,
            "stage02_final_action": stage02_final_action,
            "selected_picker": final_selected_picker,
            "final_action": final_action,
        },
        "relion_log_rerun": {
            "mode": relion_result.get("mode"),
            "f1": relion_result.get("f1"),
            "precision": relion_result.get("precision"),
            "recall": relion_result.get("recall"),
            "predicted_count": relion_result.get("predicted_count", 0),
            "gt_count": relion_result.get("gt_count", 0),
            "matched_count": relion_result.get("matched_count", 0),
            "qc_passed": relion_result.get("qc_passed"),
        },
        "extraction": {
            "n_particles": extract_result.get("n_particles", 0),
            "particles_star": extract_result.get("particles_star"),
            "returncode": extract_result.get("returncode"),
        },
        "reconstruction": {
            "map_exists": recon_result.get("map_exists", False),
            "map_path": recon_result.get("map_path"),
            "map_size_bytes": recon_result.get("map_size_bytes"),
        },
        "artifacts": {
            "01_picker_selection": str(out_root / "01_picker_selection"),
            "02_relion_log_rerun": str(out_root / "02_relion_log_rerun"),
            "03_extract_project": str(out_root / "03_extract_project"),
            "04_reconstruction": str(out_root / "04_reconstruction"),
            "05_class2d": str(out_root / "05_class2d"),
            "06_refine3d": str(out_root / "06_refine3d"),
            "07_postprocess": str(out_root / "07_postprocess"),
            "08_map2pdb_routing": str(out_root / "08_map2pdb_routing"),
            "09_model_validation": str(out_root / "09_model_validation"),
            "reports": str(out_root / "reports"),
        },
        "class2d": class2d_result or {"status": "skipped"},
        "refine3d": refine3d_result or {"status": "skipped"},
        "postprocess": (postprocess_result or {}).get("plan") if (
            postprocess_result and isinstance(postprocess_result.get("plan"), dict)
        ) else (postprocess_result or {"status": "skipped"}),
        "map_quality_gate": map_quality_gate or {
            "map_quality_status": "unvalidated",
            "allow_map2pdb_real": False,
            "decision": "STOP_MAP_QC_REQUIRED",
        },
        "map2pdb_routing": map2pdb_routing or {"status": "skipped"},
        "model_validation": model_validation or {
            "status": "skipped",
            "decision": "SKIP_MODEL_VALIDATION_NO_MODEL",
        },
        "final_readiness": final_readiness or "WORKFLOW_HANDOFF_ONLY",
        "scientific_caveat": (
            "Autonomous picker-to-reconstruction workflow with closed-loop "
            "picker fallback and downstream handoff.  Map-to-model routing "
            "is stop-gated when map QC evidence is missing.  This is "
            "picker-stage backend selection and downstream handoff, "
            "not final structure determination."
        ),
    }

    _write_json(out_root / "workflow_summary.json", summary)

    md = _SUMMARY_MD_TEMPLATE.format(
        s01=s01, s02=s02, s03=s03, s04=s04, s05=s05, s06=s06,
        s07=s07, s08=s08, s09=s09, s10=s10, s11=s11,
        final_readiness=summary["final_readiness"],
        topaz_f1=evidence.get("selected_micro_f1"),
        relion_f1=relion_result.get("f1"),
        qc_threshold=evidence.get("qc_threshold_f1"),
        qc_passed=relion_result.get("qc_passed"),
        predicted=relion_result.get("predicted_count", 0),
        gt_count=relion_result.get("gt_count", 0),
        matched=relion_result.get("matched_count", 0),
        extracted=extract_result.get("n_particles", 0),
        map_exists=recon_result.get("map_exists", False),
    )
    _write_text(out_root / "workflow_summary.md", md)
    print(f"[stage07] Workflow summary written → {out_root / 'workflow_summary.json'}", flush=True)
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_env(relion_bin_dir: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = str(relion_bin_dir) + os.pathsep + env.get("PATH", "")
    lib = relion_bin_dir.parent / "lib"
    env["LD_LIBRARY_PATH"] = str(lib) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Closed-loop picker fallback → RELION extraction → reconstruction pipeline."
    )
    ap.add_argument("--pairs-csv", required=True, type=Path,
                    help="CSV with micrograph_path / gt_coord_path columns.")
    ap.add_argument("--topaz-failed-evidence-json", type=Path, default=None,
                    help="Optional JSON with failed Topaz QC evidence.  "
                         "Constructed from canonical full84 values when absent.")
    ap.add_argument("--relion-autopick-bin", type=Path, default=None,
                    help="Path to relion_autopick binary.")
    ap.add_argument("--relion-bin-dir", type=Path, default=None,
                    help="Directory containing relion_preprocess and relion_reconstruct.")
    ap.add_argument("--out-root", required=True, type=Path,
                    help="Root output directory.")
    ap.add_argument("--limit", type=int, default=84,
                    help="Max micrographs to process (default 84).")
    ap.add_argument("--qc-threshold-f1", type=float, default=0.2,
                    help="F1 threshold for picking QC (default 0.2).")
    ap.add_argument("--particle-diameter-angstrom", type=float, default=120.0,
                    help="Particle diameter hint in Å (default 120).")
    ap.add_argument("--pixel-size-angstrom", type=float, default=1.0,
                    help="Pixel size in Å/px (default 1).")
    ap.add_argument("--extract-size", type=int, default=256,
                    help="Extraction box size in pixels (default 256).")
    ap.add_argument("--bg-radius", type=int, default=100,
                    help="Background radius for normalisation (default 100).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build and record commands but do not execute them.")

    # Downstream-gate CLI options.
    _gate_choices = ["none", "plan-only", "dry-run", "real"]
    ap.add_argument("--downstream-mode", choices=_gate_choices, default="plan-only",
                    help="Default mode for all downstream gates (Stages 07-11).")
    ap.add_argument("--class2d-mode", choices=_gate_choices, default=None,
                    help="Override Class2D gate mode (Stage 07).")
    ap.add_argument("--refine3d-mode", choices=_gate_choices, default=None,
                    help="Override 3D refinement gate mode (Stage 08).")
    ap.add_argument("--postprocess-mode", choices=_gate_choices, default=None,
                    help="Override postprocess/FSC gate mode (Stage 09).")
    ap.add_argument("--map2pdb-mode", choices=_gate_choices, default="dry-run",
                    help="Map-to-PDB routing mode (Stage 10).")
    ap.add_argument("--sequence", type=Path, default=None,
                    help="Optional FASTA path for map2pdb backends.")
    ap.add_argument("--mask", type=Path, default=None,
                    help="Optional mask path for postprocess.")
    ap.add_argument("--half-map-1", type=Path, default=None,
                    help="Optional half-map path 1.")
    ap.add_argument("--half-map-2", type=Path, default=None,
                    help="Optional half-map path 2.")
    ap.add_argument("--resolution-threshold", type=float, default=4.0,
                    help="Conservative resolution threshold (Å) to allow real "
                         "map-to-model execution.")
    ap.add_argument("--require-fsc-for-map2pdb", action="store_true", default=True,
                    help="Require FSC evidence before map-to-model routing.")
    ap.add_argument("--allow-map2pdb-without-fsc", action="store_true", default=False,
                    help="Allow dry-run map2pdb routing even without FSC.")
    ap.add_argument("--model-building-backends", type=str,
                    default="modelangelo,emodelx,mica",
                    help="Comma-separated candidate model-building backends.")
    ap.add_argument("--skip-large-real-runs", action="store_true", default=False,
                    help="Downgrade real-run gates to plan-only.")

    args = ap.parse_args(argv)
    class2d_mode = args.class2d_mode or args.downstream_mode
    refine3d_mode = args.refine3d_mode or args.downstream_mode
    postprocess_mode = args.postprocess_mode or args.downstream_mode
    backends_list = [b.strip() for b in args.model_building_backends.split(",") if b.strip()]

    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    dir_01 = out_root / "01_picker_selection"
    dir_02 = out_root / "02_relion_log_rerun"
    dir_03 = out_root / "03_extract_project"
    dir_04 = out_root / "04_reconstruction"
    dir_05 = out_root / "05_class2d"
    dir_06 = out_root / "06_refine3d"
    dir_07 = out_root / "07_postprocess"
    dir_08 = out_root / "08_map2pdb_routing"
    dir_09 = out_root / "09_model_validation"
    reports_dir = out_root / "reports"
    for d in (dir_01, dir_02, dir_03, dir_04, dir_05, dir_06, dir_07, dir_08,
              dir_09, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    relion_bin_dir = args.relion_bin_dir or Path(".")
    env = _build_env(relion_bin_dir)

    args_dict: Dict[str, Any] = {
        "pairs_csv": str(args.pairs_csv),
        "topaz_failed_evidence_json": str(args.topaz_failed_evidence_json),
        "relion_autopick_bin": str(args.relion_autopick_bin),
        "relion_bin_dir": str(relion_bin_dir),
        "out_root": str(out_root),
        "limit": args.limit,
        "qc_threshold_f1": args.qc_threshold_f1,
        "particle_diameter_angstrom": args.particle_diameter_angstrom,
        "pixel_size_angstrom": args.pixel_size_angstrom,
        "extract_size": args.extract_size,
        "bg_radius": args.bg_radius,
        "dry_run": args.dry_run,
    }

    print(f"[pipeline] out_root={out_root}", flush=True)

    # -- Stage 01 ----------------------------------------------------------------
    evidence = stage01_record_topaz_evidence(
        dir_01,
        args.topaz_failed_evidence_json,
        qc_threshold_f1=args.qc_threshold_f1,
    )

    # -- Stage 02 ----------------------------------------------------------------
    selection_result = stage02_picker_fallback_selection(
        evidence=evidence,
        out_dir=dir_01,
        relion_autopick_bin=args.relion_autopick_bin,
    )

    # -- Stage 03 ----------------------------------------------------------------
    relion_result = stage03_relion_log_rerun(
        pairs_csv=args.pairs_csv,
        out_dir=dir_02,
        relion_autopick_bin=args.relion_autopick_bin,
        limit=args.limit,
        particle_diameter_angstrom=args.particle_diameter_angstrom,
        pixel_size_angstrom=args.pixel_size_angstrom,
        qc_threshold_f1=args.qc_threshold_f1,
        dry_run=args.dry_run,
    )

    # -- Stage 04 ----------------------------------------------------------------
    extract_manifest = stage04_materialise_extract_project(
        pairs_csv=args.pairs_csv,
        relion_log_out=dir_02,
        out_dir=dir_03,
        limit=args.limit,
        pixel_size_angstrom=args.pixel_size_angstrom,
        dry_run=args.dry_run,
        extract_size=args.extract_size,
    )

    # -- Stage 05 ----------------------------------------------------------------
    extract_result = stage05_extract(
        extract_project_dir=dir_03,
        relion_bin_dir=relion_bin_dir,
        extract_size=args.extract_size,
        bg_radius=args.bg_radius,
        dry_run=args.dry_run,
        env=env,
    )

    # -- Stage 06 ----------------------------------------------------------------
    recon_result = stage06_reconstruct(
        extract_project_dir=dir_03,
        recon_out_dir=dir_04,
        relion_bin_dir=relion_bin_dir,
        n_particles=extract_result.get("n_particles", 0),
        dry_run=args.dry_run,
        env=env,
    )

    # -- Stage 07: Class2D ------------------------------------------------------
    class2d_result = _stage07_class2d(
        dir_05,
        extract_result=extract_result,
        relion_bin_dir=relion_bin_dir,
        mode=class2d_mode,
        particle_diameter_angstrom=args.particle_diameter_angstrom,
        full_dataset=(args.limit is None or args.limit >= 84),
        skip_large=args.skip_large_real_runs,
    )
    print(f"[stage07] Class2D: status={class2d_result.get('status')}, "
          f"decision={class2d_result.get('decision')}", flush=True)

    # -- Stage 08: Refine3D -----------------------------------------------------
    refine3d_result = _stage08_refine3d(
        dir_06,
        extract_result=extract_result,
        recon_result=recon_result,
        relion_bin_dir=relion_bin_dir,
        mode=refine3d_mode,
        particle_diameter_angstrom=args.particle_diameter_angstrom,
        skip_large=args.skip_large_real_runs,
    )
    print(f"[stage08] Refine3D: status={refine3d_result.get('status')}, "
          f"decision={refine3d_result.get('decision')}", flush=True)

    # -- Stage 09: Postprocess / FSC --------------------------------------------
    postprocess_bundle = _stage09_postprocess(
        dir_07,
        recon_result=recon_result,
        half_map_1=args.half_map_1,
        half_map_2=args.half_map_2,
        mask=args.mask,
        relion_bin_dir=relion_bin_dir,
        mode=postprocess_mode,
        resolution_threshold=args.resolution_threshold,
    )
    map_quality_gate = postprocess_bundle["map_quality_gate"]
    print(f"[stage09] Postprocess: status={postprocess_bundle['plan'].get('status')}, "
          f"map_quality={map_quality_gate.get('map_quality_status')}, "
          f"decision={map_quality_gate.get('decision')}", flush=True)

    # -- Stage 10: Map-to-PDB routing -------------------------------------------
    routing = _stage10_map2pdb(
        dir_08,
        recon_result=recon_result,
        map_quality_gate=map_quality_gate,
        sequence=args.sequence,
        backends=backends_list,
        mode=args.map2pdb_mode,
        require_fsc=args.require_fsc_for_map2pdb,
        allow_without_fsc=args.allow_map2pdb_without_fsc,
    )
    print(f"[stage10] Map2PDB routing: backend={routing.get('selected_backend')}, "
          f"decision={routing.get('decision')}", flush=True)

    # -- Stage 11: Model validation ---------------------------------------------
    model_validation = _stage11_model_validation(
        dir_09,
        routing=routing,
        recon_result=recon_result,
        sequence=args.sequence,
    )
    print(f"[stage11] Model validation: status={model_validation.get('status')}, "
          f"decision={model_validation.get('decision')}", flush=True)

    final_readiness = _compute_final_readiness(
        recon_result=recon_result,
        map_quality_gate=map_quality_gate,
        routing=routing,
        model_validation=model_validation,
    )

    # -- Stage 12: workflow summary ---------------------------------------------
    summary = stage07_write_summary(
        out_root=out_root,
        evidence=evidence,
        selection_result=selection_result,
        relion_result=relion_result,
        extract_manifest=extract_manifest,
        extract_result=extract_result,
        recon_result=recon_result,
        pairs_csv=args.pairs_csv,
        args_dict=args_dict,
        class2d_result=class2d_result,
        refine3d_result=refine3d_result,
        postprocess_result=postprocess_bundle,
        map_quality_gate=map_quality_gate,
        map2pdb_routing=routing,
        model_validation=model_validation,
        final_readiness=final_readiness,
    )

    print(json.dumps({
        "status": "ok",
        "out_root": str(out_root),
        "workflow_summary": str(out_root / "workflow_summary.json"),
        "relion_log_f1": summary["relion_log_rerun"]["f1"],
        "qc_passed": summary["relion_log_rerun"]["qc_passed"],
        "reconstruction_map_exists": summary["reconstruction"]["map_exists"],
        "final_readiness": summary["final_readiness"],
        "dry_run": args.dry_run,
    }, indent=2))


if __name__ == "__main__":
    main()
