"""Helpers for preparing a small CryoPPP subset for Topaz smoke runs."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.data_prep.image_conversion import convert_jpg_to_png


def _is_numeric(text: str) -> bool:
    try:
        float(text)
    except (TypeError, ValueError):
        return False
    return True


def _best_effort_metadata_value(meta_dir: Path, keys: List[str]) -> Optional[float]:
    if not meta_dir.is_dir():
        return None
    for path in sorted(meta_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        for key in keys:
            value = data.get(key)
            if _is_numeric(str(value)):
                return float(value)
    return None


def write_micrographs_star(micrograph_paths: List[Path], star_path: Path) -> Path:
    star_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "data_micrographs",
        "",
        "loop_",
        "_rlnMicrographName #1",
    ]
    lines.extend(str(p) for p in micrograph_paths)
    lines.append("")
    star_path.write_text("\n".join(lines), encoding="utf-8")
    return star_path


def prepare_cryoppp_topaz_smoke(
    *,
    dataset_root: Path,
    run_root: Path,
    dataset_id: str,
    sample_size: int = 32,
    random_seed: int = 42,
    sampling_strategy: str = "random",
    use_symlink: bool = True,
    pixel_size_angstrom_override: Optional[float] = None,
    particle_diameter_px_override: Optional[float] = None,
    require_gt_overlap: bool = True,
) -> Dict[str, Any]:
    """Prepare minimal inputs for one real Topaz smoke run."""
    dataset_root = dataset_root.resolve()
    run_root = run_root.resolve()
    src_micrographs_dir = dataset_root / "motion_corrected_micrographs"
    if not src_micrographs_dir.is_dir():
        src_micrographs_dir = dataset_root / "micrographs"
    src_gt_dir = dataset_root / "ground_truth" / "particle_coordinates"
    src_meta_dir = dataset_root / "metadata"

    if not src_micrographs_dir.is_dir():
        raise FileNotFoundError(
            f"Missing micrograph directory: {dataset_root / 'motion_corrected_micrographs'} "
            f"or {dataset_root / 'micrographs'}"
        )
    if not src_gt_dir.is_dir():
        raise FileNotFoundError(f"Missing ground_truth/particle_coordinates: {src_gt_dir}")

    all_micrographs = sorted(
        [p for p in src_micrographs_dir.iterdir() if p.is_file()],
        key=lambda p: p.name,
    )
    if not all_micrographs:
        raise ValueError("No micrographs found for smoke prep")

    gt_files_by_stem: Dict[str, Path] = {}
    for pattern in ("*.csv", "*.txt", "*.star"):
        for gt_file in sorted(src_gt_dir.glob(pattern)):
            if gt_file.is_file():
                gt_files_by_stem[gt_file.stem] = gt_file
    overlap_micrographs = [mg for mg in all_micrographs if mg.stem in gt_files_by_stem]
    requested_sample_size = int(sample_size)
    sampling_pool = overlap_micrographs if require_gt_overlap else all_micrographs
    if not sampling_pool:
        raise ValueError("No eligible micrographs found for smoke prep sampling")
    n = max(1, min(requested_sample_size, len(sampling_pool)))
    if sampling_strategy == "first_n":
        selected = sampling_pool[:n]
    else:
        rng = random.Random(random_seed)
        selected = sorted(rng.sample(sampling_pool, n), key=lambda p: p.name)
        sampling_strategy = "random"

    smoke_root = run_root / dataset_id
    data_root = smoke_root / "data" / dataset_id
    relion_inputs = smoke_root / "relion_inputs"
    gt_out_dir = relion_inputs / "gt_coords"
    outputs_dir = smoke_root / "outputs"
    logs_dir = smoke_root / "logs"
    reports_dir = outputs_dir / "reports"
    eval_dir = outputs_dir / "eval"
    topaz_out_dir = outputs_dir / "topaz_pick"
    converted_dir = data_root / "converted_micrographs"

    for path in (data_root, relion_inputs, gt_out_dir, topaz_out_dir, eval_dir, reports_dir, logs_dir, converted_dir):
        path.mkdir(parents=True, exist_ok=True)

    selected_out_micrographs: List[str] = []
    gt_coordinate_files: List[str] = []
    matched_gt_coordinate_files: List[str] = []
    converted_micrographs: List[str] = []
    micrograph_subdir_name = src_micrographs_dir.name
    source_suffixes = {mg.suffix.lower() for mg in selected if mg.suffix}
    source_format = "mixed" if len(source_suffixes) > 1 else (next(iter(source_suffixes)).lstrip(".") if source_suffixes else "unknown")
    conversion_applied = False
    for mg in selected:
        dst_mg = data_root / micrograph_subdir_name / mg.name
        dst_mg.parent.mkdir(parents=True, exist_ok=True)
        if dst_mg.exists():
            dst_mg.unlink()
        if use_symlink:
            dst_mg.symlink_to(mg)
        else:
            shutil.copy2(mg, dst_mg)
        runtime_path = dst_mg
        if mg.suffix.lower() in {".jpg", ".jpeg"}:
            converted_path = converted_dir / f"{mg.stem}.png"
            convert_jpg_to_png(dst_mg, converted_path)
            runtime_path = converted_path
            conversion_applied = True
            converted_micrographs.append(str(converted_path))
        selected_out_micrographs.append(str(runtime_path))

        gt_match = gt_files_by_stem.get(mg.stem)
        if gt_match is not None and gt_match.is_file():
            dst_gt = gt_out_dir / gt_match.name
            if dst_gt.exists():
                dst_gt.unlink()
            if use_symlink:
                dst_gt.symlink_to(gt_match)
            else:
                shutil.copy2(gt_match, dst_gt)
            gt_coordinate_files.append(str(dst_gt))
            matched_gt_coordinate_files.append(str(dst_gt))

    star_path = write_micrographs_star(
        [Path(x) for x in selected_out_micrographs],
        relion_inputs / "micrographs.star",
    )

    pixel_size = (
        float(pixel_size_angstrom_override)
        if pixel_size_angstrom_override is not None
        else _best_effort_metadata_value(src_meta_dir, ["pixel_size_angstrom", "pixel_size", "apix"])
    )
    diameter_px = (
        float(particle_diameter_px_override)
        if particle_diameter_px_override is not None
        else _best_effort_metadata_value(src_meta_dir, ["particle_diameter_px", "diameter_px"])
    )
    diameter_angstrom = None
    if pixel_size is not None and diameter_px is not None:
        diameter_angstrom = pixel_size * diameter_px

    manifest = {
        "dataset_id": dataset_id,
        "dataset_root": str(dataset_root),
        "sampling_strategy": sampling_strategy,
        "sample_size": len(selected_out_micrographs),
        "requested_sample_size": requested_sample_size,
        "effective_sample_size": len(selected_out_micrographs),
        "gt_overlap_count": len(overlap_micrographs),
        "gt_covered_sample": bool(require_gt_overlap),
        "random_seed": int(random_seed),
        "selected_micrographs": selected_out_micrographs,
        "gt_coordinate_files": gt_coordinate_files,
        "matched_gt_coordinate_files": matched_gt_coordinate_files,
        "micrograph_source_format": source_format,
        "micrograph_runtime_format": "png" if conversion_applied else source_format,
        "converted_micrographs": converted_micrographs,
        "conversion_applied": conversion_applied,
        "output_paths": {
            "run_root": str(smoke_root),
            "relion_inputs": str(relion_inputs),
            "micrographs_star_path": str(star_path),
            "gt_coords_dir": str(gt_out_dir),
            "topaz_output_dir": str(topaz_out_dir),
            "eval_dir": str(eval_dir),
            "reports_dir": str(reports_dir),
            "logs_dir": str(logs_dir),
        },
        "particle_diameter_px": diameter_px,
        "pixel_size_angstrom": pixel_size,
        "estimated_particle_diameter_angstrom": diameter_angstrom,
    }
    manifest_path = smoke_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest
