"""Stage 09 – Postprocess / FSC validation gate.

Builds a relion_postprocess plan and computes a ``map_quality_gate.json``
record describing whether the current map is scientifically validated
enough to allow real map-to-model execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


VALID_MODES = ("none", "plan-only", "dry-run", "real")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _normalise_mode(mode: str) -> str:
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode {mode!r}; expected one of {VALID_MODES}")
    return mode


def _md_table(rows: Dict[str, Any]) -> str:
    out = ["| key | value |", "|---|---|"]
    for k, v in rows.items():
        out.append(f"| {k} | {v} |")
    return "\n".join(out) + "\n"


def stage09_postprocess_gate(
    out_dir: Path,
    *,
    map_path: Optional[Path],
    map_exists: bool,
    half_map_1: Optional[Path],
    half_map_2: Optional[Path],
    mask: Optional[Path],
    relion_bin_dir: Optional[Path],
    mode: str = "plan-only",
    resolution_threshold: float = 4.0,
) -> Dict[str, Any]:
    """Plan / run relion_postprocess and emit a map_quality_gate.json record.

    Returns a dict combining the postprocess plan and the map quality gate
    decision.
    """
    mode = _normalise_mode(mode)
    half_maps_available = bool(
        half_map_1 and Path(half_map_1).exists()
        and half_map_2 and Path(half_map_2).exists()
    )
    mask_available = bool(mask and Path(mask).exists())

    plan: Dict[str, Any] = {
        "stage": "09_postprocess",
        "mode": mode,
        "map_path": str(map_path) if map_path else None,
        "map_exists": bool(map_exists),
        "half_map_1": str(half_map_1) if half_map_1 else None,
        "half_map_2": str(half_map_2) if half_map_2 else None,
        "half_maps_available": half_maps_available,
        "mask": str(mask) if mask else None,
        "mask_available": mask_available,
        "resolution_threshold": resolution_threshold,
    }

    postprocess_map: Optional[Path] = None
    fsc_file: Optional[Path] = None
    estimated_resolution: Optional[float] = None

    if mode == "none":
        plan["status"] = "skipped"
        plan["decision"] = "SKIP_POSTPROCESS"
    elif not half_maps_available:
        plan["status"] = "stop_gated"
        plan["decision"] = "STOP_POSTPROCESS_HALF_MAPS_REQUIRED"
    elif not mask_available:
        plan["status"] = "stop_gated"
        plan["decision"] = "STOP_POSTPROCESS_MASK_REQUIRED"
    else:
        bin_path = (
            str(Path(relion_bin_dir) / "relion_postprocess")
            if relion_bin_dir else "relion_postprocess"
        )
        cmd = [
            bin_path,
            "--i", str(half_map_1),
            "--i2", str(half_map_2),
            "--mask", str(mask),
            "--o", "PostProcess/job001/postprocess",
            "--auto_bfac",
            "--autob_lowres", "10",
        ]
        plan["cmd"] = cmd
        if mode in ("plan-only", "dry-run"):
            plan["status"] = "planned"
            plan["decision"] = "PLAN_POSTPROCESS"
        else:  # real
            plan["status"] = "planned"
            plan["decision"] = "REAL_POSTPROCESS_NOT_IMPLEMENTED_OR_DISABLED"
            plan["note"] = (
                "Real relion_postprocess execution is not implemented here. "
                "Provide pre-computed postprocess outputs to populate FSC."
            )

    # Look for any pre-existing postprocess artefacts the user may have placed.
    if map_path is not None:
        candidate_pp = Path(map_path).parent / "postprocess.mrc"
        if candidate_pp.is_file():
            postprocess_map = candidate_pp
        candidate_fsc = Path(map_path).parent / "postprocess_fsc.xml"
        if candidate_fsc.is_file():
            fsc_file = candidate_fsc

    fsc_available = fsc_file is not None
    map_quality_status = "unvalidated"
    allow_map2pdb_real = False
    if fsc_available and estimated_resolution is not None and \
            estimated_resolution <= resolution_threshold:
        map_quality_status = "validated"
        allow_map2pdb_real = True
    elif not map_exists:
        map_quality_status = "blocked"

    if map_quality_status == "validated":
        decision = "ALLOW_MAP2PDB_REAL"
    elif not map_exists:
        decision = "STOP_MAP_QC_REQUIRED"
    else:
        decision = "ALLOW_MAP2PDB_DRY_RUN"

    map_quality_gate = {
        "map_exists": bool(map_exists),
        "postprocess_map_exists": postprocess_map is not None,
        "half_maps_available": half_maps_available,
        "mask_available": mask_available,
        "fsc_available": fsc_available,
        "estimated_resolution": estimated_resolution,
        "resolution_threshold": resolution_threshold,
        "map_quality_status": map_quality_status,
        "allow_map2pdb_real": allow_map2pdb_real,
        "decision": decision,
        "postprocess_map_path": str(postprocess_map) if postprocess_map else None,
        "fsc_file": str(fsc_file) if fsc_file else None,
        "input_map_path": str(map_path) if map_path else None,
    }

    _write_json(out_dir / "postprocess_plan.json", plan)
    _write_text(out_dir / "postprocess_plan.md",
                "# Stage 09 – Postprocess / FSC\n\n" + _md_table({
                    "status": plan.get("status"),
                    "decision": plan.get("decision"),
                    "half_maps_available": half_maps_available,
                    "mask_available": mask_available,
                    "map_exists": map_exists,
                }))
    _write_json(out_dir / "map_quality_gate.json", map_quality_gate)
    _write_text(out_dir / "map_quality_gate.md",
                "# Map quality gate\n\n" + _md_table(map_quality_gate))

    return {"plan": plan, "map_quality_gate": map_quality_gate}
