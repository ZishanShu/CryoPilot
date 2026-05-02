"""Downstream scientific gates for the closed-loop EMPIAR-10017 pipeline.

This module provides plan-only / dry-run / real-run gate helpers for the
post-reconstruction stages: Class2D (Stage 07) and 3D refinement (Stage 08).

The helpers never fake scientific success.  When real-run is requested but
the underlying RELION command builder is not safe to execute in this
environment, the gate records ``status = planned`` and emits a clear
``decision`` value that downstream stages and the workflow summary can
consume.
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


# ---------------------------------------------------------------------------
# Stage 07 – Class2D
# ---------------------------------------------------------------------------


def stage07_class2d_gate(
    out_dir: Path,
    *,
    particles_star: Optional[Path],
    n_particles: int,
    relion_bin_dir: Optional[Path],
    mode: str = "plan-only",
    particle_diameter_angstrom: float = 120.0,
    full_dataset: bool = True,
) -> Dict[str, Any]:
    """Plan / dry-run / real-run Class2D gate.

    Returns a dict with at minimum ``status`` and ``decision`` keys.
    """
    mode = _normalise_mode(mode)

    if mode == "none":
        plan = {
            "stage": "07_class2d",
            "mode": mode,
            "status": "skipped",
            "decision": "SKIP_CLASS2D",
        }
        _write_json(out_dir / "class2d_plan.json", plan)
        _write_text(out_dir / "class2d_plan.md",
                    "# Stage 07 – Class2D\n\n" + _md_table(
                        {"status": "skipped", "decision": "SKIP_CLASS2D", "mode": mode}))
        return plan

    if particles_star is None or not Path(particles_star).is_file() or n_particles <= 0:
        plan = {
            "stage": "07_class2d",
            "mode": mode,
            "status": "blocked",
            "decision": "BLOCK_CLASS2D_NO_PARTICLES",
            "particles_star": str(particles_star) if particles_star else None,
            "n_particles": n_particles,
        }
        _write_json(out_dir / "class2d_plan.json", plan)
        _write_text(out_dir / "class2d_plan.md",
                    "# Stage 07 – Class2D\n\n" + _md_table(
                        {"status": "blocked",
                         "decision": "BLOCK_CLASS2D_NO_PARTICLES",
                         "particles_star": plan["particles_star"],
                         "n_particles": n_particles}))
        return plan

    K = 50 if full_dataset else 10
    output_dir = "Class2D/job001/run"
    cmd = []
    if relion_bin_dir is not None:
        cmd.append(str(Path(relion_bin_dir) / "relion_refine"))
    else:
        cmd.append("relion_refine")
    cmd += [
        "--i", str(particles_star),
        "--o", output_dir,
        "--iter", "25",
        "--K", str(K),
        "--particle_diameter", str(particle_diameter_angstrom),
        "--ctf",
        "--pool", "10",
        "--norm",
        "--scale",
    ]

    plan: Dict[str, Any] = {
        "stage": "07_class2d",
        "mode": mode,
        "particles_star": str(particles_star),
        "n_particles": n_particles,
        "K": K,
        "particle_diameter_angstrom": particle_diameter_angstrom,
        "output_dir": output_dir,
        "cmd": cmd,
    }

    if mode in ("plan-only", "dry-run"):
        plan["status"] = "planned"
        plan["decision"] = "PLAN_CLASS2D"
    else:  # real
        plan["status"] = "planned"
        plan["decision"] = "REAL_CLASS2D_NOT_IMPLEMENTED_OR_DISABLED"
        plan["note"] = (
            "Real RELION Class2D execution is not safely implemented in this "
            "pipeline; command recorded but not executed."
        )

    _write_json(out_dir / "class2d_plan.json", plan)
    _write_text(out_dir / "class2d_plan.md",
                "# Stage 07 – Class2D\n\n" + _md_table({
                    "status": plan["status"],
                    "decision": plan["decision"],
                    "mode": mode,
                    "K": K,
                    "particles_star": plan["particles_star"],
                    "n_particles": n_particles,
                    "output_dir": output_dir,
                }) + "\n## Command\n\n```\n" + " ".join(cmd) + "\n```\n")
    return plan


# ---------------------------------------------------------------------------
# Stage 08 – 3D refinement gate
# ---------------------------------------------------------------------------


def stage08_refine3d_gate(
    out_dir: Path,
    *,
    map_path: Optional[Path],
    map_exists: bool,
    particles_star: Optional[Path],
    n_particles: int,
    relion_bin_dir: Optional[Path],
    mode: str = "plan-only",
    particle_diameter_angstrom: float = 120.0,
) -> Dict[str, Any]:
    """Plan / dry-run / real-run 3D refinement gate."""
    mode = _normalise_mode(mode)

    if mode == "none":
        plan = {
            "stage": "08_refine3d",
            "mode": mode,
            "status": "skipped",
            "decision": "SKIP_REFINE3D",
            "map_artifact_from_reconstruct": bool(map_exists),
            "refined_map_available": False,
        }
        _write_json(out_dir / "refine3d_plan.json", plan)
        _write_text(out_dir / "refine3d_plan.md",
                    "# Stage 08 – 3D refinement\n\n" + _md_table(
                        {"status": "skipped", "decision": "SKIP_REFINE3D"}))
        return plan

    blocking_decision: Optional[str] = None
    if not map_exists or map_path is None:
        blocking_decision = "BLOCK_REFINE3D_NO_MAP"
    elif particles_star is None or not Path(particles_star).is_file() or n_particles <= 0:
        blocking_decision = "BLOCK_REFINE3D_NO_PARTICLES"

    if blocking_decision is not None:
        plan = {
            "stage": "08_refine3d",
            "mode": mode,
            "status": "blocked",
            "decision": blocking_decision,
            "map_path": str(map_path) if map_path else None,
            "map_artifact_from_reconstruct": bool(map_exists),
            "refined_map_available": False,
            "particles_star": str(particles_star) if particles_star else None,
            "n_particles": n_particles,
        }
        _write_json(out_dir / "refine3d_plan.json", plan)
        _write_text(out_dir / "refine3d_plan.md",
                    "# Stage 08 – 3D refinement\n\n" + _md_table(
                        {"status": "blocked", "decision": blocking_decision}))
        return plan

    output_dir = "Refine3D/job001/run"
    cmd = []
    if relion_bin_dir is not None:
        cmd.append(str(Path(relion_bin_dir) / "relion_refine"))
    else:
        cmd.append("relion_refine")
    cmd += [
        "--i", str(particles_star),
        "--ref", str(map_path),
        "--o", output_dir,
        "--auto_refine",
        "--split_random_halves",
        "--particle_diameter", str(particle_diameter_angstrom),
        "--ctf",
        "--ini_high", "30",
        "--healpix_order", "2",
        "--auto_local_healpix_order", "4",
        "--offset_range", "5",
        "--offset_step", "2",
        "--norm",
        "--scale",
    ]

    plan: Dict[str, Any] = {
        "stage": "08_refine3d",
        "mode": mode,
        "map_path": str(map_path),
        "map_artifact_from_reconstruct": True,
        "refined_map_available": False,
        "particles_star": str(particles_star),
        "n_particles": n_particles,
        "output_dir": output_dir,
        "cmd": cmd,
    }

    if mode in ("plan-only", "dry-run"):
        plan["status"] = "planned"
        plan["decision"] = "PLAN_REFINE3D"
    else:
        plan["status"] = "planned"
        plan["decision"] = "REAL_REFINE3D_NOT_IMPLEMENTED_OR_DISABLED"
        plan["note"] = (
            "Real RELION 3D refinement execution is not safely implemented "
            "in this pipeline; command recorded but not executed."
        )

    _write_json(out_dir / "refine3d_plan.json", plan)
    _write_text(out_dir / "refine3d_plan.md",
                "# Stage 08 – 3D refinement\n\n" + _md_table({
                    "status": plan["status"],
                    "decision": plan["decision"],
                    "mode": mode,
                    "map_path": plan["map_path"],
                    "particles_star": plan["particles_star"],
                    "n_particles": n_particles,
                    "map_artifact_from_reconstruct": True,
                    "refined_map_available": False,
                }) + "\n## Command\n\n```\n" + " ".join(cmd) + "\n```\n")
    return plan
