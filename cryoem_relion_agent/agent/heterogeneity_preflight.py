"""Input contracts and preflight for heterogeneity backends (cryoDRGN)."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


def load_cryodrgn_config(config_path: Optional[Path]) -> Dict[str, Any]:
    if config_path is None or not config_path.exists():
        return {}
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Could not load cryodrgn.yaml: %s", exc)
        return {}


def guess_particles_star(project_root: Path) -> Optional[Path]:
    """Best-effort RELION particles STAR (Refine3D / Class3D / Extract)."""
    if not project_root.is_dir():
        return None
    for sub in ("Refine3D", "Class3D", "Extract"):
        base = project_root / sub
        if not base.is_dir():
            continue
        for star in sorted(base.rglob("*particles*.star")):
            if star.is_file() and "coords" not in star.name.lower():
                return star
    return None


def resolve_cryodrgn_executable(params: Dict[str, Any], config: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    explicit = params.get("cryodrgn_executable") or config.get("cryodrgn_executable")
    if explicit is not None and str(explicit).strip():
        raw = str(explicit).strip()
        p = Path(raw).expanduser()
        if p.is_file():
            if os.access(p, os.X_OK) or os.name == "nt":
                return str(p.resolve()), None
            return None, "missing_dependency: cryodrgn_executable (file not executable)"
        w = shutil.which(raw)
        if w:
            return w, None
        return None, "missing_dependency: cryodrgn_executable"
    w = shutil.which("cryodrgn")
    if w:
        return w, None
    return None, "missing_dependency: cryodrgn_executable"


def preflight_cryodrgn(
    params: Dict[str, Any],
    *,
    project_root: Path,
    cryodrgn_config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Structured preflight for cryoDRGN heterogeneity runs."""
    config = load_cryodrgn_config(cryodrgn_config_path)
    category = "heterogeneity"
    warnings: List[str] = []

    missing: List[str] = []
    out_dir = params.get("output_dir")
    if not out_dir or not str(out_dir).strip():
        missing.append("output_dir")
    else:
        outp = Path(str(out_dir).strip())
        try:
            outp.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            warnings.append(f"output_dir not writable: {exc}")
            missing.append("output_dir")

    star_p = params.get("particles_star_path")
    mrcs = params.get("particles_mrcs_path")
    pose_e = params.get("pose_pkl_path")
    ctf_e = params.get("ctf_pkl_path")

    external_ready = bool(
        mrcs
        and pose_e
        and ctf_e
        and Path(str(mrcs)).is_file()
        and Path(str(pose_e)).is_file()
        and Path(str(ctf_e)).is_file()
    )
    star_ready = bool(star_p and Path(str(star_p).strip()).is_file())

    if not external_ready and not star_ready:
        if not star_p or not str(star_p).strip():
            missing.append("particles_star_path")
        else:
            missing.append("particles_star_path")

    if not params.get("cryodrgn_zdim") and config.get("cryodrgn_default_zdim") is None:
        warnings.append("cryodrgn_zdim not set; default will be taken from cryodrgn.yaml or runner fallback.")

    missing_deps: List[str] = []
    exe_resolved: Optional[str] = None
    exe_err: Optional[str] = None
    if not missing:
        exe_resolved, exe_err = resolve_cryodrgn_executable(params, config)
        if not exe_resolved:
            missing_deps.append("cryodrgn_executable")

    derived_paths: Dict[str, str] = {}
    if star_ready and not external_ready:
        outp = Path(str(out_dir).strip())
        derived_paths["pose_pkl"] = str(outp / "pose.pkl")
        derived_paths["ctf_pkl"] = str(outp / "ctf.pkl")

    if missing or missing_deps:
        blocked_parts: List[str] = []
        if missing:
            blocked_parts.append(f"missing_required_inputs: {','.join(missing)}")
        if missing_deps:
            blocked_parts.append(exe_err or "missing_dependency: cryodrgn_executable")
        blocked = "; ".join(blocked_parts) if blocked_parts else "unavailable"
        return {
            "backend_name": "cryodrgn",
            "category": category,
            "available": False,
            "execution_mode": "unavailable",
            "missing_inputs": missing,
            "missing_dependencies": missing_deps,
            "warnings": warnings,
            "blocked_reason": blocked,
            "executable_path": exe_resolved,
            "input_summary": f"star={star_p}, external_stack={external_ready}, output_dir={out_dir}",
            "derived_paths": derived_paths,
        }

    assert exe_resolved is not None
    mode = "external_stack" if external_ready else "relion_star"
    input_summary = f"mode={mode}, output_dir={out_dir}, executable={exe_resolved}"
    return {
        "backend_name": "cryodrgn",
        "category": category,
        "available": True,
        "execution_mode": "real",
        "missing_inputs": [],
        "missing_dependencies": [],
        "warnings": warnings,
        "blocked_reason": None,
        "executable_path": exe_resolved,
        "input_summary": input_summary,
        "derived_paths": derived_paths,
    }


def enrich_and_preflight_cryodrgn_params(
    params: Dict[str, Any],
    *,
    project_root: Path,
    cryodrgn_config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Merge RELION-oriented defaults, run cryoDRGN preflight, attach heterogeneity_* fields."""
    p = dict(params)
    cfg = load_cryodrgn_config(cryodrgn_config_path)
    if not p.get("output_dir") or not str(p.get("output_dir") or "").strip():
        p["output_dir"] = str(project_root / "CryoDRGN" / "job_agent" / "cryodrgn_run")
    Path(p["output_dir"]).mkdir(parents=True, exist_ok=True)
    if not p.get("particles_star_path") or not str(p.get("particles_star_path") or "").strip():
        g = guess_particles_star(project_root)
        if g:
            p["particles_star_path"] = str(g)
    if p.get("cryodrgn_zdim") is None and cfg.get("cryodrgn_default_zdim") is not None:
        p["cryodrgn_zdim"] = int(cfg["cryodrgn_default_zdim"])
    if p.get("cryodrgn_epochs") is None and cfg.get("cryodrgn_default_epochs") is not None:
        p["cryodrgn_epochs"] = int(cfg["cryodrgn_default_epochs"])
    if p.get("downsample_box_size") is None and cfg.get("cryodrgn_default_downsample_box_size") is not None:
        p["downsample_box_size"] = cfg["cryodrgn_default_downsample_box_size"]
    if p.get("cryodrgn_analyze_after_train") is None and cfg.get("cryodrgn_analyze_after_train") is not None:
        p["cryodrgn_analyze_after_train"] = bool(cfg["cryodrgn_analyze_after_train"])
    pf = preflight_cryodrgn(p, project_root=project_root, cryodrgn_config_path=cryodrgn_config_path)
    p["cryodrgn_preflight"] = pf
    p["heterogeneity_availability"] = "available" if pf.get("available") else "unavailable"
    p["heterogeneity_execution_mode"] = str(pf.get("execution_mode") or "unavailable")
    p["heterogeneity_blocked_reason"] = pf.get("blocked_reason") or ""
    p["heterogeneity_missing_inputs"] = list(pf.get("missing_inputs") or [])
    p["heterogeneity_missing_dependencies"] = list(pf.get("missing_dependencies") or [])
    return p
