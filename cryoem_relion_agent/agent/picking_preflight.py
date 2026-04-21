"""Input contracts and preflight for particle-picking backends (Topaz)."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


def load_topaz_config(config_path: Optional[Path]) -> Dict[str, Any]:
    if config_path is None or not config_path.exists():
        return {}
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Could not load topaz.yaml: %s", exc)
        return {}


def guess_micrographs_star(project_root: Path) -> Optional[Path]:
    """Best-effort RELION micrographs STAR (CtfFind / MotionCorr)."""
    if not project_root.is_dir():
        return None
    ctf = project_root / "CtfFind"
    if ctf.is_dir():
        for name in ("micrographs_ctf.star", "corrected_micrographs.star"):
            p = ctf / name
            if p.is_file():
                return p
        for star in sorted(ctf.glob("*micrographs*.star")):
            if star.is_file() and "coords" not in star.name.lower():
                return star
    motion = project_root / "MotionCorr"
    if motion.is_dir():
        cm = motion / "corrected_micrographs.star"
        if cm.is_file():
            return cm
    return None


def effective_micrographs_star(params: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Return (path, canonical_key_used)."""
    for key in ("micrographs_star_path", "ctf_micrographs_star", "input_micrographs_star"):
        v = params.get(key)
        if v is not None and str(v).strip():
            return str(v).strip(), key
    return None, None


def resolve_topaz_executable(params: Dict[str, Any], config: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve Topaz CLI executable.

    Returns ``(path_or_name, error_message)`` where error_message is set when not found.
    """
    explicit = params.get("topaz_executable") or config.get("topaz_executable")
    if explicit is not None and str(explicit).strip():
        raw = str(explicit).strip()
        p = Path(raw).expanduser()
        if p.is_file():
            if os.access(p, os.X_OK) or os.name == "nt":
                return str(p.resolve()), None
            return None, "missing_dependency: topaz_executable (file not executable)"
        # bare name: try PATH
        w = shutil.which(raw)
        if w:
            return w, None
        return None, "missing_dependency: topaz_executable"

    w = shutil.which("topaz")
    if w:
        return w, None
    return None, "missing_dependency: topaz_executable"


def preflight_topaz_pick(
    params: Dict[str, Any],
    *,
    project_root: Path,
    topaz_config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Structured preflight for Topaz picking.

    ``params`` is merged action parameters (may include paths from scheduler enrichment).
    """
    config = load_topaz_config(topaz_config_path)
    category = "picking"
    warnings: List[str] = []
    model_source = "explicit" if params.get("topaz_model_path") else "default_pretrained"

    mg_path, mg_key = effective_micrographs_star(params)
    missing: List[str] = []
    missing_input_paths: List[str] = []
    if not mg_path:
        missing.append("micrographs_star_path")
    else:
        mp = Path(mg_path)
        if not mp.is_file():
            missing_input_paths.append("micrographs_star_path")

    dia = params.get("particle_diameter_angstrom")
    if dia is None:
        missing.append("particle_diameter_angstrom")
    else:
        try:
            if float(dia) <= 0:
                missing.append("particle_diameter_angstrom")
        except (TypeError, ValueError):
            missing.append("particle_diameter_angstrom")

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

    missing_deps: List[str] = []
    exe_resolved: Optional[str] = None
    exe_err: Optional[str] = None
    if not missing and not missing_input_paths:
        exe_resolved, exe_err = resolve_topaz_executable(params, config)
        if not exe_resolved:
            missing_deps.append("topaz_executable")

    if missing or missing_input_paths or missing_deps:
        blocked_parts: List[str] = []
        if missing:
            blocked_parts.append(f"missing_required_inputs: {','.join(missing)}")
        if missing_input_paths:
            blocked_parts.append(f"input_path_not_found: {','.join(missing_input_paths)}")
        if missing_deps:
            blocked_parts.append(exe_err or "missing_dependency: topaz_executable")
        blocked = "; ".join(blocked_parts) if blocked_parts else "unavailable"
        return {
            "backend_name": "topaz",
            "category": category,
            "available": False,
            "execution_mode": "unavailable",
            "command_mode": "extract",
            "model_source": model_source,
            "missing_inputs": missing,
            "missing_input_paths": missing_input_paths,
            "missing_dependencies": missing_deps,
            "warnings": warnings,
            "blocked_reason": blocked,
            "executable_path": exe_resolved,
            "input_summary": f"micrographs={mg_path or 'missing'}, diameter={dia}, output_dir={out_dir}",
        }

    assert exe_resolved is not None
    input_summary = f"micrographs_star={mg_path} ({mg_key}), diameter_angstrom={dia}, output_dir={out_dir}"
    return {
        "backend_name": "topaz",
        "category": category,
        "available": True,
        "execution_mode": "real",
        "command_mode": "extract",
        "model_source": model_source,
        "missing_inputs": [],
        "missing_input_paths": [],
        "missing_dependencies": [],
        "warnings": warnings,
        "blocked_reason": None,
        "executable_path": exe_resolved,
        "input_summary": input_summary,
    }


def enrich_and_preflight_topaz_params(
    params: Dict[str, Any],
    *,
    project_root: Path,
    topaz_config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Merge RELION-oriented defaults, run Topaz preflight, and attach flattened picking_* fields.

    Used by ``Scheduler`` and ``RelionActionExecutor`` (fallback when not pre-enriched).
    """
    p = dict(params)
    cfg = load_topaz_config(topaz_config_path)
    if not p.get("output_dir") or not str(p.get("output_dir") or "").strip():
        p["output_dir"] = str(project_root / "AutoPick" / "job_agent" / "topaz")
    Path(p["output_dir"]).mkdir(parents=True, exist_ok=True)
    if effective_micrographs_star(p)[0] is None:
        g = guess_micrographs_star(project_root)
        if g:
            p["micrographs_star_path"] = str(g)
    if p.get("particle_diameter_angstrom") is None and cfg.get("default_particle_diameter_angstrom") is not None:
        p["particle_diameter_angstrom"] = float(cfg["default_particle_diameter_angstrom"])
    pf = preflight_topaz_pick(p, project_root=project_root, topaz_config_path=topaz_config_path)
    p["topaz_preflight"] = pf
    p["picking_availability"] = "available" if pf.get("available") else "unavailable"
    p["picking_execution_mode"] = str(pf.get("execution_mode") or "unavailable")
    p["picking_blocked_reason"] = pf.get("blocked_reason") or ""
    p["picking_missing_inputs"] = list(pf.get("missing_inputs") or [])
    p["picking_missing_input_paths"] = list(pf.get("missing_input_paths") or [])
    p["picking_missing_dependencies"] = list(pf.get("missing_dependencies") or [])
    return p
