"""Preflight checks for model-building backends (uses registry input specs)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import yaml

from agent.backend_registry import (
    MODEL_BUILDING_BACKEND_METADATA,
    MODEL_BUILDING_INPUT_SPECS,
    model_building_backend_to_action_name,
)


def _field_present(params: Dict[str, Any], key: str) -> bool:
    v = params.get(key)
    return bool(v is not None and str(v).strip())


def _effective_af3_dir(params: Dict[str, Any]) -> Optional[str]:
    for k in ("af3_dir", "af3_result_dir"):
        v = params.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


def _load_emodelx_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    p = config_path or (Path(__file__).resolve().parents[1] / "configs" / "emodelx.yaml")
    if not p.is_file():
        return {}
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError):
        return {}


def _load_mica_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    p = config_path or (Path(__file__).resolve().parents[1] / "configs" / "mica.yaml")
    if not p.is_file():
        return {}
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError):
        return {}


def _resolve_python(raw: Any) -> Optional[str]:
    if raw is not None and str(raw).strip():
        s = str(raw).strip()
        p = Path(s).expanduser()
        if p.is_file():
            return str(p.resolve())
        which = shutil.which(s)
        if which:
            return which
        return None
    cur = sys.executable
    if cur and Path(cur).is_file():
        return cur
    return shutil.which("python3") or shutil.which("python")


def preflight_model_building(
    backend_id: str,
    params: Dict[str, Any],
    *,
    emodelx_config_path: Optional[Path] = None,
    mica_config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Structured preflight before model-building execution.

    Returns a JSON-friendly dict (lists, scalars, no Path objects).
    """
    bid = str(backend_id or "").strip().lower()
    spec = MODEL_BUILDING_INPUT_SPECS.get(bid)
    meta = MODEL_BUILDING_BACKEND_METADATA.get(bid, {})
    category = str(meta.get("category", "model_building"))

    if spec is None:
        return {
            "backend_name": bid or "unknown",
            "category": category,
            "available": False,
            "execution_mode": "unavailable",
            "placeholder": False,
            "missing_inputs": [],
            "missing_dependencies": [],
            "warnings": [f"unknown model_building backend: {bid!r}"],
            "blocked_reason": "unknown_backend",
            "recommended_action_name": None,
            "input_summary": "",
        }

    effective = dict(params)
    for k, v in spec.get("defaults", {}).items():
        effective.setdefault(k, v)

    required: tuple[str, ...] = tuple(spec["required_fields"])
    missing: List[str] = []
    for field in required:
        if field == "af3_dir":
            if _effective_af3_dir(effective) is None:
                missing.append("af3_dir")
        elif not _field_present(effective, field):
            missing.append(field)

    missing_deps: List[str] = []
    warnings: List[str] = []

    if missing:
        blocked = "missing_required_inputs"
        if len(missing) == 1:
            blocked = f"missing_required_inputs: {missing[0]}"
        return {
            "backend_name": bid,
            "category": category,
            "available": False,
            "execution_mode": "unavailable",
            "placeholder": bool(meta.get("placeholder_execution", False)),
            "missing_inputs": missing,
            "missing_dependencies": missing_deps,
            "warnings": warnings,
            "blocked_reason": blocked,
            "recommended_action_name": model_building_backend_to_action_name(bid),
            "input_summary": f"missing={','.join(missing)}",
        }

    if bid == "modelangelo":
        execution_mode = "real"
        placeholder = False
        if meta.get("placeholder_execution"):
            execution_mode = "dry_run_placeholder"
            placeholder = True
    elif bid == "emodelx":
        execution_mode = "real"
        placeholder = False
    elif bid == "mica":
        execution_mode = "real"
        placeholder = False
    else:
        execution_mode = "dry_run_placeholder"
        placeholder = True

    if bid == "modelangelo":
        opt: Set[str] = set(spec.get("optional_fields", ()))
        absent_optional = [f for f in sorted(opt) if not _field_present(effective, f)]
        if absent_optional:
            warnings.append(f"optional inputs not set: {', '.join(absent_optional[:8])}")

    input_summary = "map_path=set, sequence_path=set"
    if bid == "mica":
        input_summary += f", af3_dir={_effective_af3_dir(effective)}"

    if bid == "emodelx":
        cfg = _load_emodelx_config(emodelx_config_path)
        protocol = str(effective.get("emodelx_protocol") or cfg.get("emodelx_default_protocol") or "temp_free").strip()
        repo_raw = effective.get("emodelx_repo_dir") or cfg.get("emodelx_repo_dir")
        py_raw = effective.get("emodelx_python") or cfg.get("emodelx_python")
        weights_raw = effective.get("emodelx_weights_dir")
        run_pulchra = bool(
            effective.get("run_pulchra")
            if effective.get("run_pulchra") is not None
            else cfg.get("emodelx_default_run_pulchra", False)
        )
        run_phenix = bool(
            effective.get("run_phenix")
            if effective.get("run_phenix") is not None
            else cfg.get("emodelx_default_run_phenix", False)
        )

        repo_path = Path(str(repo_raw).strip()).expanduser() if repo_raw is not None and str(repo_raw).strip() else None
        run_py_path = repo_path / "run.py" if repo_path else None
        yml_path = repo_path / "EModelX.yml" if repo_path else None
        weights_dir = (
            Path(str(weights_raw).strip()).expanduser()
            if weights_raw is not None and str(weights_raw).strip()
            else (repo_path / "models" if repo_path else None)
        )
        py_path = _resolve_python(py_raw)

        if not _field_present(effective, "map_path") or not Path(str(effective.get("map_path"))).is_file():
            missing.append("map_path")
        if not _field_present(effective, "sequence_path") or not Path(str(effective.get("sequence_path"))).is_file():
            missing.append("sequence_path")

        out_dir = Path(str(effective.get("output_dir")).strip())
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            missing.append("output_dir")

        if repo_path is None or not repo_path.is_dir():
            missing.append("emodelx_repo_dir")
        if run_py_path is None or not run_py_path.is_file():
            missing_deps.append("run.py")
        if yml_path is None or not yml_path.is_file():
            warnings.append("EModelX.yml not found in emodelx_repo_dir")
        if weights_dir is None or not weights_dir.is_dir() or not any(weights_dir.iterdir()):
            missing_deps.append("emodelx_weights")
        if py_path is None:
            missing_deps.append("emodelx_python")

        if run_pulchra:
            pulchra_path = effective.get("pulchra_path")
            if pulchra_path is None or not str(pulchra_path).strip() or not Path(str(pulchra_path)).is_file():
                missing_deps.append("pulchra_path")
        if run_phenix:
            phenix_act = effective.get("phenix_act")
            if phenix_act is None or not str(phenix_act).strip() or not Path(str(phenix_act)).is_file():
                missing_deps.append("phenix_act")
            if effective.get("resolution") is None and cfg.get("emodelx_default_resolution") is None:
                warnings.append("run_phenix enabled without explicit resolution; EModelX defaults will be used")
        if protocol == "temp_flex":
            has_dl = bool(effective.get("download_afdb"))
            has_tpl = bool(effective.get("template_dir")) and Path(str(effective.get("template_dir"))).is_dir()
            if not has_dl and not has_tpl:
                missing.append("template_dir_or_download_afdb")

        if missing or missing_deps:
            blocked_chunks: List[str] = []
            if missing:
                blocked_chunks.append(f"missing_required_inputs: {','.join(dict.fromkeys(missing))}")
            if missing_deps:
                dep = dict.fromkeys(missing_deps)
                if "emodelx_weights" in dep:
                    blocked_chunks.append("missing_dependency: emodelx_weights")
                elif "emodelx_python" in dep:
                    blocked_chunks.append("missing_dependency: emodelx_python")
                else:
                    blocked_chunks.append(f"missing_dependency: {','.join(dep.keys())}")
            return {
                "backend_name": bid,
                "category": category,
                "available": False,
                "execution_mode": "unavailable",
                "placeholder": False,
                "missing_inputs": list(dict.fromkeys(missing)),
                "missing_dependencies": list(dict.fromkeys(missing_deps)),
                "warnings": warnings,
                "blocked_reason": "; ".join(blocked_chunks),
                "recommended_action_name": model_building_backend_to_action_name(bid),
                "repo_path": str(repo_path) if repo_path else None,
                "python_path": py_path,
                "run_py_path": str(run_py_path) if run_py_path else None,
                "weights_dir": str(weights_dir) if weights_dir else None,
                "protocol": protocol,
                "resolution": effective.get("resolution") if effective.get("resolution") is not None else cfg.get("emodelx_default_resolution"),
                "command": (
                    f"{py_path} {run_py_path} --protocol {protocol} --EM_map {effective.get('map_path')} "
                    f"--fasta {effective.get('sequence_path')} --output_dir {out_dir}"
                )
                if py_path and run_py_path
                else None,
                "output_path": str(out_dir),
                "input_summary": (
                    f"protocol={protocol}, repo={repo_path}, "
                    f"map={effective.get('map_path')}, sequence={effective.get('sequence_path')}, output_dir={out_dir}"
                ),
            }

        input_summary = (
            f"protocol={protocol}, repo={repo_path}, map={effective.get('map_path')}, "
            f"sequence={effective.get('sequence_path')}, output_dir={out_dir}"
        )
        return {
            "backend_name": bid,
            "category": category,
            "available": True,
            "execution_mode": "real",
            "placeholder": False,
            "missing_inputs": [],
            "missing_dependencies": [],
            "warnings": warnings,
            "blocked_reason": None,
            "recommended_action_name": model_building_backend_to_action_name(bid),
            "repo_path": str(repo_path),
            "python_path": py_path,
            "run_py_path": str(run_py_path),
            "weights_dir": str(weights_dir) if weights_dir else None,
            "protocol": protocol,
            "resolution": effective.get("resolution") if effective.get("resolution") is not None else cfg.get("emodelx_default_resolution"),
            "command": (
                f"{py_path} {run_py_path} --protocol {protocol} --EM_map {effective.get('map_path')} "
                f"--fasta {effective.get('sequence_path')} --output_dir {out_dir}"
            ),
            "output_path": str(out_dir),
            "input_summary": input_summary,
        }

    if bid == "mica":
        cfg = _load_mica_config(mica_config_path)
        repo_raw = effective.get("mica_repo_dir") or cfg.get("mica_repo_dir")
        py_raw = effective.get("mica_python") or cfg.get("mica_python")
        weights_raw = effective.get("mica_weights_dir")
        protocol = str(effective.get("protocol") or "AF3_struct").strip()
        run_pulchra = bool(
            effective.get("run_pulchra")
            if effective.get("run_pulchra") is not None
            else cfg.get("mica_default_run_pulchra", False)
        )
        run_phenix = bool(
            effective.get("run_phenix")
            if effective.get("run_phenix") is not None
            else cfg.get("mica_default_run_phenix", False)
        )
        device = str(effective.get("mica_device") or cfg.get("mica_default_device") or "cuda")
        resolution = (
            effective.get("mica_resolution")
            if effective.get("mica_resolution") is not None
            else cfg.get("mica_default_resolution")
        )
        af3_dir_s = _effective_af3_dir(effective)
        af3_dir = Path(af3_dir_s).expanduser() if af3_dir_s else None

        repo_path = Path(str(repo_raw).strip()).expanduser() if repo_raw is not None and str(repo_raw).strip() else None
        run_py_path = repo_path / "run.py" if repo_path else None
        yml_path = repo_path / "environment.yml" if repo_path else None
        weights_dir = (
            Path(str(weights_raw).strip()).expanduser()
            if weights_raw is not None and str(weights_raw).strip()
            else (repo_path / "trained_models" if repo_path else None)
        )
        py_path = _resolve_python(py_raw)

        if not _field_present(effective, "map_path") or not Path(str(effective.get("map_path"))).is_file():
            missing.append("map_path")
        if not _field_present(effective, "sequence_path") or not Path(str(effective.get("sequence_path"))).is_file():
            missing.append("sequence_path")

        out_dir = Path(str(effective.get("output_dir")).strip())
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            missing.append("output_dir")

        if repo_path is None or not repo_path.is_dir():
            missing.append("mica_repo_dir")
        if af3_dir is None or not af3_dir.is_dir() or not any(af3_dir.iterdir()):
            missing.append("af3_dir")

        if run_py_path is None or not run_py_path.is_file():
            missing_deps.append("run.py")
        if yml_path is None or not yml_path.is_file():
            warnings.append("environment.yml not found in mica_repo_dir")
        if weights_dir is None or not weights_dir.is_dir() or not any(weights_dir.iterdir()):
            missing_deps.append("mica_weights")
        if py_path is None:
            missing_deps.append("mica_python")

        if run_pulchra:
            pulchra_path = effective.get("pulchra_path")
            if pulchra_path is None or not str(pulchra_path).strip() or not Path(str(pulchra_path)).is_file():
                missing_deps.append("pulchra_path")
        if run_phenix:
            phenix_act = effective.get("phenix_act")
            if phenix_act is None or not str(phenix_act).strip() or not Path(str(phenix_act)).is_file():
                missing_deps.append("phenix_act")
            if resolution is None:
                warnings.append("run_phenix enabled without explicit mica_resolution; upstream defaults will be used")

        cmd = None
        if py_path and run_py_path and af3_dir:
            cmd = (
                f"{py_path} {run_py_path} --protocol {protocol} -m {effective.get('map_path')} "
                f"-f {effective.get('sequence_path')} -a {af3_dir} -o {out_dir} --device {device}"
            )

        if missing or missing_deps:
            blocked_chunks: List[str] = []
            if missing:
                uniq_missing = list(dict.fromkeys(missing))
                if uniq_missing == ["af3_dir"]:
                    blocked_chunks.append("missing_required_inputs: af3_dir")
                else:
                    blocked_chunks.append(f"missing_required_inputs: {','.join(uniq_missing)}")
            if missing_deps:
                dep = dict.fromkeys(missing_deps)
                if "mica_weights" in dep:
                    blocked_chunks.append("missing_dependency: mica_weights")
                elif "mica_python" in dep:
                    blocked_chunks.append("missing_dependency: mica_python")
                else:
                    blocked_chunks.append(f"missing_dependency: {','.join(dep.keys())}")
            return {
                "backend_name": bid,
                "category": category,
                "available": False,
                "execution_mode": "unavailable",
                "placeholder": False,
                "missing_inputs": list(dict.fromkeys(missing)),
                "missing_dependencies": list(dict.fromkeys(missing_deps)),
                "warnings": warnings,
                "blocked_reason": "; ".join(blocked_chunks),
                "recommended_action_name": model_building_backend_to_action_name(bid),
                "repo_path": str(repo_path) if repo_path else None,
                "python_path": py_path,
                "run_py_path": str(run_py_path) if run_py_path else None,
                "weights_dir": str(weights_dir) if weights_dir else None,
                "af3_dir": str(af3_dir) if af3_dir else None,
                "resolution": resolution,
                "command": cmd,
                "output_path": str(out_dir),
                "input_summary": (
                    f"protocol={protocol}, repo={repo_path}, map={effective.get('map_path')}, "
                    f"sequence={effective.get('sequence_path')}, af3_dir={af3_dir}, output_dir={out_dir}"
                ),
            }

        return {
            "backend_name": bid,
            "category": category,
            "available": True,
            "execution_mode": "real",
            "placeholder": False,
            "missing_inputs": [],
            "missing_dependencies": [],
            "warnings": warnings,
            "blocked_reason": None,
            "recommended_action_name": model_building_backend_to_action_name(bid),
            "repo_path": str(repo_path),
            "python_path": py_path,
            "run_py_path": str(run_py_path),
            "weights_dir": str(weights_dir) if weights_dir else None,
            "af3_dir": str(af3_dir) if af3_dir else None,
            "resolution": resolution,
            "command": cmd,
            "output_path": str(out_dir),
            "input_summary": (
                f"protocol={protocol}, repo={repo_path}, map={effective.get('map_path')}, "
                f"sequence={effective.get('sequence_path')}, af3_dir={af3_dir}, output_dir={out_dir}"
            ),
        }

    return {
        "backend_name": bid,
        "category": category,
        "available": True,
        "execution_mode": execution_mode,
        "placeholder": placeholder,
        "missing_inputs": [],
        "missing_dependencies": missing_deps,
        "warnings": warnings,
        "blocked_reason": None,
        "recommended_action_name": model_building_backend_to_action_name(bid),
        "input_summary": input_summary,
    }
