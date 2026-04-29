"""Runtime portability diagnostics for backend executables and paths.

This module is intentionally separate from the existing runner preflights.  It
answers "can this migrated server run the configured backends?" without changing
runner return contracts.
"""

from __future__ import annotations

import getpass
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

from agent.backend_registry import get_backend_capability_matrix

_ABS_PATH_RE = re.compile(r"(?<![A-Za-z0-9_.-])/(?:[A-Za-z0-9._@%+=:,~-]+/?)+")
_TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".conf",
    ".env",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".star",
    ".txt",
    ".yaml",
    ".yml",
}
_VAR_DEFAULT_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]+)\}")


def _expand_config_value(value: Any) -> Any:
    """Expand env vars, ``~``, and a small bash-like ``${VAR:-default}`` subset."""
    if not isinstance(value, str):
        return value
    text = value
    if "CRYOAGENT_ROOT" not in os.environ:
        repo_root = str(Path(__file__).resolve().parents[2])
        text = text.replace("${CRYOAGENT_ROOT}", repo_root).replace("$CRYOAGENT_ROOT", repo_root)
    for _ in range(5):
        new = _VAR_DEFAULT_RE.sub(lambda m: os.environ.get(m.group(1), m.group(2)), text)
        if new == text:
            break
        text = new
    return os.path.expanduser(os.path.expandvars(text))


def _expand_config_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand_config_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_config_tree(v) for v in value]
    return _expand_config_value(value)


def _status_from_access(access: Optional[Mapping[str, Any]], *, require_executable: bool = False) -> str:
    if not access:
        return "missing"
    if access.get("path_stale"):
        return "stale_path"
    if access.get("not_accessible"):
        return "not_accessible"
    if not access.get("exists"):
        return "missing"
    if require_executable and not access.get("executable"):
        return "not_executable"
    return "ok"


def _suggested_fix(path_text: Optional[str]) -> str:
    if not path_text:
        return "Set the corresponding backend path in config or environment."
    if _is_under_root(path_text):
        return "Replace old /root path with the migrated /home/user/CryoAgent or conda_envs path, or set the backend environment variable."
    return "Create the path, install the backend, or update the backend config/environment variable."


def environment_fingerprint() -> Dict[str, Any]:
    """Small environment snapshot for audit/debug reports."""
    nvidia = shutil.which("nvidia-smi")
    torch_cuda: Any = "unavailable"
    try:
        import torch  # type: ignore

        torch_cuda = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001
        torch_cuda = f"unknown: {exc.__class__.__name__}"
    return {
        "current_user": getpass.getuser(),
        "hostname": os.uname().nodename if hasattr(os, "uname") else "",
        "sys_executable": sys.executable,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "path": os.environ.get("PATH", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_available": bool(nvidia),
        "nvidia_smi_path": nvidia,
        "torch_cuda_available": torch_cuda,
    }


def _first_access(resolve_result: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    attempts = resolve_result.get("attempts")
    if not isinstance(attempts, list):
        return None
    for attempt in attempts:
        if isinstance(attempt, Mapping) and isinstance(attempt.get("access"), Mapping):
            return attempt["access"]
    return None


def _stringify_path(path: Any) -> Optional[str]:
    if path is None:
        return None
    raw = str(path).strip()
    return raw or None


def _is_under_root(path_text: str) -> bool:
    return path_text == "/root" or path_text.startswith("/root/")


def _current_user_is_root() -> bool:
    geteuid = getattr(os, "geteuid", None)
    try:
        return bool(geteuid is not None and geteuid() == 0)
    except OSError:
        return False


def _stale_reasons(path_text: str, exists: Optional[bool]) -> List[str]:
    reasons: List[str] = []
    if _is_under_root(path_text):
        reasons.append("under_root")
    if path_text.startswith("/root/anaconda3"):
        reasons.append("root_anaconda3")
    if path_text.startswith("/root/share"):
        reasons.append("root_share")
    if path_text.startswith("/") and exists is False:
        reasons.append("absolute_path_missing")
    return reasons


def check_path_access(path: Any) -> Dict[str, Any]:
    """Return structured access status for a path, never raising path errors."""
    path_text = _stringify_path(path)
    result: Dict[str, Any] = {
        "path": path_text,
        "exists": False,
        "is_absolute": False,
        "is_file": False,
        "is_dir": False,
        "readable": False,
        "writable": False,
        "executable": False,
        "not_accessible": False,
        "path_stale": False,
        "stale_reasons": [],
        "error": None,
    }
    if path_text is None:
        result["error"] = "missing_path"
        return result

    expanded = Path(path_text).expanduser()
    result["is_absolute"] = expanded.is_absolute()
    try:
        exists = expanded.exists()
        result["exists"] = bool(exists)
        result["is_file"] = bool(expanded.is_file()) if exists else False
        result["is_dir"] = bool(expanded.is_dir()) if exists else False
        result["readable"] = bool(os.access(expanded, os.R_OK)) if exists else False
        result["writable"] = bool(os.access(expanded, os.W_OK)) if exists else False
        result["executable"] = bool(os.access(expanded, os.X_OK)) if exists else False
    except PermissionError as exc:
        result["not_accessible"] = True
        result["error"] = f"PermissionError: {exc}"
        result["exists"] = None
    except OSError as exc:
        result["not_accessible"] = True
        result["error"] = f"{exc.__class__.__name__}: {exc}"
        result["exists"] = None

    if _is_under_root(path_text) and not _current_user_is_root():
        result["not_accessible"] = True
    stale = _stale_reasons(path_text, result["exists"])
    result["stale_reasons"] = stale
    result["path_stale"] = bool(stale)
    return result


def resolve_executable(
    executable: Optional[str] = None,
    *,
    default_names: Sequence[str] = (),
    env_var: Optional[str] = None,
    env_vars: Sequence[str] = (),
    candidate_dirs: Sequence[Any] = (),
) -> Dict[str, Any]:
    """Resolve an executable from an explicit path/name, env var, or PATH."""
    attempts: List[Dict[str, Any]] = []
    candidates: List[Tuple[str, str]] = []
    if executable and str(executable).strip():
        candidates.append(("explicit", str(_expand_config_value(str(executable).strip()))))
    if env_var and os.environ.get(env_var):
        candidates.append((f"env:{env_var}", str(_expand_config_value(os.environ[env_var].strip()))))
    for ev in env_vars:
        if ev and os.environ.get(ev):
            candidates.append((f"env:{ev}", str(_expand_config_value(os.environ[ev].strip()))))
    for name in default_names:
        if name and str(name).strip():
            candidates.append(("path", str(name).strip()))
    for d in candidate_dirs:
        d_text = _stringify_path(_expand_config_value(d))
        if not d_text:
            continue
        for name in default_names:
            if name and str(name).strip():
                candidates.append(("candidate_dir", str(Path(d_text) / str(name).strip())))

    seen_candidates: set[Tuple[str, str]] = set()
    for source, raw in candidates:
        key = (source, raw)
        if key in seen_candidates:
            continue
        seen_candidates.add(key)
        p = Path(raw).expanduser()
        if p.parent != Path(".") or raw.startswith(("~", "/")):
            access = check_path_access(str(p))
            attempts.append({"source": source, "candidate": raw, "access": access})
            if access.get("exists") and access.get("is_file") and access.get("executable"):
                try:
                    resolved = str(p.resolve())
                except (OSError, PermissionError):
                    resolved = str(p)
                return {
                    "resolved": resolved,
                    "available": True,
                    "source": source,
                    "attempts": attempts,
                    "error": None,
                }
            continue
        which = shutil.which(raw)
        access = check_path_access(which) if which else check_path_access(raw)
        attempts.append({"source": source, "candidate": raw, "which": which, "access": access})
        if which:
            return {"resolved": which, "available": True, "source": source, "attempts": attempts, "error": None}

    return {
        "resolved": None,
        "available": False,
        "source": None,
        "attempts": attempts,
        "error": "executable_not_found" if attempts else "no_executable_candidates",
    }


def check_output_dir_writable(path: Any) -> Dict[str, Any]:
    """Check/create an output directory and verify a small test file can be written."""
    path_text = _stringify_path(path)
    result = check_path_access(path_text)
    result.update({"writable_output_dir": False, "created": False})
    if path_text is None:
        result["error"] = "missing_output_dir"
        return result
    out = Path(path_text).expanduser()
    try:
        existed = out.exists()
        out.mkdir(parents=True, exist_ok=True)
        result["created"] = not existed
        probe = out / ".cryoagent_write_test"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink(missing_ok=True)
        result.update(check_path_access(str(out)))
        result["writable_output_dir"] = True
    except PermissionError as exc:
        result["not_accessible"] = True
        result["error"] = f"PermissionError: {exc}"
    except OSError as exc:
        result["not_accessible"] = True
        result["error"] = f"{exc.__class__.__name__}: {exc}"
    return result


def _safe_load_yaml(path: Path) -> Dict[str, Any]:
    try:
        if not path.is_file():
            return {}
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return _expand_config_tree(data) if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError):
        return {}


def _config_root(config_root: Optional[Path] = None) -> Path:
    return config_root or (Path(__file__).resolve().parents[1] / "configs")


def _env_first(*names: str) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return _expand_config_value(value.strip())
    return None


def _modelangelo_config(config_root: Optional[Path] = None) -> Dict[str, Any]:
    data = _safe_load_yaml(_config_root(config_root) / "model_builders.yaml")
    section = data.get("model_angelo") or data.get("modelangelo") or {}
    return section if isinstance(section, dict) else {}


_BACKEND_ENV: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "modelangelo": {
        "executable": ("MODELANGELO_EXECUTABLE",),
        "repo_dir": ("MODELANGELO_REPO_DIR",),
        "weights_dir": ("MODELANGELO_WEIGHTS_DIR", "MODELANGELO_TORCH_HOME"),
    },
    "emodelx": {
        "executable": ("EMODELX_EXECUTABLE", "EMODELX_PYTHON"),
        "repo_dir": ("EMODELX_REPO_DIR",),
        "weights_dir": ("EMODELX_WEIGHTS_DIR",),
    },
    "mica": {
        "executable": ("MICA_EXECUTABLE", "MICA_PYTHON"),
        "repo_dir": ("MICA_REPO_DIR",),
        "weights_dir": ("MICA_WEIGHTS_DIR",),
    },
    "topaz": {"executable": ("TOPAZ_EXECUTABLE",)},
    "cryodrgn": {"executable": ("CRYODRGN_EXECUTABLE", "CRYODRGN_PYTHON")},
}


def _first_env(names: Sequence[str]) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return _expand_config_value(value)
    return None


def _safe_resolve(path: Path) -> str:
    try:
        return str(path.resolve())
    except (OSError, PermissionError):
        return str(path)


def _candidate_paths(project_root: Optional[Path], *parts: str) -> Tuple[str, ...]:
    roots: List[Path] = []
    for raw in (
        os.environ.get("CRYOAGENT_BACKEND_ROOT"),
        os.environ.get("CRYOAGENT_ROOT"),
        str(project_root.parent) if project_root else None,
        str(Path.home() / "CryoAgent"),
    ):
        if raw and str(raw).strip():
            roots.append(Path(_expand_config_value(str(raw))))
    out: List[str] = []
    for root in roots:
        for part in parts:
            out.append(str(root / part))
    return tuple(dict.fromkeys(out))


def _backend_definition(backend_name: str, *, config_root: Optional[Path] = None) -> Dict[str, Any]:
    cfg_root = _config_root(config_root)
    name = str(backend_name).strip().lower()
    if name == "modelangelo":
        cfg = _modelangelo_config(cfg_root)
        torch_home = _env_first("MODELANGELO_TORCH_HOME") or cfg.get("torch_home")
        weights_rel = cfg.get("weights_relpath")
        weights = _env_first("MODELANGELO_WEIGHTS_DIR") or (
            str(Path(str(torch_home)) / str(weights_rel)) if torch_home and weights_rel else torch_home
        )
        return {
            "category": "model_building",
            "executable": _env_first("MODELANGELO_EXECUTABLE") or cfg.get("executable"),
            "default_executables": ("model_angelo", "model-angelo"),
            "repo_dir": _env_first("MODELANGELO_REPO_DIR") or cfg.get("repo_dir"),
            "weights_dir": weights,
            "required_runtime": ("executable", "weights_dir"),
            "recommended_action": "Install ModelAngelo or set MODELANGELO_EXECUTABLE/MODELANGELO_WEIGHTS_DIR.",
        }
    if name == "emodelx":
        cfg = _safe_load_yaml(cfg_root / "emodelx.yaml")
        repo = _env_first("EMODELX_REPO_DIR") or cfg.get("emodelx_repo_dir")
        return {
            "category": "model_building",
            "executable": _env_first("EMODELX_EXECUTABLE", "EMODELX_PYTHON") or cfg.get("emodelx_python") or sys.executable,
            "default_executables": ("python3", "python"),
            "repo_dir": repo,
            "weights_dir": _env_first("EMODELX_WEIGHTS_DIR") or (str(Path(str(repo)) / "models") if repo else None),
            "required_runtime": ("executable", "repo_dir", "weights_dir"),
            "recommended_action": "Install/configure EModelX or set EMODELX_REPO_DIR/EMODELX_WEIGHTS_DIR.",
        }
    if name == "mica":
        cfg = _safe_load_yaml(cfg_root / "mica.yaml")
        repo = _env_first("MICA_REPO_DIR") or cfg.get("mica_repo_dir")
        return {
            "category": "model_building",
            "executable": _env_first("MICA_EXECUTABLE", "MICA_PYTHON") or cfg.get("mica_python") or sys.executable,
            "default_executables": ("python3", "python"),
            "repo_dir": repo,
            "weights_dir": _env_first("MICA_WEIGHTS_DIR") or (str(Path(str(repo)) / "trained_models") if repo else None),
            "required_runtime": ("executable", "repo_dir", "weights_dir"),
            "recommended_action": "Install/configure MICA or set MICA_REPO_DIR/MICA_WEIGHTS_DIR.",
        }
    if name == "topaz":
        cfg = _safe_load_yaml(cfg_root / "topaz.yaml")
        return {
            "category": "picking",
            "executable": _env_first("TOPAZ_EXECUTABLE") or cfg.get("topaz_executable"),
            "default_executables": ("topaz",),
            "repo_dir": None,
            "weights_dir": cfg.get("topaz_model_path") or cfg.get("default_model_path"),
            "required_runtime": ("executable",),
            "recommended_action": "Install Topaz or set TOPAZ_EXECUTABLE.",
        }
    if name == "cryodrgn":
        cfg = _safe_load_yaml(cfg_root / "cryodrgn.yaml")
        return {
            "category": "heterogeneity",
            "executable": _env_first("CRYODRGN_EXECUTABLE", "CRYODRGN_PYTHON") or cfg.get("cryodrgn_executable"),
            "default_executables": ("cryodrgn",),
            "repo_dir": None,
            "weights_dir": None,
            "required_runtime": ("executable",),
            "recommended_action": "Install cryoDRGN in the active environment or set CRYODRGN_EXECUTABLE.",
        }
    if name == "relion":
        relion_bin = _env_first("RELION_BIN_DIR")

        def _relion_cmd(env_name: str, default_name: str) -> str:
            explicit = _env_first(env_name)
            if explicit:
                return explicit
            return str(Path(relion_bin) / default_name) if relion_bin else default_name

        return {
            "category": "reconstruction",
            "relion_commands": (
                _relion_cmd("RELION_AUTOPICK_EXECUTABLE", "relion_autopick"),
                _relion_cmd("RELION_PREPROCESS_EXECUTABLE", "relion_preprocess"),
                _relion_cmd("RELION_REFINE_EXECUTABLE", "relion_refine"),
                "relion_ctf_refine",
                "relion_particle_polish",
                _relion_cmd("RELION_POSTPROCESS_EXECUTABLE", "relion_postprocess"),
                _relion_cmd("RELION_PIPELINER_EXECUTABLE", "relion_pipeliner"),
                _relion_cmd("RELION_STAR_PRINTTABLE_EXECUTABLE", "relion_star_printtable"),
            ),
            "required_runtime": ("relion_commands",),
            "recommended_action": "Install RELION CLI tools or set RELION_BIN_DIR/RELION_*_EXECUTABLE.",
        }
    return {"category": "unknown", "required_runtime": ()}


def _path_component_ready(access: Mapping[str, Any], *, require_non_empty_dir: bool = False) -> bool:
    if not access.get("exists") or access.get("not_accessible"):
        return False
    if require_non_empty_dir and access.get("is_dir"):
        try:
            return any(Path(str(access["path"])).iterdir())
        except (OSError, PermissionError):
            return False
    return True


def preflight_backend(
    backend_name: str,
    *,
    project_root: Optional[Path] = None,
    sequence_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    dry_run: bool = False,
    config_root: Optional[Path] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Structured runtime preflight for one backend."""
    name = str(backend_name).strip().lower()
    definition = _backend_definition(name, config_root=config_root)
    params = dict(params or {})
    warnings: List[str] = []
    missing_dependencies: List[str] = []
    missing_inputs: List[str] = []
    path_checks: Dict[str, Any] = {}

    exe_override = (
        params.get("executable")
        or params.get(f"{name}_executable")
        or params.get("model_angelo_executable")
        or params.get("topaz_executable")
        or params.get("cryodrgn_executable")
    )
    executable = resolve_executable(
        str(exe_override).strip() if exe_override else definition.get("executable"),
        default_names=tuple(definition.get("default_executables") or ()),
        env_vars=tuple(definition.get("executable_env_vars") or ()),
        candidate_dirs=tuple(definition.get("candidate_dirs") or ()),
    )
    executable_access = _first_access(executable)
    executable_status = _status_from_access(executable_access, require_executable=True)
    if executable.get("available"):
        executable_status = "ok"
    if "executable" in definition.get("required_runtime", ()):
        path_checks["executable"] = executable
        if not executable.get("available"):
            missing_dependencies.append(f"{name}_executable")

    repo_dir = params.get("repo_dir", definition.get("repo_dir"))
    if "repo_dir" in definition.get("required_runtime", ()) or repo_dir:
        repo_access = check_path_access(repo_dir)
        path_checks["repo_dir"] = repo_access
        if "repo_dir" in definition.get("required_runtime", ()) and not _path_component_ready(repo_access):
            missing_dependencies.append(f"{name}_repo_dir")
    else:
        repo_access = None

    weights_dir = params.get("weights_dir", definition.get("weights_dir"))
    if "weights_dir" in definition.get("required_runtime", ()) or weights_dir:
        weights_access = check_path_access(weights_dir)
        path_checks["weights_dir"] = weights_access
        if "weights_dir" in definition.get("required_runtime", ()) and weights_dir and not _path_component_ready(
            weights_access, require_non_empty_dir=True
        ):
            missing_dependencies.append(f"{name}_weights_dir")
    else:
        weights_access = None

    relion_status = "ok"
    if "relion_commands" in definition.get("required_runtime", ()):
        command_checks = {
            cmd: resolve_executable(cmd, default_names=(cmd,)) for cmd in definition.get("relion_commands", ())
        }
        path_checks["relion_commands"] = command_checks
        missing = [cmd for cmd, check in command_checks.items() if not check.get("available")]
        if missing:
            missing_dependencies.extend(missing)
            relion_status = "missing"

    out = output_dir or Path(project_root or ".") / ".cryoem_agent" / "backend_preflight" / name
    output_check = check_output_dir_writable(out)
    path_checks["output_dir"] = output_check
    if not output_check.get("writable_output_dir"):
        missing_inputs.append("output_dir")
    if name in {"modelangelo", "emodelx", "mica"} and not sequence_path and not dry_run:
        missing_inputs.append("sequence_path")

    for label, check in path_checks.items():
        if isinstance(check, dict) and check.get("path_stale"):
            warnings.append(f"{label} has stale path markers: {','.join(check.get('stale_reasons') or [])}")

    runtime_missing = list(dict.fromkeys(missing_dependencies))
    real_run_available = not runtime_missing and not missing_inputs
    dry_run_available = not missing_inputs if dry_run else real_run_available
    if dry_run and runtime_missing:
        warnings.append(
            "dry_run_available because missing executable/repo/weights do not block dry-run; real_run_available=false"
        )

    parts: List[str] = []
    if missing_inputs:
        parts.append(f"missing_or_unwritable_inputs: {','.join(dict.fromkeys(missing_inputs))}")
    if runtime_missing:
        parts.append(f"missing_runtime_dependencies: {','.join(runtime_missing)}")
    blocked_reason = "; ".join(parts) if parts else None
    output_dir_status = "ok" if output_check.get("writable_output_dir") else _status_from_access(output_check)
    status = {
        "backend_name": name,
        "category": definition.get("category", "unknown"),
        "available": bool(dry_run_available if dry_run else real_run_available),
        "dry_run": bool(dry_run),
        "input_suitable": not missing_inputs,
        "dry_run_available": bool(dry_run_available),
        "real_run_available": bool(real_run_available),
        "executable_path": executable.get("resolved") or (executable_access or {}).get("path"),
        "executable_status": relion_status if name == "relion" else executable_status,
        "repo_dir_status": _status_from_access(repo_access) if repo_access is not None else "not_required_for_dry_run",
        "weights_status": _status_from_access(weights_access) if weights_access is not None else "not_required_for_dry_run",
        "python_module_status": "not_required_for_dry_run",
        "gpu_status": "unknown" if os.environ.get("CUDA_VISIBLE_DEVICES") else "not_required_for_dry_run",
        "output_dir_status": output_dir_status,
        "execution_mode": "dry_run" if dry_run and dry_run_available else ("real" if real_run_available else "unavailable"),
        "missing_inputs": list(dict.fromkeys(missing_inputs)),
        "missing_dependencies": runtime_missing,
        "warnings": warnings,
        "blocked_reason": blocked_reason,
        "recommended_action": str(definition.get("recommended_action") or "Install/configure backend dependencies, or run dry-run for planning only."),
        "fallback_candidates": list(definition.get("fallback_candidates") or []),
        "environment_fingerprint": environment_fingerprint(),
        "path_checks": path_checks,
    }
    if dry_run and not real_run_available and dry_run_available:
        status["executable_status"] = status["executable_status"] or "not_required_for_dry_run"
    return status


def _iter_scan_files(root: Path) -> Iterable[Path]:
    try:
        for path in root.rglob("*"):
            try:
                if path.is_file() and path.suffix.lower() in _TEXT_SUFFIXES:
                    yield path
            except (OSError, PermissionError):
                continue
    except (OSError, PermissionError):
        return


def scan_stale_paths(root_dir: Any) -> Dict[str, Any]:
    """Scan text-like files under ``root_dir`` for stale or missing absolute paths."""
    root_text = _stringify_path(root_dir)
    findings: List[Dict[str, Any]] = []
    root_access = check_path_access(root_text)
    if not root_text or not root_access.get("exists") or not root_access.get("is_dir"):
        return {"root_dir": root_text, "root_access": root_access, "findings": findings, "finding_count": 0}
    root = Path(root_text)
    seen: set[Tuple[str, int, str]] = set()
    for file_path in _iter_scan_files(root):
        try:
            lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except (OSError, PermissionError):
            continue
        for lineno, line in enumerate(lines, start=1):
            for match in _ABS_PATH_RE.finditer(line):
                candidate = match.group(0).rstrip(".,;:)'\" ]}")
                access = check_path_access(candidate)
                if not access.get("path_stale"):
                    continue
                key = (str(file_path), lineno, candidate)
                if key in seen:
                    continue
                seen.add(key)
                findings.append(
                    {
                        "file": str(file_path),
                        "line": lineno,
                        "path": candidate,
                        "stale_reasons": list(access.get("stale_reasons") or []),
                        "not_accessible": bool(access.get("not_accessible")),
                        "accessible": not bool(access.get("not_accessible")),
                        "exists": access.get("exists"),
                        "suggested_fix": _suggested_fix(candidate),
                    }
                )
    return {"root_dir": str(root), "root_access": root_access, "findings": findings, "finding_count": len(findings)}


def _runtime_backend_names() -> Tuple[str, ...]:
    names = ["modelangelo", "emodelx", "mica", "topaz", "cryodrgn"]
    names.append("relion")
    return tuple(names)


def preflight_all_backends(
    *,
    project_root: Path,
    sequence_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    dry_run: bool = True,
    config_root: Optional[Path] = None,
    backends: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Run runtime preflight for all supported/registered backend interfaces."""
    backend_names = tuple(backends or _runtime_backend_names())
    results = {
        name: preflight_backend(
            name,
            project_root=project_root,
            sequence_path=sequence_path,
            output_dir=(Path(output_dir) / name if output_dir else None),
            dry_run=dry_run,
            config_root=config_root,
        )
        for name in backend_names
    }
    fp = environment_fingerprint()
    return {
        "project_root": str(project_root),
        "sequence_path": str(sequence_path) if sequence_path else None,
        "dry_run": bool(dry_run),
        "current_user": fp["current_user"],
        "hostname": fp["hostname"],
        "python_executable": fp["sys_executable"],
        "environment_fingerprint": fp,
        "backends": results,
    }


def summarize_backend_matrix(report: Mapping[str, Any]) -> str:
    """Render a compact Markdown summary for backend runtime diagnostics."""
    lines = [
        "# CryoAgent backend runtime preflight",
        "",
        f"- Project: `{report.get('project_root')}`",
        f"- Dry run: `{report.get('dry_run')}`",
        "",
        "| Backend | Category | Dry-run | Real-run | Missing | Warnings |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    backends = report.get("backends") or {}
    if isinstance(backends, Mapping):
        for name, row_any in backends.items():
            row = row_any if isinstance(row_any, Mapping) else {}
            missing = ", ".join(list(row.get("missing_inputs") or []) + list(row.get("missing_dependencies") or []))
            warnings = "; ".join(str(w) for w in row.get("warnings") or [])
            lines.append(
                "| {name} | {category} | {dry} | {real} | {missing} | {warnings} |".format(
                    name=name,
                    category=row.get("category", ""),
                    dry=row.get("dry_run_available"),
                    real=row.get("real_run_available"),
                    missing=missing or "-",
                    warnings=warnings or "-",
                )
            )
    stale = report.get("stale_paths")
    if isinstance(stale, Mapping):
        lines.extend(["", "## Stale path findings", ""])
        findings = stale.get("findings") or []
        if findings:
            for item in findings[:100]:
                if not isinstance(item, Mapping):
                    continue
                lines.append(
                    f"- `{item.get('path')}` in `{item.get('file')}:{item.get('line')}` "
                    f"({', '.join(item.get('stale_reasons') or [])})"
                )
        else:
            lines.append("- None")
    return "\n".join(lines) + "\n"

