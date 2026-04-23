"""Execute ModelAngelo ``build`` with sequence (map + FASTA + output dir)."""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from schemas.action import Action
from schemas.execution_result import ExecutionResult
from schemas.sample_state import SampleState

logger = logging.getLogger(__name__)

DEFAULT_MODEL_BUILDERS: Dict[str, Any] = {
    "model_angelo": {
        "executable": "model_angelo",
        "subcommand_with_seq": ["build"],
        "flags": {"map": "-v", "protein_fasta": "-pf", "output": "-o"},
        "output_runs_subdir": "modelangelo_runs",
        "torch_home": None,
        "weights_relpath": "hub/checkpoints/model_angelo_v1.0",
    }
}


def _coerce_subcommand_list(val: Any) -> List[str]:
    """Allow YAML to use either a list or a single string (e.g. ``build``)."""
    if val is None:
        return ["build"]
    if isinstance(val, str):
        v = val.strip()
        return [v] if v else ["build"]
    if isinstance(val, (list, tuple)):
        out = [str(x).strip() for x in val if str(x).strip()]
        return out if out else ["build"]
    return ["build"]


def _load_model_angelo_config(config_path: Path) -> Dict[str, Any]:
    try:
        if not config_path.exists():
            return DEFAULT_MODEL_BUILDERS["model_angelo"]
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        ma = data.get("model_angelo")
        if ma is None:
            ma = data.get("modelangelo") or {}
        if not isinstance(ma, dict):
            ma = {}
        merged = {**DEFAULT_MODEL_BUILDERS["model_angelo"], **ma}
        merged["flags"] = {**DEFAULT_MODEL_BUILDERS["model_angelo"]["flags"], **ma.get("flags", {})}
        merged["subcommand_with_seq"] = _coerce_subcommand_list(merged.get("subcommand_with_seq"))
        return merged
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Failed to load model_builders.yaml, using defaults: %s", exc)
        return DEFAULT_MODEL_BUILDERS["model_angelo"]


def _is_executable_path_like(s: str) -> bool:
    """True when ``executable`` should be resolved on disk (not looked up only via PATH)."""
    if s.startswith("~"):
        return True
    p = Path(s)
    if p.is_absolute():
        return True
    if "/" in s or s.startswith((".", "./", "../")):
        return True
    return False


def _resolve_executable(exe_raw: Any) -> tuple[str, Optional[str]]:
    """
    Return (argv0, error_message).

    Bare names (e.g. ``model_angelo``) are passed through for PATH lookup.
    Path-like values must exist and be regular files.
    """
    if exe_raw is None:
        return "model_angelo", None
    s = str(exe_raw).strip()
    if not s:
        return "model_angelo", None
    if not _is_executable_path_like(s):
        return s, None
    p = Path(s).expanduser().resolve()
    if not p.is_file():
        return str(p), f"executable not found: {p}"
    return str(p), None


def _argv_to_shell_string(argv: List[str]) -> str:
    return " ".join(shlex.quote(a) for a in argv)


def _pick_first_cif(output_dir: Path) -> Optional[Path]:
    try:
        cifs = sorted(output_dir.glob("*.cif"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cifs:
            return cifs[0]
    except OSError:
        pass
    return None


_MODEL_ANGELO_STRUCTURE_GLOBS: tuple = ("*.cif", "*.pdb")
_MODEL_ANGELO_SUMMARY_GLOBS: tuple = (
    "*summary*",
    "*ranking*",
    "*score*",
)


def _harvest_model_angelo_artifacts(output_dir: Path) -> List[str]:
    """Collect ModelAngelo success products from ``output_dir``.

    Priority is the primary structure files (``.cif`` / ``.pdb``) followed by
    summary / ranking / score auxiliary outputs.  Log files (``*.log``,
    ``*.txt``) are intentionally excluded — they are exposed via
    ``log_path`` separately, not as main artifacts.
    """
    if not output_dir.is_dir():
        return []
    found: List[str] = []
    seen: set[str] = set()

    def _add(p: Path) -> None:
        if not p.is_file():
            return
        s = str(p)
        if s in seen:
            return
        seen.add(s)
        found.append(s)

    for pattern in _MODEL_ANGELO_STRUCTURE_GLOBS:
        for p in sorted(output_dir.rglob(pattern), key=lambda q: q.stat().st_mtime, reverse=True):
            _add(p)
    for pattern in _MODEL_ANGELO_SUMMARY_GLOBS:
        for p in sorted(output_dir.rglob(pattern)):
            # Skip obvious log files even if they happen to match.
            if p.suffix.lower() in {".log", ".txt"}:
                continue
            _add(p)
    return found


def _resolve_torch_home(cfg: Dict[str, Any]) -> Optional[Path]:
    """
    Resolve TORCH_HOME with precedence:
    1) process env TORCH_HOME
    2) config torch_home
    """
    env_val = os.getenv("TORCH_HOME")
    if env_val and env_val.strip():
        return Path(env_val.strip()).expanduser().resolve()
    cfg_val = cfg.get("torch_home")
    if cfg_val and str(cfg_val).strip():
        return Path(str(cfg_val).strip()).expanduser().resolve()
    return None


def _resolve_weights_path(torch_home: Optional[Path], cfg: Dict[str, Any]) -> Optional[Path]:
    if torch_home is None:
        return None
    rel = str(cfg.get("weights_relpath", "hub/checkpoints/model_angelo_v1.0")).strip()
    return (torch_home / rel).resolve()


def run_model_angelo(
    action: Action,
    project_root: Path,
    state: Optional[SampleState],
    dry_run: bool,
    model_builders_yaml: Path,
) -> Dict[str, Any]:
    """
    Run ``model_angelo build -v MAP -pf FASTA -o OUTPUT_DIR``.

    Paths are resolved to absolute. On failure, returns structured error without raising.
    """
    cfg = _load_model_angelo_config(model_builders_yaml)
    map_raw = (action.parameters or {}).get("map_path") or (state.map_path if state else None)
    fasta_raw = (action.parameters or {}).get("sequence_path") or (state.sequence_path if state else None)

    if not map_raw:
        er = ExecutionResult(
            action_name=action.name.value,
            dry_run=dry_run,
            status="error_validation",
            error_message="missing map_path (action.parameters and state.map_path both empty)",
        )
        return er.to_serializable_dict()

    if not fasta_raw:
        er = ExecutionResult(
            action_name=action.name.value,
            dry_run=dry_run,
            status="error_validation",
            error_message="missing sequence_path (action.parameters and state.sequence_path both empty)",
        )
        return er.to_serializable_dict()

    map_path = Path(map_raw).expanduser().resolve()
    fasta_path = Path(fasta_raw).expanduser().resolve()

    if not map_path.is_file():
        er = ExecutionResult(
            action_name=action.name.value,
            dry_run=dry_run,
            status="error_validation",
            error_message=f"map file not found: {map_path}",
        )
        return er.to_serializable_dict()

    if not fasta_path.is_file():
        er = ExecutionResult(
            action_name=action.name.value,
            dry_run=dry_run,
            status="error_validation",
            error_message=f"FASTA file not found: {fasta_path}",
        )
        return er.to_serializable_dict()

    explicit_out_raw = (action.parameters or {}).get("output_dir")
    if explicit_out_raw and str(explicit_out_raw).strip():
        output_dir = Path(str(explicit_out_raw).strip()).expanduser().resolve()
    else:
        agent_root = project_root / ".cryoem_agent"
        runs_sub = cfg.get("output_runs_subdir", "modelangelo_runs")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = agent_root / runs_sub / stamp

    exe, exe_err = _resolve_executable(cfg.get("executable", "model_angelo"))
    if exe_err:
        er = ExecutionResult(
            action_name=action.name.value,
            dry_run=dry_run,
            status="error_validation",
            error_message=exe_err,
        )
        return er.to_serializable_dict()

    sub: List[str] = list(cfg.get("subcommand_with_seq") or ["build"])
    flags: Dict[str, str] = cfg.get("flags") or {}

    fm = flags.get("map", "-v")
    fp = flags.get("protein_fasta", "-pf")
    fo = flags.get("output", "-o")

    argv: List[str] = [exe] + sub + [fm, str(map_path), fp, str(fasta_path), fo, str(output_dir)]
    cmd_str = _argv_to_shell_string(argv)

    torch_home = _resolve_torch_home(cfg)
    weights_path = _resolve_weights_path(torch_home, cfg)
    weights_cached = bool(weights_path and weights_path.exists())
    env = os.environ.copy()
    if torch_home is not None:
        env["TORCH_HOME"] = str(torch_home)

    cache_msg = (
        f"TORCH_HOME={torch_home}; weights_path={weights_path}; cached={weights_cached}"
        if torch_home is not None
        else "TORCH_HOME not set; ModelAngelo will use default torch cache behavior"
    )

    if dry_run:
        er = ExecutionResult(
            action_name=action.name.value,
            command=cmd_str,
            command_argv=argv,
            dry_run=True,
            status="dry_run",
            return_code=0,
            stdout=f"[DRY-RUN] Would execute: {cmd_str}\n{cache_msg}",
            stderr="",
            output_dir=str(output_dir),
            generated_model_path=None,
            log_path=None,
            error_message=None,
        )
        logger.info("ModelAngelo dry-run: %s", cmd_str)
        return er.to_serializable_dict()

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        er = ExecutionResult(
            action_name=action.name.value,
            command=cmd_str,
            command_argv=argv,
            dry_run=False,
            status="error",
            error_message=f"failed to create output_dir: {exc}",
        )
        return er.to_serializable_dict()

    log_path = output_dir / "model_angelo_run.log"
    try:
        completed = subprocess.run(
            argv,
            cwd=str(project_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=86400,
            check=False,
        )
        log_body = (
            f"return_code={completed.returncode}\n"
            f"{cache_msg}\n\n--- stdout ---\n{completed.stdout}\n\n--- stderr ---\n{completed.stderr}\n"
        )
        log_path.write_text(log_body, encoding="utf-8")
    except subprocess.TimeoutExpired as exc:
        er = ExecutionResult(
            action_name=action.name.value,
            command=cmd_str,
            command_argv=argv,
            dry_run=False,
            status="error",
            stderr=str(exc),
            output_dir=str(output_dir),
            log_path=str(log_path) if log_path.exists() else None,
            error_message=f"ModelAngelo timeout: {exc}",
        )
        return er.to_serializable_dict()
    except Exception as exc:  # noqa: BLE001
        logger.exception("ModelAngelo subprocess failed: %s", exc)
        er = ExecutionResult(
            action_name=action.name.value,
            command=cmd_str,
            command_argv=argv,
            dry_run=False,
            status="error",
            output_dir=str(output_dir),
            error_message=str(exc),
        )
        return er.to_serializable_dict()

    gen_cif = _pick_first_cif(output_dir)
    ok = completed.returncode == 0
    harvested = _harvest_model_angelo_artifacts(output_dir) if ok else []
    er = ExecutionResult(
        action_name=action.name.value,
        command=cmd_str,
        command_argv=argv,
        dry_run=False,
        status="success" if ok else "failed",
        return_code=completed.returncode,
        stdout=((completed.stdout or "") + f"\n[weights] {cache_msg}").strip(),
        stderr=completed.stderr or "",
        output_dir=str(output_dir),
        generated_paths=harvested,
        generated_model_path=str(gen_cif.resolve()) if gen_cif else None,
        log_path=str(log_path),
        error_message=None if ok else f"model_angelo exited with code {completed.returncode}",
    )
    return er.to_serializable_dict()
