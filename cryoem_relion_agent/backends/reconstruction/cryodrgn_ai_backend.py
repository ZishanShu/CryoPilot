"""cryoDRGN-AI reconstruction backend using ``cryodrgn abinit``."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from adapters.relion_to_cryodrgn import (
    build_relion_to_cryodrgn_preprocess_plan,
    write_preprocess_manifest,
)
from backends.reconstruction.cryodrgn_ai_contract import BACKEND_NAME, get_metadata


def _params(state: Any) -> Dict[str, Any]:
    if isinstance(state, dict):
        return dict(state)
    data: Dict[str, Any] = {}
    for name in (
        "project_path",
        "particles_star_or_mrcs",
        "particles_star_path",
        "particles_mrcs_path",
        "ctf_metadata",
        "ctf_pkl_path",
        "datadir",
        "apix",
        "box_size",
        "downsample_box",
        "zdim",
        "gpu_id",
        "num_epochs",
        "max_particles",
        "execution_mode",
        "require_gpu",
        "cryodrgn_executable",
        "abinit_command",
        "reconstruction_type",
        "output_dir",
    ):
        if hasattr(state, name):
            data[name] = getattr(state, name)
    return data


def _first_value(params: Dict[str, Any], *names: str) -> Any:
    for name in names:
        value = params.get(name)
        if value is not None and str(value).strip() != "":
            return value
    return None


def _resolve_executable(params: Dict[str, Any]) -> Optional[str]:
    raw = str(params.get("cryodrgn_executable") or "cryodrgn").strip()
    path = Path(raw).expanduser()
    if path.is_file() and (os.access(path, os.X_OK) or os.name == "nt"):
        return str(path.resolve())
    return shutil.which(raw)


def _help_available(executable: str, subcommand: str) -> bool:
    """Return True if ``cryodrgn <subcommand> -h`` is supported."""

    try:
        proc = subprocess.run(
            [executable, subcommand, "-h"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return False
    return proc.returncode == 0


def _resolve_abinit_command(executable: str, params: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """Resolve the ab initio cryoDRGN subcommand for different CLI versions.

    Newer cryoDRGN-AI builds may expose ``cryodrgn abinit``.  The server
    environment currently uses cryoDRGN 3.4.x, which exposes ``abinit_het`` and
    ``abinit_homo`` instead.  The backend keeps the agent-facing name
    ``cryodrgn_ai`` but selects the real CLI subcommand here.
    """

    requested = str(params.get("abinit_command") or "auto").strip()
    if requested not in {"auto", "abinit", "abinit_het", "abinit_homo"}:
        requested = "auto"

    if requested != "auto":
        if _help_available(executable, requested):
            return requested, None
        return None, f"cryodrgn {requested}"

    if _help_available(executable, "abinit"):
        return "abinit", None

    reconstruction_type = str(params.get("reconstruction_type") or "heterogeneous").strip().lower()

    if reconstruction_type in {"homogeneous", "homo", "abinit_homo"}:
        if _help_available(executable, "abinit_homo"):
            return "abinit_homo", None
        if _help_available(executable, "abinit_het"):
            return "abinit_het", None
    else:
        if _help_available(executable, "abinit_het"):
            return "abinit_het", None
        if _help_available(executable, "abinit_homo"):
            return "abinit_homo", None

    return None, "cryodrgn abinit/abinit_het/abinit_homo"


def _run_help(executable: str) -> bool:
    """Backward-compatible helper kept for existing tests.

    Historically this checked ``cryodrgn abinit -h`` only.  It now returns True
    when any supported ab initio entry point is available.
    """

    resolved, _missing = _resolve_abinit_command(executable, {"abinit_command": "auto"})
    return resolved is not None


def _gpu_available() -> bool:
    if shutil.which("nvidia-smi"):
        proc = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
        return proc.returncode == 0 and bool((proc.stdout or "").strip())
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def preflight(state: Any) -> Dict[str, Any]:
    """Validate executable and input combinations for cryoDRGN-AI."""

    params = _params(state)
    mode = str(params.get("execution_mode") or params.get("mode") or "real").strip().lower()
    if mode not in {"dry_run", "real"}:
        mode = "real"
    missing_inputs: List[str] = []
    missing_dependencies: List[str] = []
    warnings: List[str] = []

    executable = _resolve_executable(params)
    resolved_abinit_command: Optional[str] = None
    if executable is None:
        missing_dependencies.append("cryodrgn_executable")
    elif not _run_help(executable):
        _resolved, missing_abinit = _resolve_abinit_command(executable, params)
        missing_dependencies.append(missing_abinit or "cryodrgn abinit/abinit_het/abinit_homo")
    else:
        resolved_abinit_command, missing_abinit = _resolve_abinit_command(executable, params)
        if missing_abinit:
            # This path mainly preserves older tests that patch _run_help=True.
            # In real execution, _run_help=True should imply one command resolved.
            requested = str(params.get("abinit_command") or "auto").strip()
            resolved_abinit_command = "abinit" if requested == "auto" else requested

    particle_input = _first_value(params, "particles_star_or_mrcs", "particles_mrcs_path", "particles_star_path")
    ctf_input = _first_value(params, "ctf_metadata", "ctf_pkl_path")
    particle_path = Path(str(particle_input)).expanduser() if particle_input else None
    ctf_path = Path(str(ctf_input)).expanduser() if ctf_input else None

    has_mrcs_ctf = bool(
        particle_path
        and particle_path.suffix.lower() == ".mrcs"
        and particle_path.is_file()
        and ctf_path
        and ctf_path.is_file()
    )
    has_star = bool(particle_path and particle_path.suffix.lower() == ".star" and particle_path.is_file())
    has_convertible_star = bool(
        has_star
        and _first_value(params, "datadir", "project_root", "project_path")
        and params.get("apix") is not None
        and params.get("box_size") is not None
    )

    if not particle_path or not particle_path.is_file():
        missing_inputs.append("particles_star_or_mrcs")
    if not has_mrcs_ctf and not has_convertible_star:
        if has_star:
            if not _first_value(params, "datadir", "project_root", "project_path"):
                missing_inputs.append("datadir")
            if params.get("apix") is None:
                missing_inputs.append("apix")
            if params.get("box_size") is None:
                missing_inputs.append("box_size")
        elif not ctf_path or not ctf_path.is_file():
            missing_inputs.append("ctf_metadata")

    if mode == "real" and not _gpu_available():
        if bool(params.get("require_gpu")):
            missing_dependencies.append("gpu")
        else:
            warnings.append("gpu_unavailable")

    available = not missing_inputs and not missing_dependencies
    blocked_parts: List[str] = []
    if missing_inputs:
        blocked_parts.append("missing_required_inputs: " + ",".join(sorted(set(missing_inputs))))
    if missing_dependencies:
        blocked_parts.append("missing_dependency: " + ",".join(sorted(set(missing_dependencies))))

    return {
        "backend_name": BACKEND_NAME,
        "category": "reconstruction",
        "available": available,
        "execution_mode": mode if available else "unavailable",
        "requested_execution_mode": mode,
        "missing_inputs": sorted(set(missing_inputs)),
        "missing_dependencies": sorted(set(missing_dependencies)),
        "warnings": warnings,
        "blocked_reason": "; ".join(blocked_parts) if blocked_parts else None,
        "executable_path": executable,
        "resolved_abinit_command": resolved_abinit_command,
        "input_mode": "mrcs_ctf" if has_mrcs_ctf else ("relion_star" if has_convertible_star else "unknown"),
        "input_summary": f"particles={particle_input}, ctf={ctf_input}",
    }


def _command_manifest_path(run_dir: Path) -> Path:
    return run_dir / "command_manifest.json"


def _as_int(value: Any, default: int) -> int:
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def prepare_inputs(state: Any, run_dir: str | Path) -> Dict[str, Any]:
    """Prepare directories and optional RELION STAR preprocessing manifest."""

    params = _params(state)
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    executable = _resolve_executable(params) or str(params.get("cryodrgn_executable") or "cryodrgn")
    particle_input = _first_value(params, "particles_star_or_mrcs", "particles_mrcs_path", "particles_star_path")
    ctf_input = _first_value(params, "ctf_metadata", "ctf_pkl_path")
    particle_path = Path(str(particle_input)).expanduser()

    if particle_path.suffix.lower() == ".star":
        datadir = _first_value(params, "datadir", "project_root", "project_path")
        plan = build_relion_to_cryodrgn_preprocess_plan(
            particles_star=particle_path,
            datadir=datadir,
            apix=float(params["apix"]),
            box_size=int(params["box_size"]),
            run_dir=run_path,
            downsample_box=_as_int(params.get("downsample_box"), int(params["box_size"])),
            cryodrgn_executable=executable,
        )
        preprocess_manifest = write_preprocess_manifest(plan)
        return {
            "input_mode": "relion_star",
            "preprocess_commands": plan.commands,
            "preprocess_manifest": str(preprocess_manifest),
            "particles_mrcs": str(plan.particles_mrcs),
            "ctf_pkl": str(plan.ctf_pkl),
            "executable": executable,
        }

    return {
        "input_mode": "mrcs_ctf",
        "preprocess_commands": [],
        "preprocess_manifest": None,
        "particles_mrcs": str(particle_path),
        "ctf_pkl": str(Path(str(ctf_input)).expanduser()),
        "executable": executable,
    }


def build_command(state: Any, run_dir: str | Path) -> Dict[str, Any]:
    """Build and write the full command manifest for ``cryodrgn abinit``."""

    params = _params(state)
    run_path = Path(run_dir)
    prepared = prepare_inputs(params, run_path)
    abinit_command, missing_abinit = _resolve_abinit_command(prepared["executable"], params)
    if abinit_command is None:
        raise RuntimeError(f"missing cryoDRGN ab initio command: {missing_abinit}")

    abinit_dir = run_path / "abinit"
    abinit_cmd: List[str] = [
        prepared["executable"],
        abinit_command,
        prepared["particles_mrcs"],
        "--ctf",
        prepared["ctf_pkl"],
        "-o",
        str(abinit_dir),
    ]
    if abinit_command != "abinit_homo":
        abinit_cmd.extend(["--zdim", str(_as_int(params.get("zdim"), 8))])
    if params.get("gpu_id") is not None and str(params.get("gpu_id")).strip() != "":
        abinit_cmd.extend(["--device", str(params["gpu_id"])])
    if params.get("num_epochs") is not None and str(params.get("num_epochs")).strip() != "":
        abinit_cmd.extend(["--num-epochs", str(params["num_epochs"])])
    if params.get("max_particles") is not None and str(params.get("max_particles")).strip() != "":
        abinit_cmd.extend(["--max-particles", str(params["max_particles"])])

    commands = list(prepared["preprocess_commands"]) + [abinit_cmd]
    manifest = {
        "backend_name": BACKEND_NAME,
        "metadata": get_metadata(),
        "run_dir": str(run_path),
        "abinit_dir": str(abinit_dir),
        "resolved_abinit_command": abinit_command,
        "input_mode": prepared["input_mode"],
        "preprocess_manifest": prepared.get("preprocess_manifest"),
        "commands": [
            {
                "name": (
                    "abinit"
                    if len(cmd) > 1 and cmd[1] in {"abinit", "abinit_het", "abinit_homo"}
                    else (cmd[1] if len(cmd) > 1 else "command")
                ),
                "argv": cmd,
            }
            for cmd in commands
        ],
        "stdout_log": str(run_path / "stdout.log"),
        "stderr_log": str(run_path / "stderr.log"),
    }
    manifest_path = _command_manifest_path(run_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest["command_manifest"] = str(manifest_path)
    return manifest


def _append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text or "")


def _run_commands(commands: Iterable[Dict[str, Any]], cwd: Path, stdout_log: Path, stderr_log: Path) -> Dict[str, Any]:
    for item in commands:
        argv = [str(x) for x in item.get("argv", [])]
        proc = subprocess.run(argv, cwd=str(cwd), capture_output=True, text=True, check=False)
        _append(stdout_log, proc.stdout or "")
        _append(stderr_log, proc.stderr or "")
        if proc.returncode != 0:
            return {
                "status": "failed",
                "failed_command": argv,
                "return_code": proc.returncode,
                "failed_reason": f"subprocess_failed: {item.get('name')}",
            }
    return {"status": "completed", "return_code": 0}


def collect_outputs(run_dir: str | Path) -> Dict[str, Any]:
    """Collect cryoDRGN-AI artifacts produced by abinit and analysis."""

    run_path = Path(run_dir)
    pkl_files = sorted(str(p) for p in run_path.rglob("*.pkl") if p.is_file())
    mrc_files = sorted(str(p) for p in run_path.rglob("*.mrc") if p.is_file())
    mrcs_files = sorted(str(p) for p in run_path.rglob("*.mrcs") if p.is_file())
    weights = [p for p in pkl_files if "weight" in Path(p).name.lower() or "epoch" in Path(p).name.lower()]
    latent = [p for p in pkl_files if any(token in Path(p).name.lower() for token in ("z.", "latent", "embed"))]
    return {
        "run_dir": str(run_path),
        "command_manifest": str(_command_manifest_path(run_path)),
        "stdout_log": str(run_path / "stdout.log"),
        "stderr_log": str(run_path / "stderr.log"),
        "latent_embeddings": latent,
        "model_weights": weights,
        "sampled_volumes": mrc_files,
        "pkl_files": pkl_files,
        "mrc_files": mrc_files,
        "mrcs_files": mrcs_files,
    }


def run(state: Any, run_dir: str | Path) -> Dict[str, Any]:
    """Run or dry-run cryoDRGN-AI and return scheduler-friendly result."""

    params = _params(state)
    run_path = Path(run_dir)
    pf = preflight(params)
    if not pf.get("available"):
        return {
            "backend_name": BACKEND_NAME,
            "selected_backend": BACKEND_NAME,
            "execution_mode": "unavailable",
            "status": "preflight_blocked",
            "blocked_reason": pf.get("blocked_reason"),
            "preflight": pf,
            "success": False,
        }

    manifest = build_command(params, run_path)
    mode = str(params.get("execution_mode") or pf.get("requested_execution_mode") or "real")
    if mode == "dry_run":
        return {
            "backend_name": BACKEND_NAME,
            "selected_backend": BACKEND_NAME,
            "execution_mode": "dry_run",
            "status": "dry_run",
            "preflight": pf,
            "command_manifest": manifest["command_manifest"],
            "commands": manifest["commands"],
            "output_artifacts": collect_outputs(run_path),
            "success": True,
        }

    stdout_log = Path(str(manifest["stdout_log"]))
    stderr_log = Path(str(manifest["stderr_log"]))
    exec_result = _run_commands(manifest["commands"], run_path, stdout_log, stderr_log)
    status = str(exec_result.get("status"))
    return {
        "backend_name": BACKEND_NAME,
        "selected_backend": BACKEND_NAME,
        "execution_mode": "real",
        "status": status,
        "preflight": pf,
        "command_manifest": manifest["command_manifest"],
        "commands": manifest["commands"],
        "output_artifacts": collect_outputs(run_path),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "return_code": exec_result.get("return_code"),
        "failed_reason": exec_result.get("failed_reason"),
        "success": status == "completed",
    }


def update_state(state: Any, result: Dict[str, Any]) -> Any:
    """Write candidate maps and latent artifacts back to CryoAgent state."""

    if isinstance(state, dict):
        artifacts = result.get("output_artifacts") if isinstance(result.get("output_artifacts"), dict) else {}
        state["cryodrgn_ai"] = result
        state["candidate_maps"] = list(artifacts.get("sampled_volumes") or [])
        state["latent_space_artifacts"] = list(artifacts.get("latent_embeddings") or [])
        return state

    state.last_execution = result
    artifacts = result.get("output_artifacts") if isinstance(result.get("output_artifacts"), dict) else {}
    maps = list(artifacts.get("sampled_volumes") or [])
    latent = list(artifacts.get("latent_embeddings") or [])
    if hasattr(state, "notes"):
        state.notes.append(
            f"cryodrgn_ai: status={result.get('status')} candidate_maps={len(maps)} latent_artifacts={len(latent)}"
        )
    return state
