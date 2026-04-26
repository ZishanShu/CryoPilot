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
        # sampled-volume generation params
        "sample_volumes",
        "n_sampled_volumes",
        "representative_z_path",
        "sampled_volumes_dir",
        "eval_vol_device",
        "eval_vol_apix",
        "eval_vol_box_size",
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


def sample_representative_latents(run_dir: Path, z_path: Path, n_volumes: int = 5) -> Path:
    """Compute representative latent points via PCA (numpy SVD) and write to a txt file.

    Returns the path to ``representative_z.txt`` suitable for ``cryodrgn eval_vol --zfile``.
    Uses evenly-spaced quantiles along the first principal component so that the
    selected points span the full latent distribution.
    """
    import pickle  # stdlib — no sklearn dependency

    import numpy as np

    with open(z_path, "rb") as fh:
        z = pickle.load(fh)
    if hasattr(z, "numpy"):
        z = z.numpy()
    z = np.asarray(z, dtype=float)
    n_samples = z.shape[0]

    if n_samples <= n_volumes:
        selected_z = z
    else:
        z_centered = z - z.mean(axis=0)
        _U, _s, Vt = np.linalg.svd(z_centered, full_matrices=False)
        pc1 = Vt[0]
        proj = z_centered @ pc1
        sorted_idx = np.argsort(proj)
        indices = [
            sorted_idx[int(round(i * (n_samples - 1) / max(n_volumes - 1, 1)))]
            for i in range(n_volumes)
        ]
        selected_z = z[indices]

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    rep_z_path = run_path / "representative_z.txt"
    np.savetxt(str(rep_z_path), selected_z)
    return rep_z_path


def build_eval_vol_command(
    executable: str,
    weights_path: str,
    config_path: str,
    zfile_path: str,
    out_dir: str,
    *,
    apix: Optional[float] = None,
    box_size: Optional[int] = None,
    device: Optional[str] = None,
    prefix: str = "vol_",
) -> List[str]:
    """Return the argv list for cryoDRGN 3.4 ``cryodrgn eval_vol``.

    Canonical form:
        cryodrgn eval_vol <weights.pkl> -c <config.yaml> -o <out_dir> \\
            --prefix vol_ --zfile <representative_z.txt> --Apix <apix> -D <box>

    Notes:
    - Uses ``-c`` (short for ``--config``); ``config_path`` must point to a YAML
      file (e.g. ``abinit/config.yaml``), never a ``.pkl``.
    - Uses ``-D`` for the box size; cryoDRGN 3.4 does not accept ``--box``.
    - Always uses ``--zfile`` (not the ambiguous ``--z``) to read multiple
      latent rows from a text file.
    """
    argv: List[str] = [
        executable, "eval_vol", weights_path,
        "-c", config_path,
        "-o", out_dir,
        "--prefix", prefix,
        "--zfile", zfile_path,
    ]
    if apix is not None:
        argv.extend(["--Apix", str(apix)])
    if box_size is not None:
        argv.extend(["-D", str(box_size)])
    if device is not None:
        argv.extend(["--device", str(device)])
    return argv


def run_eval_vol(
    params: Dict[str, Any],
    run_dir: Path,
    *,
    abinit_dir: Optional[Path] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Sample representative latent points and run ``cryodrgn eval_vol``.

    Returns a dict with status, representative_z_path, sampled_volumes_dir,
    argv, and (in real mode) return_code / stdout_log / stderr_log.

    Sets ``TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`` in the subprocess env to avoid
    PyTorch 2.6+ ``weights_only=True`` failures on trusted local cryoDRGN outputs.
    """
    abinit_path = abinit_dir if abinit_dir is not None else (run_dir / "abinit")
    executable = _resolve_executable(params) or str(params.get("cryodrgn_executable") or "cryodrgn")
    n_volumes = _as_int(params.get("n_sampled_volumes"), 5)

    # Determine representative_z.txt path
    z_path_override = params.get("representative_z_path")
    if z_path_override and Path(str(z_path_override)).is_file():
        rep_z_path = Path(str(z_path_override))
    else:
        z_pkl = abinit_path / "z.pkl"
        if not z_pkl.is_file():
            return {"status": "skipped", "reason": f"z.pkl not found at {z_pkl}"}
        rep_z_path = sample_representative_latents(run_dir, z_pkl, n_volumes=n_volumes)

    weights_pkl = abinit_path / "weights.pkl"
    # cryoDRGN 3.4 abinit_het writes ``config.yaml`` (not ``config.pkl``).
    config_yaml = abinit_path / "config.yaml"
    sampled_volumes_dir = str(params.get("sampled_volumes_dir") or (run_dir / "sampled_volumes"))
    Path(sampled_volumes_dir).mkdir(parents=True, exist_ok=True)

    apix = params.get("eval_vol_apix")
    box_size = params.get("eval_vol_box_size")
    device = params.get("eval_vol_device")
    argv = build_eval_vol_command(
        executable,
        str(weights_pkl),
        str(config_yaml),
        str(rep_z_path),
        sampled_volumes_dir,
        apix=float(apix) if apix is not None else None,
        box_size=int(box_size) if box_size is not None else None,
        device=str(device) if device is not None else None,
    )

    if dry_run:
        return {
            "status": "dry_run",
            "representative_z_path": str(rep_z_path),
            "sampled_volumes_dir": sampled_volumes_dir,
            "argv": argv,
        }

    if not weights_pkl.is_file():
        return {
            "status": "skipped",
            "reason": f"weights.pkl not found at {weights_pkl}",
            "representative_z_path": str(rep_z_path),
            "sampled_volumes_dir": sampled_volumes_dir,
        }

    stdout_log = run_dir / "eval_vol_stdout.log"
    stderr_log = run_dir / "eval_vol_stderr.log"
    # TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 suppresses PyTorch 2.6+ weights_only
    # error for trusted local cryoDRGN checkpoint files only (subprocess scope).
    env = {**os.environ, "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}
    try:
        proc = subprocess.run(argv, cwd=str(run_dir), capture_output=True, text=True, env=env, check=False)
    except (FileNotFoundError, OSError) as exc:
        return {
            "status": "failed",
            "reason": str(exc),
            "representative_z_path": str(rep_z_path),
            "sampled_volumes_dir": sampled_volumes_dir,
            "argv": argv,
            "return_code": 127,
        }
    _append(stdout_log, proc.stdout or "")
    _append(stderr_log, proc.stderr or "")
    return {
        "status": "completed" if proc.returncode == 0 else "failed",
        "representative_z_path": str(rep_z_path),
        "sampled_volumes_dir": sampled_volumes_dir,
        "argv": argv,
        "return_code": proc.returncode,
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
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

    # When sample_volumes requested, include a planned eval_vol entry in the manifest
    # so dry-run callers can see the full intended command sequence.  In real mode
    # eval_vol is invoked separately via run_eval_vol() after abinit completes and
    # z.pkl exists, because the representative latent file is built at that point.
    eval_vol_planned: Optional[List[str]] = None
    if params.get("sample_volumes"):
        sampled_volumes_dir = str(params.get("sampled_volumes_dir") or (run_path / "sampled_volumes"))
        eval_vol_planned = build_eval_vol_command(
            prepared["executable"],
            str(abinit_dir / "weights.pkl"),
            str(abinit_dir / "config.yaml"),
            str(run_path / "representative_z.txt"),
            sampled_volumes_dir,
            apix=float(params["eval_vol_apix"]) if params.get("eval_vol_apix") is not None else None,
            box_size=int(params["eval_vol_box_size"]) if params.get("eval_vol_box_size") is not None else None,
            device=str(params["eval_vol_device"]) if params.get("eval_vol_device") is not None else None,
        )

    def _cmd_name(cmd: List[str]) -> str:
        if len(cmd) > 1 and cmd[1] in {"abinit", "abinit_het", "abinit_homo"}:
            return "abinit"
        if len(cmd) > 1 and cmd[1] == "eval_vol":
            return "eval_vol"
        return cmd[1] if len(cmd) > 1 else "command"

    manifest_commands = [{"name": _cmd_name(cmd), "argv": cmd} for cmd in commands]
    if eval_vol_planned is not None:
        manifest_commands.append({"name": "eval_vol", "argv": eval_vol_planned})

    manifest = {
        "backend_name": BACKEND_NAME,
        "metadata": get_metadata(),
        "run_dir": str(run_path),
        "abinit_dir": str(abinit_dir),
        "resolved_abinit_command": abinit_command,
        "input_mode": prepared["input_mode"],
        "preprocess_manifest": prepared.get("preprocess_manifest"),
        "commands": manifest_commands,
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
    """Collect cryoDRGN-AI artifacts produced by abinit and eval_vol."""

    run_path = Path(run_dir)
    pkl_files = sorted(str(p) for p in run_path.rglob("*.pkl") if p.is_file())
    mrc_files = sorted(str(p) for p in run_path.rglob("*.mrc") if p.is_file())
    mrcs_files = sorted(str(p) for p in run_path.rglob("*.mrcs") if p.is_file())
    weights = [p for p in pkl_files if "weight" in Path(p).name.lower() or "epoch" in Path(p).name.lower()]
    latent = [p for p in pkl_files if any(token in Path(p).name.lower() for token in ("z.", "latent", "embed"))]
    # sampled_volumes: only .mrc files produced by eval_vol in the sampled_volumes/ subdir
    sv_dir = run_path / "sampled_volumes"
    sampled_volumes = sorted(str(p) for p in sv_dir.glob("*.mrc") if p.is_file()) if sv_dir.is_dir() else []
    rep_z_path = run_path / "representative_z.txt"
    return {
        "run_dir": str(run_path),
        "command_manifest": str(_command_manifest_path(run_path)),
        "stdout_log": str(run_path / "stdout.log"),
        "stderr_log": str(run_path / "stderr.log"),
        "latent_embeddings": latent,
        "model_weights": weights,
        "sampled_volumes": sampled_volumes,
        "mrc_files": mrc_files,
        "pkl_files": pkl_files,
        "mrcs_files": mrcs_files,
        "representative_z_path": str(rep_z_path) if rep_z_path.is_file() else None,
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
        ev_result: Optional[Dict[str, Any]] = None
        if params.get("sample_volumes"):
            ev_result = run_eval_vol(params, run_path, dry_run=True)
        artifacts = collect_outputs(run_path)
        candidate_maps = _build_candidate_maps(artifacts, {})
        return {
            "backend_name": BACKEND_NAME,
            "selected_backend": BACKEND_NAME,
            "execution_mode": "dry_run",
            "status": "dry_run",
            "preflight": pf,
            "command_manifest": manifest["command_manifest"],
            "commands": manifest["commands"],
            "output_artifacts": artifacts,
            "sampled_volumes": list(artifacts.get("sampled_volumes") or []),
            "candidate_maps": candidate_maps,
            "eval_vol_result": ev_result,
            "success": True,
        }

    # Abinit commands only (eval_vol runs separately after z.pkl exists)
    abinit_commands = [c for c in manifest["commands"] if c.get("name") != "eval_vol"]
    stdout_log = Path(str(manifest["stdout_log"]))
    stderr_log = Path(str(manifest["stderr_log"]))
    exec_result = _run_commands(abinit_commands, run_path, stdout_log, stderr_log)
    status = str(exec_result.get("status"))

    eval_vol_result: Optional[Dict[str, Any]] = None
    if status == "completed" and params.get("sample_volumes"):
        eval_vol_result = run_eval_vol(params, run_path, dry_run=False)

    # Refresh artifacts AFTER eval_vol so sampled_volumes/*.mrc and
    # representative_z.txt land in the returned result.
    output_artifacts = collect_outputs(run_path)
    candidate_maps = _build_candidate_maps(output_artifacts, {})
    return {
        "backend_name": BACKEND_NAME,
        "selected_backend": BACKEND_NAME,
        "execution_mode": "real",
        "status": status,
        "preflight": pf,
        "command_manifest": manifest["command_manifest"],
        "commands": manifest["commands"],
        "output_artifacts": output_artifacts,
        "sampled_volumes": list(output_artifacts.get("sampled_volumes") or []),
        "candidate_maps": candidate_maps,
        "eval_vol_result": eval_vol_result,
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "return_code": exec_result.get("return_code"),
        "failed_reason": exec_result.get("failed_reason"),
        "success": status == "completed",
    }


def _build_candidate_maps(artifacts: Dict[str, Any], result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert sampled volume paths to structured candidate_maps dicts."""
    sampled_volumes = list(artifacts.get("sampled_volumes") or [])
    if not sampled_volumes:
        return []
    run_dir = artifacts.get("run_dir", "")
    rep_z_path = artifacts.get("representative_z_path") or ""
    weights_paths = [str(p) for p in (artifacts.get("model_weights") or [])]
    weights_path = next(
        (p for p in weights_paths if Path(str(p)).name == "weights.pkl"),
        weights_paths[0] if weights_paths else "",
    )
    return [
        {
            "source_backend": BACKEND_NAME,
            "source_run_dir": run_dir,
            "map_path": map_path,
            "latent_source": rep_z_path,
            "weights_path": weights_path,
            "qc_status": "pending",
            "recommended_next_action": "route_to_model_building",
        }
        for map_path in sampled_volumes
    ]


def update_state(state: Any, result: Dict[str, Any]) -> Any:
    """Write candidate maps and latent artifacts back to CryoAgent state.

    For dict state (used by the closed-loop tests and HTTP layer), this writes
    both top-level convenience keys and a structured ``last_execution`` dict so
    consumers can reach ``state["last_execution"]["sampled_volumes"]`` and
    ``state["last_execution"]["output_artifacts"]["sampled_volumes"]``.
    """

    artifacts = result.get("output_artifacts") if isinstance(result.get("output_artifacts"), dict) else {}
    sampled_volumes = list(artifacts.get("sampled_volumes") or result.get("sampled_volumes") or [])
    candidate_maps = list(result.get("candidate_maps") or _build_candidate_maps(artifacts, result))
    latent = list(artifacts.get("latent_embeddings") or [])
    recommended = "route_to_model_building" if candidate_maps else None

    # Idempotently enrich result so ``state["last_execution"]`` exposes the
    # closed-loop fields regardless of whether run() already populated them.
    result["candidate_maps"] = candidate_maps
    result["sampled_volumes"] = sampled_volumes
    if recommended:
        result.setdefault("recommended_downstream_action", recommended)
    if isinstance(artifacts, dict):
        artifacts.setdefault("sampled_volumes", sampled_volumes)
        artifacts.setdefault("mrc_files", list(artifacts.get("mrc_files") or sampled_volumes))
        result["output_artifacts"] = artifacts

    if isinstance(state, dict):
        state["cryodrgn_ai"] = result
        state["last_execution"] = result
        state["candidate_maps"] = candidate_maps
        state["latent_space_artifacts"] = latent
        if recommended:
            state["recommended_downstream_action"] = recommended
        return state

    state.last_execution = result
    if hasattr(state, "notes"):
        state.notes.append(
            f"cryodrgn_ai: status={result.get('status')} candidate_maps={len(candidate_maps)} latent_artifacts={len(latent)}"
        )
    return state
