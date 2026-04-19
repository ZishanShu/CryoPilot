"""Structured MICA runner for model-building backend."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List


def run_mica_pipeline(
    *,
    params: Dict[str, Any],
    preflight: Dict[str, Any],
    dry_run: bool,
) -> Dict[str, Any]:
    repo = Path(str(preflight.get("repo_path") or params.get("mica_repo_dir") or "."))
    output_dir = Path(str(params.get("output_dir")))
    python_path = str(preflight.get("python_path") or params.get("mica_python") or "python")
    protocol = str(params.get("protocol") or "AF3_struct")
    af3_dir = str(preflight.get("af3_dir") or params.get("af3_dir") or params.get("af3_result_dir"))
    cmd: List[str] = [
        python_path,
        str(preflight.get("run_py_path") or (repo / "run.py")),
        "--protocol",
        protocol,
        "-m",
        str(params["map_path"]),
        "-f",
        str(params["sequence_path"]),
        "-a",
        af3_dir,
        "-o",
        str(output_dir),
    ]
    if params.get("mica_device"):
        cmd.extend(["--device", str(params.get("mica_device"))])
    if bool(params.get("run_pulchra")):
        cmd.extend(["--run_pulchra", "-p", str(params.get("pulchra_path"))])
    if bool(params.get("run_phenix")):
        cmd.extend(["--run_phenix", "-x", str(params.get("phenix_act"))])
        if params.get("mica_resolution") is not None:
            cmd.extend(["-r", str(params.get("mica_resolution"))])

    if dry_run:
        return {
            "status": "dry_run",
            "execution_mode": "dry_run",
            "command_argv": cmd,
            "cwd": str(repo),
            "output_dir": str(output_dir),
            "repo_path": str(repo),
            "python_path": python_path,
            "af3_dir": af3_dir,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    before = {str(p) for p in output_dir.rglob("*") if p.exists()}
    proc = subprocess.run(
        cmd,
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    )
    after = {str(p) for p in output_dir.rglob("*") if p.exists()}
    created = sorted(after - before)
    has_artifacts = len(created) > 0 or any(output_dir.iterdir())

    if proc.returncode != 0:
        return {
            "status": "failed",
            "execution_mode": "real",
            "command_argv": cmd,
            "cwd": str(repo),
            "output_dir": str(output_dir),
            "repo_path": str(repo),
            "python_path": python_path,
            "af3_dir": af3_dir,
            "return_code": int(proc.returncode),
            "stdout": (proc.stdout or "")[-8000:],
            "stderr": (proc.stderr or "")[-8000:],
            "artifact_paths": created[:200],
            "error": "mica subprocess failed",
        }
    if not has_artifacts:
        return {
            "status": "failed",
            "execution_mode": "real",
            "command_argv": cmd,
            "cwd": str(repo),
            "output_dir": str(output_dir),
            "repo_path": str(repo),
            "python_path": python_path,
            "af3_dir": af3_dir,
            "return_code": int(proc.returncode),
            "stdout": (proc.stdout or "")[-8000:],
            "stderr": (proc.stderr or "")[-8000:],
            "artifact_paths": created[:200],
            "error": "missing_output_artifacts",
        }

    return {
        "status": "completed",
        "execution_mode": "real",
        "command_argv": cmd,
        "cwd": str(repo),
        "output_dir": str(output_dir),
        "repo_path": str(repo),
        "python_path": python_path,
        "af3_dir": af3_dir,
        "return_code": int(proc.returncode),
        "stdout": (proc.stdout or "")[-8000:],
        "stderr": (proc.stderr or "")[-8000:],
        "artifact_paths": created[:200],
    }
