"""Structured EModelX runner for model-building backend."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List


def _resolve_resolution(params: Dict[str, Any], preflight: Dict[str, Any]) -> Any:
    value = params.get("resolution")
    if value is not None:
        return value
    return preflight.get("resolution")


_EMODELX_ARTIFACT_GLOBS: tuple = (
    "*.pdb",
    "*.cif",
    "time_cost*.csv",
    "time_cost*.txt",
)


def _harvest_emodelx_artifacts(
    output_dir: Path,
    created: List[str],
) -> List[str]:
    """Union the before/after diff with known EModelX output file patterns.

    The diff-based ``created`` list is empty when the output_dir was
    pre-populated (re-runs) or when EModelX overwrites files in place.  Rather
    than rely solely on the diff, also scan the output_dir for known seq_free
    / temp_free / temp_flex product patterns — specifically the
    ``_EModelX(noseq)_ca_model.pdb`` and ``time_cost__EModelX(noseq).csv``
    that the seq_free protocol emits.  This is additive; no pattern is
    required for the run to be considered successful.
    """
    found = {str(p) for p in (created or [])}
    if output_dir.is_dir():
        for pattern in _EMODELX_ARTIFACT_GLOBS:
            for p in output_dir.rglob(pattern):
                if p.is_file():
                    found.add(str(p))
    return sorted(found)


def run_emodelx_pipeline(
    *,
    params: Dict[str, Any],
    preflight: Dict[str, Any],
    dry_run: bool,
) -> Dict[str, Any]:
    repo = Path(str(preflight.get("repo_path") or params.get("emodelx_repo_dir") or "."))
    output_dir = Path(str(params.get("output_dir")))
    python_path = str(preflight.get("python_path") or params.get("emodelx_python") or "python")
    protocol = str(params.get("emodelx_protocol") or preflight.get("protocol") or "temp_free")
    cmd: List[str] = [
        python_path,
        str(preflight.get("run_py_path") or (repo / "run.py")),
        "--protocol",
        protocol,
        "--EM_map",
        str(params["map_path"]),
    ]
    # Protocol-aware inputs: seq_free omits --fasta entirely; temp_flex also
    # forwards --template_dir when the case supplies one.
    if protocol != "seq_free":
        cmd.extend(["--fasta", str(params["sequence_path"])])
    if protocol == "temp_flex" and params.get("template_dir"):
        cmd.extend(["--template_dir", str(params.get("template_dir"))])
    cmd.extend(["--output_dir", str(output_dir)])

    if bool(params.get("run_pulchra")):
        cmd.extend(["--run_pulchra", "--pulchra_path", str(params.get("pulchra_path"))])
    if bool(params.get("run_phenix")):
        cmd.extend(["--run_phenix", "--phenix_act", str(params.get("phenix_act"))])
        resolution = _resolve_resolution(params, preflight)
        if resolution is not None:
            cmd.extend(["--resolution", str(resolution)])

    # Ensure output_dir exists before subprocess launch, even when the
    # benchmark harness or experiment framework has not pre-created it.
    output_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        return {
            "status": "dry_run",
            "execution_mode": "dry_run",
            "command_argv": cmd,
            "command": " ".join(str(x) for x in cmd),
            "cwd": str(repo),
            "output_dir": str(output_dir),
            "repo_path": str(repo),
            "python_path": python_path,
            "artifact_paths": [],
        }

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
    harvested = _harvest_emodelx_artifacts(output_dir, created)
    has_artifacts = len(harvested) > 0 or any(output_dir.iterdir())

    if proc.returncode != 0:
        return {
            "status": "failed",
            "execution_mode": "real",
            "command_argv": cmd,
            "command": " ".join(str(x) for x in cmd),
            "cwd": str(repo),
            "output_dir": str(output_dir),
            "repo_path": str(repo),
            "python_path": python_path,
            "return_code": int(proc.returncode),
            "stdout": (proc.stdout or "")[-8000:],
            "stderr": (proc.stderr or "")[-8000:],
            "artifact_paths": harvested[:200],
            "generated_paths": harvested[:200],
            "error": "emodelx subprocess failed",
        }

    if not has_artifacts:
        return {
            "status": "failed",
            "execution_mode": "real",
            "command_argv": cmd,
            "command": " ".join(str(x) for x in cmd),
            "cwd": str(repo),
            "output_dir": str(output_dir),
            "repo_path": str(repo),
            "python_path": python_path,
            "return_code": int(proc.returncode),
            "stdout": (proc.stdout or "")[-8000:],
            "stderr": (proc.stderr or "")[-8000:],
            "artifact_paths": harvested[:200],
            "generated_paths": harvested[:200],
            "error": "missing_output_artifacts",
        }

    return {
        "status": "completed",
        "execution_mode": "real",
        "command_argv": cmd,
        "command": " ".join(str(x) for x in cmd),
        "cwd": str(repo),
        "output_dir": str(output_dir),
        "repo_path": str(repo),
        "python_path": python_path,
        "return_code": int(proc.returncode),
        "stdout": (proc.stdout or "")[-8000:],
        "stderr": (proc.stderr or "")[-8000:],
        "artifact_paths": harvested[:200],
        "generated_paths": harvested[:200],
    }
