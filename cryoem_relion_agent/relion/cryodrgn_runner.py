"""Structured cryoDRGN CLI execution (heterogeneity backend)."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _run_step(
    exe: str,
    args: List[str],
    cwd: Optional[Path],
) -> Dict[str, Any]:
    cmd = [exe] + args
    logger.info("cryoDRGN: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "")[-8000:],
        "stderr": (proc.stderr or "")[-8000:],
    }


def _has_weights(train_out: Path) -> bool:
    if not train_out.is_dir():
        return False
    if any(train_out.glob("weights*.pkl")) or any(train_out.glob("epoch*.pkl")):
        return True
    return (train_out / "config.yaml").is_file() and any(train_out.glob("*.pkl"))


def run_cryodrgn_pipeline(
    params: Dict[str, Any],
    *,
    preflight: Dict[str, Any],
    project_root: Path,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run parse → optional downsample → train_vae → optional analyze."""
    exe = str(preflight.get("executable_path") or "")
    out_base = Path(str(params["output_dir"]))
    steps: List[Dict[str, Any]] = []
    commands: List[str] = []

    star_p = params.get("particles_star_path")
    pose_path = Path(params["pose_pkl_path"]) if params.get("pose_pkl_path") else out_base / "pose.pkl"
    ctf_path = Path(params["ctf_pkl_path"]) if params.get("ctf_pkl_path") else out_base / "ctf.pkl"

    mrcs_in = params.get("particles_mrcs_path")
    zdim = int(params.get("cryodrgn_zdim") or 10)
    epochs = int(params.get("cryodrgn_epochs") or 10)
    down_d = params.get("downsample_box_size")
    apix = params.get("apix")
    box = params.get("box_size")
    analyze_after = bool(params.get("cryodrgn_analyze_after_train"))
    train_sub = params.get("cryodrgn_train_subdir") or "train_vae"
    train_out = out_base / str(train_sub)

    external = bool(
        mrcs_in
        and params.get("pose_pkl_path")
        and params.get("ctf_pkl_path")
        and Path(str(mrcs_in)).is_file()
        and Path(str(params["pose_pkl_path"])).is_file()
        and Path(str(params["ctf_pkl_path"])).is_file()
    )

    train_input: str
    if dry_run:
        steps.append({"name": "dry_run", "status": "dry_run"})
        return {
            "status": "dry_run",
            "heterogeneity_backend": "cryodrgn",
            "execution_mode": "dry_run",
            "steps": steps,
            "commands": commands,
            "output_dir": str(out_base),
            "train_outdir": str(train_out),
        }

    if external:
        train_input = str(Path(mrcs_in))
        pose_path = Path(str(params["pose_pkl_path"]))
        ctf_path = Path(str(params["ctf_pkl_path"]))
    else:
        if not star_p or not Path(str(star_p)).is_file():
            return {
                "status": "failed",
                "heterogeneity_backend": "cryodrgn",
                "execution_mode": "real",
                "error": "particles_star_path missing",
                "steps": steps,
                "commands": commands,
            }
        star_path = Path(str(star_p))
        if not pose_path.is_file():
            r = _run_step(exe, ["parse_pose_star", str(star_path), "-o", str(pose_path)], cwd=out_base.parent)
            steps.append({"name": "parse_pose_star", **r})
            commands.append(r["command"])
            if r["returncode"] != 0 or not pose_path.is_file():
                return {
                    "status": "failed",
                    "heterogeneity_backend": "cryodrgn",
                    "execution_mode": "real",
                    "error": "parse_pose_star failed",
                    "steps": steps,
                    "commands": commands,
                }
        ctf_args = ["parse_ctf_star", str(star_path), "-o", str(ctf_path)]
        if box is not None:
            ctf_args.extend(["-D", str(int(box))])
        if apix is not None:
            ctf_args.extend(["-Apix", str(float(apix))])
        if not ctf_path.is_file():
            r2 = _run_step(exe, ctf_args, cwd=out_base.parent)
            steps.append({"name": "parse_ctf_star", **r2})
            commands.append(r2["command"])
            if r2["returncode"] != 0 or not ctf_path.is_file():
                return {
                    "status": "failed",
                    "heterogeneity_backend": "cryodrgn",
                    "execution_mode": "real",
                    "error": "parse_ctf_star failed",
                    "steps": steps,
                    "commands": commands,
                }
        train_input = str(star_path)
        if down_d is not None:
            ds_out = out_base / f"particles_ds_{int(down_d)}.mrcs"
            ds_args = ["downsample", str(star_path), "-D", str(int(down_d)), "-o", str(ds_out)]
            rd = _run_step(exe, ds_args, cwd=out_base.parent)
            steps.append({"name": "downsample", **rd})
            commands.append(rd["command"])
            if rd["returncode"] != 0 or not ds_out.is_file():
                return {
                    "status": "failed",
                    "heterogeneity_backend": "cryodrgn",
                    "execution_mode": "real",
                    "error": "downsample failed",
                    "steps": steps,
                    "commands": commands,
                }
            train_input = str(ds_out)

    train_out.mkdir(parents=True, exist_ok=True)
    tv_args: List[str] = [
        "train_vae",
        train_input,
        "--poses",
        str(pose_path),
        "--ctf",
        str(ctf_path),
        "--zdim",
        str(zdim),
        "-n",
        str(epochs),
        "-o",
        str(train_out),
    ]
    rt = _run_step(exe, tv_args, cwd=out_base.parent)
    steps.append({"name": "train_vae", **rt})
    commands.append(rt["command"])
    if rt["returncode"] != 0 or not _has_weights(train_out):
        return {
            "status": "failed",
            "heterogeneity_backend": "cryodrgn",
            "execution_mode": "real",
            "error": "train_vae failed or missing weights",
            "steps": steps,
            "commands": commands,
            "output_dir": str(out_base),
            "train_outdir": str(train_out),
        }

    if analyze_after and apix is not None:
        ep = epochs - 1 if epochs > 0 else 0
        ra = _run_step(exe, ["analyze", str(train_out), str(ep), "--Apix", str(float(apix))], cwd=out_base.parent)
        steps.append({"name": "analyze", **ra})
        commands.append(ra["command"])
        if ra["returncode"] != 0:
            return {
                "status": "failed",
                "heterogeneity_backend": "cryodrgn",
                "execution_mode": "real",
                "error": "analyze failed",
                "steps": steps,
                "commands": commands,
                "output_dir": str(out_base),
                "train_outdir": str(train_out),
            }

    return {
        "status": "completed",
        "heterogeneity_backend": "cryodrgn",
        "execution_mode": "real",
        "steps": steps,
        "commands": commands,
        "output_dir": str(out_base),
        "train_outdir": str(train_out),
        "pose_pkl": str(pose_path),
        "ctf_pkl": str(ctf_path),
    }
