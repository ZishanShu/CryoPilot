"""RELION particles STAR to cryoDRGN-AI preprocessing adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class CryodrgnPreprocessPlan:
    """Command plan for converting RELION particle inputs to cryoDRGN inputs."""

    particles_star: Path
    datadir: Path
    apix: float
    box_size: int
    downsample_box: int
    run_dir: Path
    particles_mrcs: Path
    ctf_pkl: Path
    commands: List[List[str]]
    manifest_path: Path

    def to_manifest(self) -> Dict[str, Any]:
        return {
            "input": {
                "particles_star": str(self.particles_star),
                "datadir": str(self.datadir),
                "apix": self.apix,
                "box_size": self.box_size,
                "downsample_box": self.downsample_box,
            },
            "output": {
                "particles_mrcs": str(self.particles_mrcs),
                "ctf_pkl": str(self.ctf_pkl),
            },
            "commands": self.commands,
        }


def _require_path(value: Any, field_name: str, *, must_exist: bool = True) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"{field_name} is required")
    path = Path(str(value)).expanduser()
    if must_exist and not path.exists():
        raise ValueError(f"{field_name} does not exist: {path}")
    return path


def build_relion_to_cryodrgn_preprocess_plan(
    *,
    particles_star: str | Path,
    datadir: str | Path,
    apix: float,
    box_size: int,
    run_dir: str | Path,
    downsample_box: Optional[int] = None,
    cryodrgn_executable: str = "cryodrgn",
) -> CryodrgnPreprocessPlan:
    """Build downsample and CTF parsing commands for RELION particle STAR input."""

    star_path = _require_path(particles_star, "particles_star")
    data_dir = _require_path(datadir, "datadir")
    run_path = Path(run_dir).expanduser()
    dbox = int(downsample_box or box_size)
    if dbox <= 0:
        raise ValueError("downsample_box must be positive")
    ibox = int(box_size)
    if ibox <= 0:
        raise ValueError("box_size must be positive")
    pix = float(apix)
    if pix <= 0:
        raise ValueError("apix must be positive")

    preprocess_dir = run_path / "preprocess"
    particles_out = preprocess_dir / f"particles_{dbox}.mrcs"
    ctf_out = preprocess_dir / "ctf.pkl"
    commands = [
        [
            cryodrgn_executable,
            "downsample",
            str(star_path),
            "-D",
            str(dbox),
            "-o",
            str(particles_out),
            "--datadir",
            str(data_dir),
        ],
        [
            cryodrgn_executable,
            "parse_ctf_star",
            str(star_path),
            "-D",
            str(ibox),
            "--Apix",
            str(pix),
            "-o",
            str(ctf_out),
        ],
    ]
    return CryodrgnPreprocessPlan(
        particles_star=star_path,
        datadir=data_dir,
        apix=pix,
        box_size=ibox,
        downsample_box=dbox,
        run_dir=run_path,
        particles_mrcs=particles_out,
        ctf_pkl=ctf_out,
        commands=commands,
        manifest_path=preprocess_dir / "preprocess_manifest.json",
    )


def write_preprocess_manifest(plan: CryodrgnPreprocessPlan) -> Path:
    """Write ``preprocess_manifest.json`` for reproducible cryoDRGN-AI input prep."""

    plan.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    plan.manifest_path.write_text(
        json.dumps(plan.to_manifest(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return plan.manifest_path

