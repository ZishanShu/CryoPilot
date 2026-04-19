"""Small RELION-like fixture project builders for smoke validation tests."""

from __future__ import annotations

from pathlib import Path


def _write(path: Path, content: str = "data_\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_fixture_project(root: Path, fixture_name: str) -> Path:
    """
    Build a tiny RELION-like project tree under ``root / fixture_name``.

    Fixtures are intentionally minimal and only include markers required by
    watcher/QC/policy smoke coverage.
    """
    p = root / fixture_name
    p.mkdir(parents=True, exist_ok=True)

    # Common upstream baseline for most fixtures.
    if fixture_name in {
        "picking_ready",
        "extract_ready",
        "class2d_ready",
        "init3d_ready",
        "refine3d_ready",
        "postprocess_ready",
    }:
        _write(p / "MotionCorr" / "job001" / "corrected_micrographs.star", "data_micrographs\nloop_\n_rlnMicrographName #1\nm1.mrc\n")
        _write(p / "CtfFind" / "job001" / "micrographs_ctf.star", "data_\nloop_\n_rlnMicrographName #1\nm1.mrc\n")

    if fixture_name in {"extract_ready", "class2d_ready", "init3d_ready", "refine3d_ready", "postprocess_ready"}:
        _write(p / "AutoPick" / "job001" / "autopick.star")

    if fixture_name in {"class2d_ready", "init3d_ready", "refine3d_ready", "postprocess_ready"}:
        _write(p / "Extract" / "job001" / "particles.star", "data_\nloop_\n_rlnImageName #1\n1@a.mrcs\n")

    if fixture_name in {"init3d_ready", "refine3d_ready", "postprocess_ready"}:
        _write(p / "Class2D" / "job001" / "classes.star", "data_\nloop_\n_rlnImageName #1\n1@a.mrcs\n")

    if fixture_name in {"refine3d_ready", "postprocess_ready"}:
        _write(p / "InitialModel" / "job001" / "model.star", "data_\nloop_\n_rlnClassNumber #1\n1\n")
        _write(p / "InitialModel" / "job001" / "init_model.mrc", "fake")

    if fixture_name in {"postprocess_ready"}:
        _write(p / "Refine3D" / "job001" / "refined_map.mrc", "fake")
        _write(p / "PostProcess" / "job001" / "postprocess_masked.mrc", "fake")
        _write(
            p / "PostProcess" / "job001" / "postprocess.star",
            "data_\n_rlnFinalResolution 3.60\n_rlnPostprocessedMapMasked PostProcess/job001/postprocess_masked.mrc\n",
        )

    return p

