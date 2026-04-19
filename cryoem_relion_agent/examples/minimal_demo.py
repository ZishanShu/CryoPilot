"""Minimal local end-to-end demo for cryoem_relion_agent.

This script creates a fake RELION-like project tree, then runs:
1) scan-project
2) decide-next
3) generate-report

Usage:
    python examples/minimal_demo.py
    python examples/minimal_demo.py --workspace /tmp/cryo_demo
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for demo script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def create_fake_relion_project(project_dir: Path) -> None:
    """Create a conservative fake RELION project for local demonstration."""
    # Clean old project if exists for deterministic behavior.
    if project_dir.exists():
        shutil.rmtree(project_dir)

    stage_dirs = [
        "Import/job001",
        "MotionCorr/job002",
        "CtfFind/job003",
        "AutoPick/job004",
        "Extract/job005",
        "Class2D/job006",
        "InitialModel/job007",
        "Refine3D/job008",
    ]
    for stage in stage_dirs:
        (project_dir / stage).mkdir(parents=True, exist_ok=True)

    # Minimal files for parser/watcher heuristics.
    (project_dir / "MotionCorr/job002/micrograph_0001.mrc").write_text(
        "fake mrc marker\n",
        encoding="utf-8",
    )
    (project_dir / "MotionCorr/job002/micrograph_0002.mrc").write_text(
        "fake mrc marker\n",
        encoding="utf-8",
    )

    # STAR file with simple row-like content for counting particles.
    (project_dir / "Extract/job005/particles.star").write_text(
        "\n".join(
            [
                "data_particles",
                "",
                "loop_",
                "_rlnImageName #1",
                "000001@stack.mrcs",
                "000002@stack.mrcs",
                "000003@stack.mrcs",
                "000004@stack.mrcs",
                "000005@stack.mrcs",
            ]
        ),
        encoding="utf-8",
    )

    (project_dir / "Class2D/job006/class2d_model.star").write_text(
        "\n".join(
            [
                "data_model_general",
                "_rlnEstimatedResolution 6.2",
                "completed",
            ]
        ),
        encoding="utf-8",
    )

    (project_dir / "Refine3D/job008/refine3d.log").write_text(
        "\n".join(
            [
                "Job finished successfully",
                "Final resolution = 3.8",
                "particles 120000",
                "map written to refined_map.mrc",
            ]
        ),
        encoding="utf-8",
    )

    (project_dir / "Refine3D/job008/refined_map.mrc").write_text(
        "fake map content\n",
        encoding="utf-8",
    )


def run_command(args: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run one command and raise clear error when failed."""
    logging.info("Running command: %s", " ".join(args))
    completed = subprocess.run(
        args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(args)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def main() -> int:
    """Create fake project and run full CLI chain."""
    parser = argparse.ArgumentParser(description="Minimal local demo for cryoem_relion_agent.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("demo_workspace"),
        help="Workspace directory for fake project and outputs.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs.")
    args = parser.parse_args()

    setup_logging(args.verbose)

    repo_root = Path(__file__).resolve().parents[1]
    workspace = args.workspace if args.workspace.is_absolute() else (repo_root / args.workspace)
    project_dir = workspace / "fake_relion_project"
    output_dir = workspace / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        create_fake_relion_project(project_dir)

        scan_result = run_command(
            [sys.executable, "-m", "app.main", "scan-project", "--project", str(project_dir)],
            cwd=repo_root,
        )
        decide_result = run_command(
            [sys.executable, "-m", "app.main", "decide-next", "--project", str(project_dir)],
            cwd=repo_root,
        )
        report_md = output_dir / "demo_report.md"
        report_json = output_dir / "demo_report.json"
        run_command(
            [
                sys.executable,
                "-m",
                "app.main",
                "generate-report",
                "--project",
                str(project_dir),
                "--out",
                str(report_md),
                "--out-json",
                str(report_json),
            ],
            cwd=repo_root,
        )

        scan_json_path = output_dir / "scan_result.json"
        decide_json_path = output_dir / "decide_result.json"
        scan_json_path.write_text(scan_result.stdout, encoding="utf-8")
        decide_json_path.write_text(decide_result.stdout, encoding="utf-8")

        # Validate outputs are parseable JSON where expected.
        json.loads(scan_result.stdout)
        json.loads(decide_result.stdout)

        logging.info("Demo finished successfully.")
        print(f"Fake project: {project_dir}")
        print(f"Scan output:  {scan_json_path}")
        print(f"Decision:     {decide_json_path}")
        print(f"Report MD:    {report_md}")
        print(f"Report JSON:  {report_json}")
        return 0
    except Exception as exc:  # noqa: BLE001
        logging.error("Demo failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
