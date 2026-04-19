"""Stage smoke project layout: which RELION stage dirs to strip per fixture name.

Used by ``scripts/make_stage_smoke_projects.sh`` (via ``python -m app.stage_smoke_layout``)
to build full-project copies for manual CLI smoke validation.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Tuple

# Order matches iteration in helper scripts and docs.
STAGE_SMOKE_FIXTURE_ORDER: Tuple[str, ...] = (
    "picking_ready",
    "extract_ready",
    "class2d_ready",
    "init3d_ready",
    "refine3d_ready",
    "postprocess_ready",
)

# Directories removed from each full copy (best-effort; missing dirs are skipped).
STAGE_SMOKE_REMOVALS: Dict[str, Tuple[str, ...]] = {
    "picking_ready": (
        "AutoPick",
        "Extract",
        "Class2D",
        "InitialModel",
        "Class3D",
        "Refine3D",
        "PostProcess",
        "CtfRefine",
        "Polish",
        "DynaMight",
        "ModelAngelo",
        "LocalRes",
        "MaskCreate",
    ),
    "extract_ready": (
        "Extract",
        "Class2D",
        "InitialModel",
        "Class3D",
        "Refine3D",
        "PostProcess",
        "CtfRefine",
        "Polish",
        "DynaMight",
        "ModelAngelo",
        "LocalRes",
        "MaskCreate",
    ),
    "class2d_ready": (
        "Class2D",
        "InitialModel",
        "Class3D",
        "Refine3D",
        "PostProcess",
        "CtfRefine",
        "Polish",
        "DynaMight",
        "ModelAngelo",
        "LocalRes",
        "MaskCreate",
    ),
    "init3d_ready": (
        "InitialModel",
        "Class3D",
        "Refine3D",
        "PostProcess",
        "CtfRefine",
        "Polish",
        "DynaMight",
        "ModelAngelo",
        "LocalRes",
        "MaskCreate",
    ),
    "refine3d_ready": (
        "Refine3D",
        "PostProcess",
        "CtfRefine",
        "Polish",
        "DynaMight",
        "ModelAngelo",
        "LocalRes",
        "MaskCreate",
    ),
    "postprocess_ready": (
        "PostProcess",
        "CtfRefine",
        "Polish",
        "DynaMight",
        "ModelAngelo",
        "LocalRes",
    ),
}


def _rm_tree_if_exists(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def create_smoke_projects(source: Path, dest_base: Path) -> None:
    """Copy ``source`` into ``dest_base/<fixture>/`` for each fixture, then strip dirs and agent state."""
    src = source.resolve()
    base = dest_base.resolve()
    if not src.is_dir():
        raise SystemExit(f"source is not a directory: {src}")
    base.mkdir(parents=True, exist_ok=True)

    for name in STAGE_SMOKE_FIXTURE_ORDER:
        if name not in STAGE_SMOKE_REMOVALS:
            raise RuntimeError(f"internal error: missing removals for {name}")
        target = base / name
        print(f"[stage_smoke] preparing {name} -> {target}")
        _rm_tree_if_exists(target)
        shutil.copytree(src, target, symlinks=True)
        agent = target / ".cryoem_agent"
        if agent.exists():
            _rm_tree_if_exists(agent)
            print(f"  removed {agent}")
        for dirname in STAGE_SMOKE_REMOVALS[name]:
            d = target / dirname
            if d.exists():
                _rm_tree_if_exists(d)
                print(f"  removed {d}")
        print(f"  done: {name}")


def _json_action_name(act: dict) -> str:
    nm = act.get("name")
    if isinstance(nm, str):
        return nm
    if isinstance(nm, dict) and "value" in nm:
        return str(nm["value"])
    return str(nm)


def summarize_smoke_outputs(out_dir: Path) -> None:
    """Print a small table from per-fixture ``scan.json`` files under ``out_dir``."""
    root = out_dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"output dir not found: {root}")

    rows: list[tuple[str, str, str]] = []
    for name in STAGE_SMOKE_FIXTURE_ORDER:
        scan_path = root / name / "scan.json"
        stage = "?"
        action = "?"
        if scan_path.is_file():
            try:
                data = json.loads(scan_path.read_text(encoding="utf-8"))
                st = data.get("state") or {}
                stage = str(st.get("stage", "?"))
            except (OSError, json.JSONDecodeError):
                stage = "(parse error)"
        run_path = root / name / "run_next.json"
        if run_path.is_file():
            try:
                data = json.loads(run_path.read_text(encoding="utf-8"))
                act = data.get("action") or {}
                action = _json_action_name(act if isinstance(act, dict) else {})
            except (OSError, json.JSONDecodeError):
                action = "(parse error)"
        rows.append((name, stage, action))

    w0 = max(len(r[0]) for r in rows)
    w1 = max(len(r[1]) for r in rows)
    print(f"{'fixture':<{w0}}  {'scan stage':<{w1}}  run_next action (dry-run)")
    print(f"{'-' * w0}  {'-' * w1}  {'-' * 24}")
    for name, stage, action in rows:
        print(f"{name:<{w0}}  {stage:<{w1}}  {action}")


def _cmd_create(args: argparse.Namespace) -> None:
    create_smoke_projects(args.source, args.dest_base)


def _cmd_summarize(args: argparse.Namespace) -> None:
    summarize_smoke_outputs(args.out_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stage smoke project helpers.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="Copy a RELION project into six smoke fixtures.")
    p_create.add_argument("--source", type=Path, required=True, help="Full RELION project path.")
    p_create.add_argument(
        "--dest-base",
        type=Path,
        required=True,
        help="Directory that will contain picking_ready/, extract_ready/, ...",
    )
    p_create.set_defaults(func=_cmd_create)

    p_sum = sub.add_parser("summarize", help="Summarize scan.json / run_next.json from a smoke run output dir.")
    p_sum.add_argument("--out-dir", type=Path, required=True, help="Directory passed to run_stage_smoke_checks.sh.")
    p_sum.set_defaults(func=_cmd_summarize)

    ns = parser.parse_args(argv)
    ns.func(ns)


if __name__ == "__main__":
    main()
