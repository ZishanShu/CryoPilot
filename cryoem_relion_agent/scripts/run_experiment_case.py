#!/usr/bin/env python3
"""CLI: run one (protocol, case) through the existing benchmark harness.

Usage examples:

  # fixed_workflow baseline via built-in defaults, dry-run (no real exec)
  python -m scripts.run_experiment_case \\
      --project-root . \\
      --protocol fixed_workflow_topaz \\
      --case cryoppp_lite_topaz_smoke \\
      --dry-run

  # rule_only_agent baseline using a case file from configs/cases
  python -m scripts.run_experiment_case \\
      --project-root . \\
      --protocol rule_only_agent \\
      --case configs/cases/cryoppp_lite_topaz_smoke.yaml \\
      --output-dir real_runs/baselines/rule_only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.baseline_runner import run_experiment_case
from agent.benchmark_harness import BenchmarkHarness
from agent.case_registry import resolve_case
from agent.experiment_protocol import resolve_protocol
from app.scheduler import Scheduler


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", required=True, type=Path)
    p.add_argument("--protocol", required=True, help="protocol name or YAML/JSON path")
    p.add_argument("--case", required=True, help="case id or YAML/JSON path")
    p.add_argument(
        "--protocol-dir",
        action="append",
        default=None,
        help="Additional directory to search for protocol configs (repeatable).",
    )
    p.add_argument(
        "--case-dir",
        action="append",
        default=None,
        help="Additional directory to search for case configs (repeatable).",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--no-persist",
        action="store_true",
        help="Skip writing to .cryoem_agent/benchmarks/{latest,history}.",
    )
    p.add_argument(
        "--no-audit",
        action="store_true",
        help="Skip writing success_audit / failure_audit.",
    )
    return p


def _default_protocol_dirs(project_root: Path) -> list[Path]:
    return [
        project_root / "configs" / "experiment_protocols",
        Path("configs/experiment_protocols"),
    ]


def _default_case_dirs(project_root: Path) -> list[Path]:
    return [
        project_root / "configs" / "cases",
        Path("configs/cases"),
    ]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(args.project_root).resolve()

    protocol_search_dirs = [Path(d) for d in (args.protocol_dir or [])] + _default_protocol_dirs(project_root)
    case_search_dirs = [Path(d) for d in (args.case_dir or [])] + _default_case_dirs(project_root)

    protocol = resolve_protocol(args.protocol, search_dirs=protocol_search_dirs)
    case = resolve_case(args.case, search_dirs=case_search_dirs)

    scheduler = Scheduler(project_root=project_root)
    harness = BenchmarkHarness(executor=scheduler.action_executor, project_root=project_root)

    result = run_experiment_case(
        protocol=protocol,
        case=case,
        harness=harness,
        project_root=project_root,
        output_dir=args.output_dir,
        dry_run=bool(args.dry_run),
        persist=not bool(args.no_persist),
        write_audit=not bool(args.no_audit),
    )

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    status = str(result.status or "").lower()
    if status in {"completed", "success"}:
        return 0
    if status == "preflight_blocked":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
