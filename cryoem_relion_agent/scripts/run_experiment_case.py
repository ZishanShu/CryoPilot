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

  # Server-ready run with path variable overrides (paths in the case YAML can
  # reference ${PROJECT_ROOT}, ${DATA_ROOT}, ${RUN_ROOT}, ${CASE_ROOT})
  python -m scripts.run_experiment_case \\
      --project-root /srv/cryoagent \\
      --data-root /srv/data \\
      --run-root /srv/runs \\
      --case-root /srv/cases \\
      --protocol rule_only_agent \\
      --case F1_cryoppp_lite_topaz_input_format_failure
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.baseline_runner import run_experiment_case
from agent.benchmark_harness import BenchmarkHarness
from agent.case_registry import (
    CaseMaterializationError,
    materialize_case,
    resolve_case,
)
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
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Expands ${DATA_ROOT} in case configs.",
    )
    p.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Expands ${RUN_ROOT} in case configs.",
    )
    p.add_argument(
        "--case-root",
        type=Path,
        default=None,
        help="Expands ${CASE_ROOT} in case configs.",
    )
    p.add_argument(
        "--var",
        action="append",
        default=None,
        metavar="NAME=VALUE",
        help="Extra variable binding for case materialization (repeatable).",
    )
    p.add_argument(
        "--allow-env-vars",
        action="store_true",
        help="Also read PROJECT_ROOT/DATA_ROOT/RUN_ROOT/CASE_ROOT from env.",
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


def _build_variables(args: argparse.Namespace, project_root: Path) -> dict[str, str]:
    variables: dict[str, str] = {"PROJECT_ROOT": str(project_root)}
    if args.data_root is not None:
        variables["DATA_ROOT"] = str(Path(args.data_root).resolve())
    if args.run_root is not None:
        variables["RUN_ROOT"] = str(Path(args.run_root).resolve())
    if args.case_root is not None:
        variables["CASE_ROOT"] = str(Path(args.case_root).resolve())
    for pair in args.var or []:
        if "=" not in pair:
            raise ValueError(f"--var expects NAME=VALUE, got {pair!r}")
        name, value = pair.split("=", 1)
        variables[name.strip()] = value.strip()
    return variables


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(args.project_root).resolve()

    protocol_search_dirs = [Path(d) for d in (args.protocol_dir or [])] + _default_protocol_dirs(project_root)
    case_search_dirs = [Path(d) for d in (args.case_dir or [])] + _default_case_dirs(project_root)

    protocol = resolve_protocol(args.protocol, search_dirs=protocol_search_dirs)
    raw_case = resolve_case(args.case, search_dirs=case_search_dirs)
    variables = _build_variables(args, project_root)
    try:
        case = materialize_case(
            raw_case,
            variables=variables,
            include_env=bool(args.allow_env_vars),
        )
    except CaseMaterializationError as exc:
        raise SystemExit(f"case materialization failed: {exc}")

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
