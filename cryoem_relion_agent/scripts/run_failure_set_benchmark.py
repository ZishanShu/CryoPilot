#!/usr/bin/env python3
"""CLI: run the E2 failure-set benchmark over a list of cases.

Usage:

  # Default: all 4 F1..F4 failure cases with all three advice sources.
  python -m scripts.run_failure_set_benchmark \\
      --project-root . \\
      --cases F1_cryoppp_lite_topaz_input_format_failure \\
              F2_topaz_preflight_blocked_missing_executable \\
              F3_topaz_empty_output_case \\
              F4_mica_unavailable_missing_af3_dir

  # With explicit case registry dir override and only rule_only+llm sources.
  python -m scripts.run_failure_set_benchmark \\
      --project-root . \\
      --case-dir configs/cases \\
      --advice-sources rule_only llm \\
      --data-root /srv/data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from agent.case_registry import (
    CaseMaterializationError,
    ExperimentCase,
    materialize_case,
    resolve_case,
)
from agent.failure_benchmark import (
    ADVICE_SOURCES,
    persist_failure_benchmark,
    render_failure_benchmark_markdown,
    run_failure_benchmark,
)

_DEFAULT_FAILURE_CASE_IDS = (
    "F1_cryoppp_lite_topaz_input_format_failure",
    "F2_topaz_preflight_blocked_missing_executable",
    "F3_topaz_empty_output_case",
    "F4_mica_unavailable_missing_af3_dir",
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", required=True, type=Path)
    p.add_argument(
        "--cases",
        nargs="*",
        default=list(_DEFAULT_FAILURE_CASE_IDS),
        help="Case ids or YAML/JSON paths (defaults to F1..F4).",
    )
    p.add_argument(
        "--case-dir",
        action="append",
        default=None,
        help="Additional directory to search for case configs.",
    )
    p.add_argument(
        "--advice-sources",
        nargs="*",
        default=list(ADVICE_SOURCES),
        choices=list(ADVICE_SOURCES),
        help="Subset of advice sources to run (default: all).",
    )
    p.add_argument("--benchmark-name", default="e2_failure_set")
    p.add_argument("--output-subdir", default="e2_failure_set")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--run-root", type=Path, default=None)
    p.add_argument("--case-root", type=Path, default=None)
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
    p.add_argument(
        "--no-persist",
        action="store_true",
        help="Skip writing latest.json / latest.md under .cryoem_agent/experiments.",
    )
    return p


def _default_case_dirs(project_root: Path) -> List[Path]:
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


def _materialize_many(
    cases: Iterable[str],
    *,
    project_root: Path,
    search_dirs: List[Path],
    variables: dict[str, str],
    include_env: bool,
) -> List[ExperimentCase]:
    out: List[ExperimentCase] = []
    for name in cases:
        raw = resolve_case(name, search_dirs=search_dirs)
        try:
            out.append(materialize_case(raw, variables=variables, include_env=include_env))
        except CaseMaterializationError as exc:
            raise SystemExit(f"case materialization failed for {name!r}: {exc}")
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(args.project_root).resolve()

    search_dirs = [Path(d) for d in (args.case_dir or [])] + _default_case_dirs(project_root)
    variables = _build_variables(args, project_root)

    cases = _materialize_many(
        args.cases,
        project_root=project_root,
        search_dirs=search_dirs,
        variables=variables,
        include_env=bool(args.allow_env_vars),
    )

    payload = run_failure_benchmark(
        cases,
        benchmark_name=args.benchmark_name,
        advice_sources=tuple(args.advice_sources),
    )

    if not args.no_persist:
        persist_info = persist_failure_benchmark(
            project_root=project_root,
            payload=payload,
            subdir=args.output_subdir,
        )
        payload = dict(payload)
        payload["persistence"] = persist_info

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("")
    print("=== markdown preview ===")
    print(render_failure_benchmark_markdown(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
