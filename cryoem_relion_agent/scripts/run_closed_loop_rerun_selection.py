"""Closed-loop LLM-supervised retrieval/rerun/selection CLI for CryoAgent.

Loads baseline evidence, retrieves fallback strategies, selects a rerun plan,
optionally executes available reruns (Topaz threshold sweep), compares baseline
vs reruns, and emits a final selection plus a closed-loop report.

The CLI is robust: missing inputs degrade to plan-only outputs.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow `python -m scripts.run_closed_loop_rerun_selection` and direct invocation.
_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from agent.closed_loop import (  # noqa: E402
    EvidenceCase, build_comparison, execute_topaz_threshold_rerun, load_evidence_case,
    retrieve_candidate_strategies, select_final, select_rerun_strategy,
    write_candidates, write_closed_loop_report, write_comparison, write_plan,
)


def _parse_topaz_pred_dir_threshold(values: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for v in values or []:
        if "=" not in v:
            continue
        thr, path = v.split("=", 1)
        out[thr.strip()] = path.strip()
    return out


def _bool_flag(s: str) -> bool:
    return str(s).strip().lower() in {"1", "true", "yes", "y", "on"}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="CryoAgent closed-loop rerun/selection controller.")
    p.add_argument("--baseline-root", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--dataset-id", default="EMPIAR-10017")
    p.add_argument("--pairs-csv", default=None)
    p.add_argument("--relion-bin", default=None)
    p.add_argument("--sequence", default=None)
    p.add_argument("--map2pdb-mode", default="dry-run")
    p.add_argument(
        "--topaz-pred-dir-threshold", action="append", default=[],
        help="Format: threshold=/abs/path/to/preds (repeatable).",
    )
    p.add_argument("--allow-real-topaz-rerun", action="store_true")
    p.add_argument("--run-llm", default="false")
    p.add_argument("--llm-model", default=None)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    baseline = load_evidence_case(args.baseline_root)
    (out_root / "baseline").mkdir(parents=True, exist_ok=True)
    (out_root / "baseline" / "baseline_summary.json").write_text(
        json.dumps(baseline.to_dict(), indent=2, default=str)
    )

    extra_topaz = _parse_topaz_pred_dir_threshold(args.topaz_pred_dir_threshold)
    candidates = retrieve_candidate_strategies(
        baseline, backend_availability=None, extra_topaz_pred_dirs=extra_topaz,
    )
    write_candidates(candidates, out_root / "retrieval")

    plan = select_rerun_strategy(baseline, candidates)
    write_plan(plan, out_root / "selection")

    rerun_cases: List[Dict[str, Any]] = []
    reruns_root = out_root / "reruns"
    reruns_root.mkdir(parents=True, exist_ok=True)

    can_workflow = bool(args.pairs_csv and args.relion_bin)
    run_llm_bool = _bool_flag(args.run_llm)

    if plan.get("selected_strategy_id") == "topaz_threshold_sweep" and extra_topaz:
        for thr, pred_dir in extra_topaz.items():
            safe_thr = thr.replace("/", "_")
            rerun_root = reruns_root / f"topaz_threshold_{safe_thr}"
            execute_topaz_threshold_rerun(
                threshold=thr, pred_dir=Path(pred_dir), rerun_root=rerun_root,
                pairs_csv=Path(args.pairs_csv) if args.pairs_csv else None,
                relion_bin=args.relion_bin, sequence=args.sequence,
                dataset_id=args.dataset_id, map2pdb_mode=args.map2pdb_mode,
                run_workflow=can_workflow, run_llm=run_llm_bool,
                llm_model=args.llm_model, limit=args.limit,
            )
            try:
                rerun_case = load_evidence_case(rerun_root, case_id=f"topaz_t{thr}")
            except Exception:
                continue
            rerun_cases.append({
                "case": rerun_case, "strategy_id": "topaz_threshold_sweep",
                "stage": "particle_picking", "backend": "topaz", "config": f"threshold={thr}",
            })

    rows = build_comparison(baseline, rerun_cases)
    final = select_final(baseline, rows, rerun_cases)
    write_comparison(rows, final, out_root / "comparison")
    write_closed_loop_report(out_root, baseline, candidates, plan, rows, final)

    print(f"[closed-loop] wrote outputs under: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
