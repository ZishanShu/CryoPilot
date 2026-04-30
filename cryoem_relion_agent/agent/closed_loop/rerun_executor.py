"""Execute reruns. First implementation: Topaz threshold sweep.

Two modes:
- evidence-only / existing-preds: count coordinates per micrograph, optionally
  invoke scripts.run_same_dataset_agent_workflow if pairs/relion paths provided.
- command mode (real Topaz rerun): conservative; only runs if --allow-real-topaz-rerun.

Always degrades to plan-only when required inputs are missing.
"""
from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _count_topaz_preds(pred_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not pred_dir.exists():
        return counts
    for f in pred_dir.iterdir():
        if not f.is_file():
            continue
        if f.suffix not in (".coord", ".txt", ".star"):
            continue
        try:
            n = 0
            for line in f.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # skip headers
                if any(c.isalpha() for c in line.split()[0]):
                    continue
                n += 1
            counts[f.stem] = n
        except Exception:
            counts[f.stem] = 0
    return counts


def execute_topaz_threshold_rerun(
    threshold: str,
    pred_dir: Path,
    rerun_root: Path,
    pairs_csv: Optional[Path] = None,
    relion_bin: Optional[str] = None,
    sequence: Optional[str] = None,
    dataset_id: str = "EMPIAR-10017",
    map2pdb_mode: str = "dry-run",
    run_workflow: bool = False,
    run_llm: bool = False,
    llm_model: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    rerun_root.mkdir(parents=True, exist_ok=True)
    (rerun_root / "logs").mkdir(parents=True, exist_ok=True)

    pred_counts = _count_topaz_preds(Path(pred_dir))
    total_coords = sum(pred_counts.values())

    summary: Dict[str, Any] = {
        "threshold": threshold,
        "pred_dir": str(pred_dir),
        "n_micrographs_with_preds": len(pred_counts),
        "total_topaz_coords_used": total_coords,
        "per_micrograph_coord_counts": pred_counts,
        "status": "evidence_only",
        "errors": [],
    }

    if run_workflow and pairs_csv and relion_bin:
        project_id = f"closed_loop_topaz_t{threshold}"
        cmd = [
            sys.executable, "-m", "scripts.run_same_dataset_agent_workflow",
            "--dataset-id", dataset_id,
            "--project-id", project_id,
            "--pairs-csv", str(pairs_csv),
            "--topaz-pred-dir", str(pred_dir),
            "--out-root", str(rerun_root),
            "--map2pdb-mode", map2pdb_mode,
            "--relion-bin", relion_bin,
        ]
        if sequence:
            cmd += ["--sequence", sequence]
        if limit is not None:
            cmd += ["--limit", str(limit)]
        if run_llm:
            cmd += ["--run-llm"]
            if llm_model:
                cmd += ["--llm-model", llm_model]
        log_path = rerun_root / "logs" / "rerun_workflow.log"
        try:
            with open(log_path, "w") as logf:
                proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, timeout=3600)
            summary["status"] = "workflow_executed" if proc.returncode == 0 else "workflow_failed"
            summary["return_code"] = proc.returncode
            summary["command"] = " ".join(shlex.quote(c) for c in cmd)
            ws_path = rerun_root / "workflow_summary.json"
            if ws_path.exists():
                summary["workflow_summary_path"] = str(ws_path)
        except Exception as e:
            summary["status"] = "workflow_error"
            summary["errors"].append(str(e))
    else:
        if not (pairs_csv and relion_bin):
            summary["status"] = "evidence_only_no_workflow"
            summary["note"] = (
                "pairs_csv or relion_bin not provided; produced evidence-only summary "
                "from existing Topaz prediction directory."
            )

    out_summary_path = rerun_root / "workflow_summary.json"
    if not out_summary_path.exists():
        # Write a minimal stand-in summary so comparator can read uniformly.
        stand_in = {
            "dataset_id": dataset_id,
            "project_id": f"closed_loop_topaz_t{threshold}",
            "metrics": {
                "n_micrographs": len(pred_counts),
                "total_topaz_coords_used": total_coords,
                "n_extracted_particles": 0,
                "n_particle_stacks": 0,
                "reconstruction_map_exists": False,
                "reconstruction_map_size_bytes": 0,
            },
            "stage_status": {
                "picking": "evidence_only",
                "extraction": "not_executed",
                "reconstruction": "not_executed",
                "map2pdb": "not_executed",
            },
            "closed_loop_rerun": summary,
        }
        out_summary_path.write_text(json.dumps(stand_in, indent=2))

    (rerun_root / "rerun_status.json").write_text(json.dumps(summary, indent=2))
    return summary
