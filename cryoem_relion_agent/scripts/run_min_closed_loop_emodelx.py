#!/usr/bin/env python3
"""Run a minimal real closed loop demo routed to EModelX."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.model_building_preflight import preflight_model_building
from agent.structure_router import ModelingContext
from app.scheduler import Scheduler


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _list_output_files(root: Path) -> List[str]:
    if not root.exists():
        return []
    files: List[str] = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            files.append(str(p))
    return files


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", required=True, help="RELION project path.")
    p.add_argument("--sequence", required=True, help="Sequence FASTA path.")
    p.add_argument("--run-root", default="real_runs/min_closed_loop_emodelx", help="Run root directory.")
    p.add_argument("--chain-count", type=int, default=2, help="Chain count hint for routing.")
    p.add_argument("--emodelx-repo", default=None, help="Optional EModelX repo path override.")
    p.add_argument("--emodelx-python", default=None, help="Optional EModelX python path override.")
    p.add_argument("--protocol", default="temp_free", choices=["temp_free", "temp_flex"], help="EModelX protocol.")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    project = Path(args.project).resolve()
    sequence = Path(args.sequence).resolve()

    if not project.is_dir():
        raise SystemExit(f"project not found: {project}")
    if not sequence.is_file():
        raise SystemExit(f"sequence not found: {sequence}")

    run_dir = (Path(args.run_root).resolve() / _now_stamp())
    run_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = run_dir / "audit"
    output_dir = run_dir / "outputs" / "emodelx"
    output_dir.mkdir(parents=True, exist_ok=True)

    exact_command = " ".join(shlex.quote(x) for x in sys.argv)
    _write_text(audit_dir / "exact_command.txt", exact_command + "\n")

    scheduler = Scheduler(project_root=project)
    state, qc = scheduler.scan_project(sequence_override=sequence)
    context = ModelingContext(
        known_sequence=True,
        chain_count=max(1, int(args.chain_count)),
        af3_available=False,
        preferred_model_building_backend="emodelx",
    )
    action = scheduler.decide_next(sequence_override=sequence, modeling_context=context)

    if action.name.value != "run_emodelx":
        raise SystemExit(
            f"router did not select EModelX, got: {action.name.value}, reason: {action.reason}"
        )

    params = dict(action.parameters or {})
    params["output_dir"] = str(output_dir)
    params["emodelx_protocol"] = str(args.protocol)
    if args.emodelx_repo:
        params["emodelx_repo_dir"] = str(Path(args.emodelx_repo).resolve())
    if args.emodelx_python:
        params["emodelx_python"] = str(Path(args.emodelx_python).resolve())

    pf = preflight_model_building(
        "emodelx",
        params,
        emodelx_config_path=REPO_ROOT / "configs" / "emodelx.yaml",
    )
    params["model_building_preflight"] = pf
    params["emodelx_preflight"] = pf
    params["model_building_backend"] = "emodelx"
    params["model_building_availability"] = "available" if pf.get("available") else "unavailable"
    params["model_building_execution_mode"] = str(pf.get("execution_mode") or "unavailable")
    params["model_building_blocked_reason"] = pf.get("blocked_reason") or ""
    params["model_building_missing_inputs"] = list(pf.get("missing_inputs") or [])
    params["model_building_missing_dependencies"] = list(pf.get("missing_dependencies") or [])
    action = action.model_copy(update={"parameters": params})

    result = scheduler.action_executor.execute(
        action=action,
        project_root=project,
        dry_run=False,
        state=state,
    )

    stdout_text = str(result.get("stdout") or "")
    stderr_text = str(result.get("stderr") or "")
    _write_text(audit_dir / "stdout.log", stdout_text)
    _write_text(audit_dir / "stderr.log", stderr_text)

    output_files = _list_output_files(output_dir)
    _write_text(audit_dir / "output_files.txt", "\n".join(output_files) + ("\n" if output_files else ""))

    audit_payload: Dict[str, Any] = {
        "exact_command": exact_command,
        "selected_backend": "emodelx",
        "key_input_args": {
            "project": str(project),
            "sequence": str(sequence),
            "chain_count": int(args.chain_count),
            "protocol": str(args.protocol),
            "emodelx_repo": str(params.get("emodelx_repo_dir") or ""),
            "emodelx_python": str(params.get("emodelx_python") or ""),
            "output_dir": str(output_dir),
        },
        "input_state": state.model_dump(mode="json"),
        "qc_metrics": qc.model_dump(mode="json"),
        "planner_router_action": action.model_dump(mode="json"),
        "preflight": pf,
        "execution_result": result,
        "output_files": output_files,
        "audit_files": {
            "exact_command": str(audit_dir / "exact_command.txt"),
            "stdout": str(audit_dir / "stdout.log"),
            "stderr": str(audit_dir / "stderr.log"),
            "output_files": str(audit_dir / "output_files.txt"),
            "audit_json": str(audit_dir / "audit.json"),
        },
        "scheduler_state_file": str(project / ".cryoem_agent" / "sample_state.json"),
        "scheduler_history_file": str(project / ".cryoem_agent" / "action_history.jsonl"),
    }
    _write_text(audit_dir / "audit.json", json.dumps(audit_payload, ensure_ascii=False, indent=2))

    print(json.dumps(audit_payload["audit_files"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
