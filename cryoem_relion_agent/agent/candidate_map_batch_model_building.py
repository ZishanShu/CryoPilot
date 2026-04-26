"""Build top-k dry-run model-building plans for candidate maps."""

from __future__ import annotations

import copy
import shlex
from pathlib import Path
from typing import Any, Dict, List, Sequence

from agent.candidate_map_router import create_candidate_map_routing_records


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _derive_output_dir(record: Dict[str, Any]) -> str:
    params = record.get("action_parameters")
    if isinstance(params, dict) and str(params.get("output_dir") or "").strip():
        return str(params.get("output_dir")).strip()
    preflight = record.get("preflight")
    if isinstance(preflight, dict) and str(preflight.get("output_path") or "").strip():
        return str(preflight.get("output_path")).strip()
    source_run_dir = str(record.get("source_run_dir") or "").strip()
    project_path = str(record.get("project_path") or "").strip()
    candidate_index = _coerce_int(record.get("candidate_index"), 0)
    if project_path:
        root = Path(project_path).expanduser()
        return str(root / "CandidateMapModelBuilding" / f"candidate_{candidate_index:03d}")
    if source_run_dir:
        root = Path(source_run_dir).expanduser()
        return str(root / "CandidateMapModelBuilding" / f"candidate_{candidate_index:03d}")
    return str(Path("CandidateMapModelBuilding") / f"candidate_{candidate_index:03d}")


def _extract_command_fields(record: Dict[str, Any]) -> tuple[List[str] | None, str | None]:
    preflight = record.get("preflight")
    if isinstance(preflight, dict):
        argv_raw = preflight.get("command_argv")
        if isinstance(argv_raw, list) and argv_raw:
            argv = [str(x) for x in argv_raw]
            return argv, " ".join(shlex.quote(x) for x in argv)
        command_raw = preflight.get("command")
        if isinstance(command_raw, str) and command_raw.strip():
            command = command_raw.strip()
            try:
                return shlex.split(command), command
            except ValueError:
                return None, command
    return None, None


def build_candidate_model_building_plan(
    candidate_map_routing_records: Sequence[Dict[str, Any]],
    top_k: int = 5,
    dry_run: bool = True,
) -> List[Dict[str, Any]]:
    """Build deterministic top-k downstream dry-run plans from routing records."""

    if top_k <= 0:
        return []
    rows = [dict(x) for x in candidate_map_routing_records if isinstance(x, dict)]
    rows = sorted(
        rows,
        key=lambda r: (
            _coerce_int(r.get("qc_rank"), 999999),
            -_coerce_float(r.get("qc_score"), 0.0),
            _coerce_int(r.get("candidate_index"), 999999),
        ),
    )
    plans: List[Dict[str, Any]] = []
    for record in rows[: int(top_k)]:
        command_argv, command = _extract_command_fields(record)
        plan: Dict[str, Any] = {
            "candidate_index": _coerce_int(record.get("candidate_index"), 0),
            "qc_rank": _coerce_int(record.get("qc_rank"), 0),
            "qc_score": _coerce_float(record.get("qc_score"), 0.0),
            "qc_status": str(record.get("qc_status") or record.get("candidate_qc_status") or "unknown"),
            "map_path": record.get("map_path"),
            "candidate_map_path": record.get("map_path"),
            "source_backend": record.get("source_backend"),
            "source_run_dir": record.get("source_run_dir"),
            "latent_source": record.get("latent_source"),
            "weights_path": record.get("weights_path"),
            "selected_backend": record.get("recommended_backend"),
            "selected_action": record.get("downstream_action") or record.get("recommended_action"),
            "routing_reason": record.get("routing_reason"),
            "model_building_preflight": copy.deepcopy(record.get("preflight"))
            if isinstance(record.get("preflight"), dict)
            else None,
            "output_dir": _derive_output_dir(record),
            "command_argv": command_argv,
            "command": command,
            "dry_run": bool(dry_run),
            "planned_status": "planned",
        }
        plans.append(plan)
    return plans


def build_candidate_model_building_plans_from_state(
    state: Any,
    sequence_path: str | None = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Build candidate batch model-building plans directly from state."""

    state_input = dict(state) if isinstance(state, dict) else state
    if isinstance(state_input, dict) and sequence_path:
        state_input = dict(state_input)
        state_input["sequence_path"] = sequence_path
    records = create_candidate_map_routing_records(
        state_input,
        max_candidates=max(0, int(top_k)),
    )
    return build_candidate_model_building_plan(records, top_k=top_k, dry_run=True)

