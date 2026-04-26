"""Route reconstruction candidate maps into model-building dry-run payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from agent.backend_registry import model_building_backend_to_action_name
from agent.candidate_map_qc import evaluate_candidate_maps, rank_candidate_maps
from agent.model_building_preflight import preflight_model_building

DEFAULT_MAX_CANDIDATES = 5
BRANCH_NAME = "candidate_map_model_building"


def _state_value(state: Any, key: str, default: Any = None) -> Any:
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _as_candidate_dict(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return dict(item)
    return {"map_path": str(item)}


def _candidate_sources(state: Any) -> Iterable[Any]:
    direct = _state_value(state, "candidate_maps")
    if isinstance(direct, list):
        yield from direct

    last_execution = _state_value(state, "last_execution")
    if not isinstance(last_execution, dict):
        return

    for key in ("candidate_maps", "sampled_volumes"):
        value = last_execution.get(key)
        if isinstance(value, list):
            yield from value

    artifacts = last_execution.get("output_artifacts")
    if isinstance(artifacts, dict):
        sampled = artifacts.get("sampled_volumes")
        if isinstance(sampled, list):
            yield from sampled


def read_candidate_maps(state: Any) -> List[Dict[str, Any]]:
    """Read candidate maps from dict state, SampleState, or last_execution."""

    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in _candidate_sources(state):
        candidate = _as_candidate_dict(item)
        map_path = str(candidate.get("map_path") or candidate.get("path") or "").strip()
        key = map_path or repr(candidate)
        if key in seen:
            continue
        seen.add(key)
        if map_path:
            candidate["map_path"] = map_path
        out.append(candidate)
    return out


def select_candidate_maps(
    candidates: Iterable[Dict[str, Any]],
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> List[Dict[str, Any]]:
    """Select the first N candidate maps while preserving source order."""

    limit = max(0, int(max_candidates))
    return [dict(x) for x in list(candidates)[:limit]]


def _candidate_map_path(candidate: Dict[str, Any]) -> Optional[Path]:
    raw = str(candidate.get("map_path") or "").strip()
    return Path(raw).expanduser() if raw else None


def _sequence_path(state: Any) -> Optional[str]:
    raw = _state_value(state, "sequence_path")
    if raw is None or not str(raw).strip():
        return None
    return str(raw).strip()


def _choose_backend(
    *,
    base_params: Dict[str, Any],
    sequence_path: Optional[str],
    af3_dir: Optional[str],
    allow_seq_free: bool,
    emodelx_config_path: Optional[Path],
    mica_config_path: Optional[Path],
) -> tuple[Optional[str], Optional[Dict[str, Any]], str]:
    if af3_dir and sequence_path:
        mica_params = {**base_params, "sequence_path": sequence_path, "af3_dir": af3_dir}
        mica_pf = preflight_model_building("mica", mica_params, mica_config_path=mica_config_path)
        if mica_pf.get("available"):
            return "mica", mica_pf, "af3_dir_present_and_mica_available"

    if sequence_path:
        params = {**base_params, "sequence_path": sequence_path}
        pf = preflight_model_building("modelangelo", params)
        return "modelangelo", pf, "sequence_path_present_modelangelo_default"

    if allow_seq_free:
        params = {**base_params, "emodelx_protocol": "seq_free"}
        pf = preflight_model_building("emodelx", params, emodelx_config_path=emodelx_config_path)
        return "emodelx", pf, "sequence_missing_seq_free_emodelx"

    return None, None, "missing_sequence_and_seq_free_not_allowed"


def create_candidate_map_routing_records(
    state: Any,
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    allow_seq_free: bool = False,
    af3_dir: Optional[str] = None,
    output_root: Optional[str | Path] = None,
    emodelx_config_path: Optional[Path] = None,
    mica_config_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Create model-building routing records for candidate map dictionaries."""

    records: List[Dict[str, Any]] = []
    sequence_path = _sequence_path(state)
    project_path = str(_state_value(state, "project_path", "") or "")
    root = Path(output_root or (Path(project_path) / "CandidateMapModelBuilding" if project_path else "CandidateMapModelBuilding"))
    selected_candidates = select_candidate_maps(read_candidate_maps(state), max_candidates=max_candidates)
    evaluated_candidates = evaluate_candidate_maps([dict(x) for x in selected_candidates])
    ranked_candidates = rank_candidate_maps(evaluated_candidates)
    ranked_candidate_maps: List[Dict[str, Any]] = []

    for fallback_idx, candidate in enumerate(ranked_candidates):
        candidate_index = candidate.get("candidate_index")
        if candidate_index is None:
            candidate_index = fallback_idx
        candidate_map_qc = candidate.get("candidate_map_qc") if isinstance(candidate.get("candidate_map_qc"), dict) else {}
        qc_score = float(candidate.get("qc_score") or candidate_map_qc.get("qc_score") or 0.0)
        qc_status = str(candidate.get("qc_status") or candidate_map_qc.get("qc_status") or "failed")
        qc_reason = str(candidate.get("qc_reason") or candidate_map_qc.get("qc_reason") or "")
        ranked_candidate_maps.append(
            {
                "candidate_index": candidate_index,
                "map_path": candidate.get("map_path"),
                "source_backend": candidate.get("source_backend"),
                "source_run_dir": candidate.get("source_run_dir"),
                "latent_source": candidate.get("latent_source"),
                "weights_path": candidate.get("weights_path"),
                "qc_score": qc_score,
                "qc_status": qc_status,
                "qc_reason": qc_reason,
                "candidate_map_qc": candidate_map_qc,
                "qc_rank": fallback_idx,
            }
        )
        map_path = _candidate_map_path(candidate)
        exists = bool(map_path and map_path.is_file())
        record: Dict[str, Any] = {
            "candidate_index": candidate_index,
            "source_backend": candidate.get("source_backend"),
            "source_run_dir": candidate.get("source_run_dir"),
            "map_path": str(map_path) if map_path else None,
            "sequence_path": sequence_path,
            "latent_source": candidate.get("latent_source"),
            "weights_path": candidate.get("weights_path"),
            "candidate_qc_status": candidate.get("qc_status"),
            "candidate_map_qc": candidate_map_qc,
            "qc_score": qc_score,
            "qc_status": qc_status,
            "qc_reason": qc_reason,
            "qc_rank": fallback_idx,
            "branch_name": BRANCH_NAME,
        }
        if not exists:
            record.update(
                {
                    "recommended_backend": None,
                    "recommended_action": None,
                    "routing_reason": "missing_map_path" if map_path is None else "map_path_unavailable",
                    "routing_available": False,
                    "preflight": {
                        "available": False,
                        "execution_mode": "unavailable",
                        "blocked_reason": "missing_map_path" if map_path is None else f"map_path_unavailable: {map_path}",
                    },
                    "downstream_action": None,
                }
            )
            records.append(record)
            continue

        base_params = {
            "map_path": str(map_path),
            "output_dir": str(root / f"candidate_{int(candidate_index):03d}"),
        }
        backend, pf, reason = _choose_backend(
            base_params=base_params,
            sequence_path=sequence_path,
            af3_dir=af3_dir,
            allow_seq_free=allow_seq_free,
            emodelx_config_path=emodelx_config_path,
            mica_config_path=mica_config_path,
        )
        action_params = dict(base_params)
        if sequence_path:
            action_params["sequence_path"] = sequence_path
        if backend == "emodelx" and allow_seq_free and not sequence_path:
            action_params["emodelx_protocol"] = "seq_free"
        if backend == "mica" and af3_dir:
            action_params["af3_dir"] = af3_dir

        record.update(
            {
                "recommended_backend": backend,
                "recommended_action": model_building_backend_to_action_name(backend) if backend else None,
                "routing_reason": reason,
                "routing_available": bool(backend and pf and pf.get("available")),
                "preflight": pf,
                "downstream_action": model_building_backend_to_action_name(backend) if backend else None,
                "action_parameters": action_params,
            }
        )
        records.append(record)
    if not records:
        return records
    selected = records[0]
    selected_candidate_index = selected.get("candidate_index")
    selected_candidate_qc_score = float(selected.get("qc_score") or 0.0)
    selection_reason = "highest_qc_score"
    for record in records:
        record["selected_candidate_index"] = selected_candidate_index
        record["selected_candidate_qc_score"] = selected_candidate_qc_score
        record["candidate_map_selection_reason"] = selection_reason
        record["selected_candidate"] = record.get("candidate_index") == selected_candidate_index
        record["ranked_candidate_maps"] = list(ranked_candidate_maps)
    return records


def action_parameters_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Build action parameters that preserve candidate provenance."""

    params = {
        "branch_name": BRANCH_NAME,
        "candidate_index": record.get("candidate_index"),
        "source_backend": record.get("source_backend"),
        "source_run_dir": record.get("source_run_dir"),
        "map_path": record.get("map_path"),
        "candidate_map_path": record.get("map_path"),
        "sequence_path": record.get("sequence_path"),
        "latent_source": record.get("latent_source"),
        "weights_path": record.get("weights_path"),
        "candidate_qc_status": record.get("candidate_qc_status"),
        "candidate_map_qc": record.get("candidate_map_qc"),
        "selected_candidate_qc_score": record.get("selected_candidate_qc_score", record.get("qc_score")),
        "candidate_map_selection_reason": record.get("candidate_map_selection_reason", "highest_qc_score"),
        "selected_candidate_index": record.get("selected_candidate_index", record.get("candidate_index")),
        "ranked_candidate_maps": record.get("ranked_candidate_maps"),
        "candidate_map_routing_record": record,
        "routing_reason": record.get("routing_reason"),
        "routing_triggered": True,
        "model_building_backend": record.get("recommended_backend"),
    }
    action_params = record.get("action_parameters")
    if isinstance(action_params, dict):
        params.update({k: v for k, v in action_params.items() if v not in (None, "")})
    pf = record.get("preflight")
    if isinstance(pf, dict):
        params["model_building_preflight"] = pf
        params["model_building_availability"] = "available" if pf.get("available") else "unavailable"
        params["model_building_execution_mode"] = str(pf.get("execution_mode") or "unavailable")
        params["model_building_blocked_reason"] = pf.get("blocked_reason") or ""
        params["model_building_missing_inputs"] = list(pf.get("missing_inputs") or [])
        params["model_building_missing_dependencies"] = list(pf.get("missing_dependencies") or [])
    return params


class CandidateMapRouter:
    """Small OO wrapper for callers that prefer an injectable router."""

    def __init__(
        self,
        *,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        allow_seq_free: bool = False,
        af3_dir: Optional[str] = None,
        emodelx_config_path: Optional[Path] = None,
        mica_config_path: Optional[Path] = None,
    ) -> None:
        self.max_candidates = max_candidates
        self.allow_seq_free = allow_seq_free
        self.af3_dir = af3_dir
        self.emodelx_config_path = emodelx_config_path
        self.mica_config_path = mica_config_path

    def route(self, state: Any) -> List[Dict[str, Any]]:
        return create_candidate_map_routing_records(
            state,
            max_candidates=self.max_candidates,
            allow_seq_free=self.allow_seq_free,
            af3_dir=self.af3_dir,
            emodelx_config_path=self.emodelx_config_path,
            mica_config_path=self.mica_config_path,
        )
