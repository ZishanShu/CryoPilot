"""LLM-supervised routing v0 for cryo-EM candidate-map model building.

This module adds an optional LLM-driven decision layer on top of the
rule-based candidate map router. The LLM (or a deterministic mock LLM)
sees a compact observation of the current state, returns a structured
JSON decision, and the decision is validated. If validation fails, the
caller falls back to the rule-based selection.

This is a minimal-publishable v0:
- No real LLM API is invoked here; provider="mock" is sufficient.
- No real model-building backend is launched.
- Existing rule-based default behavior is preserved.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agent.candidate_map_qc import evaluate_candidate_maps, rank_candidate_maps
from agent.candidate_map_router import (
    create_candidate_map_routing_records,
    read_candidate_maps,
)


ALLOWED_BACKENDS = ("modelangelo", "emodelx", "mica", "human_review")
BACKEND_TO_ACTION: Dict[str, str] = {
    "modelangelo": "run_modelangelo",
    "emodelx": "run_emodelx",
    "mica": "run_mica",
    "human_review": "human_review",
}
DECISION_TYPE = "candidate_map_model_building"


def _state_value(state: Any, key: str, default: Any = None) -> Any:
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _bool_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def _compact_candidate(record: Dict[str, Any]) -> Dict[str, Any]:
    """Project a routing record into a compact LLM-friendly dict."""

    cqc = record.get("candidate_map_qc") if isinstance(record.get("candidate_map_qc"), dict) else {}
    pf = record.get("preflight") if isinstance(record.get("preflight"), dict) else {}
    return {
        "candidate_index": record.get("candidate_index"),
        "map_path": record.get("map_path"),
        "qc_score": record.get("qc_score"),
        "qc_status": record.get("qc_status"),
        "qc_reason": record.get("qc_reason"),
        "finite_fraction": cqc.get("finite_fraction"),
        "density_std": cqc.get("density_std"),
        "edge_artifact_score": cqc.get("edge_artifact_score"),
        "latent_source": record.get("latent_source"),
        "weights_path": record.get("weights_path"),
        "recommended_backend": record.get("recommended_backend"),
        "recommended_action": record.get("recommended_action"),
        "preflight": {
            "available": bool(pf.get("available")),
            "blocked_reason": pf.get("blocked_reason"),
        },
    }


def _routing_records_for_state(state: Any) -> List[Dict[str, Any]]:
    """Return rule-based routing records for ``state`` (use cached ones if present)."""

    cached = _state_value(state, "candidate_map_routing_records")
    if isinstance(cached, list) and cached:
        return [r for r in cached if isinstance(r, dict)]

    last_exec = _state_value(state, "last_execution")
    if isinstance(last_exec, dict):
        cached = last_exec.get("candidate_map_routing_records")
        if isinstance(cached, list) and cached:
            return [r for r in cached if isinstance(r, dict)]

    af3_dir: Optional[str] = None
    allow_seq_free = False
    if isinstance(last_exec, dict):
        if last_exec.get("af3_dir"):
            af3_dir = str(last_exec.get("af3_dir"))
        if last_exec.get("allow_seq_free"):
            allow_seq_free = bool(last_exec.get("allow_seq_free"))
    return create_candidate_map_routing_records(
        state,
        allow_seq_free=allow_seq_free,
        af3_dir=af3_dir,
    )


def build_llm_observation(state: Any) -> Dict[str, Any]:
    """Build a compact observation dict that the LLM can act on."""

    last_execution = _state_value(state, "last_execution") or {}
    if not isinstance(last_execution, dict):
        last_execution = {}

    candidates = read_candidate_maps(state)
    routing_records = _routing_records_for_state(state)

    ranked_raw = _state_value(state, "ranked_candidate_maps")
    if not isinstance(ranked_raw, list) or not ranked_raw:
        if routing_records:
            top_records = routing_records
            ranked_raw = top_records
        elif candidates:
            ranked_raw = rank_candidate_maps(evaluate_candidate_maps([dict(c) for c in candidates]))
        else:
            ranked_raw = []

    ranked_compact = [
        _compact_candidate(r if isinstance(r, dict) else {})
        for r in ranked_raw
        if isinstance(r, dict)
    ]
    candidates_compact = [
        _compact_candidate(r if isinstance(r, dict) else {})
        for r in routing_records
        if isinstance(r, dict)
    ] or ranked_compact

    selected_candidate_index = _state_value(state, "selected_candidate_index")
    selected_candidate_qc_score = _state_value(state, "selected_candidate_qc_score")
    selection_reason = _state_value(state, "candidate_map_selection_reason")
    rule_based_action: Optional[str] = None
    rule_based_backend: Optional[str] = None
    if routing_records:
        top = routing_records[0]
        if selected_candidate_index is None and top.get("selected_candidate_index") is not None:
            selected_candidate_index = top.get("selected_candidate_index")
        if selected_candidate_index is None:
            selected_candidate_index = top.get("candidate_index")
        if selected_candidate_qc_score is None:
            selected_candidate_qc_score = top.get("selected_candidate_qc_score") or top.get("qc_score")
        if not selection_reason:
            selection_reason = top.get("candidate_map_selection_reason") or "highest_qc_score"
        rule_based_backend = top.get("recommended_backend")
        rule_based_action = top.get("downstream_action") or top.get("recommended_action")

    available_backends: List[str] = []
    for record in routing_records:
        pf = record.get("preflight") if isinstance(record.get("preflight"), dict) else {}
        if pf.get("available") and record.get("recommended_backend"):
            backend = str(record.get("recommended_backend"))
            if backend not in available_backends:
                available_backends.append(backend)

    preflight_summary = {
        "available_backends": available_backends,
        "candidates_with_available_preflight": sum(
            1
            for r in routing_records
            if isinstance(r.get("preflight"), dict) and r["preflight"].get("available")
        ),
        "candidates_total": len(routing_records),
    }

    af3_dir = None
    if isinstance(last_execution, dict) and last_execution.get("af3_dir"):
        af3_dir = str(last_execution.get("af3_dir"))

    return {
        "stage": _state_value(state, "stage"),
        "last_execution": {
            "action_name": last_execution.get("action_name") if isinstance(last_execution, dict) else None,
            "status": last_execution.get("status") if isinstance(last_execution, dict) else None,
        },
        "candidate_count": len(candidates),
        "candidate_maps": candidates_compact,
        "ranked_candidate_maps": ranked_compact,
        "candidate_map_routing_records": [
            {
                "candidate_index": r.get("candidate_index"),
                "recommended_backend": r.get("recommended_backend"),
                "recommended_action": r.get("downstream_action") or r.get("recommended_action"),
                "routing_available": bool(r.get("routing_available")),
                "qc_score": r.get("qc_score"),
                "preflight_available": bool(
                    isinstance(r.get("preflight"), dict) and r["preflight"].get("available")
                ),
            }
            for r in routing_records
        ],
        "selected_candidate_index": selected_candidate_index,
        "selected_candidate_qc_score": selected_candidate_qc_score,
        "candidate_map_selection_reason": selection_reason,
        "sequence_path_present": _bool_present(_state_value(state, "sequence_path")),
        "af3_dir_present": _bool_present(af3_dir),
        "af3_dir": af3_dir,
        "model_building_preflight_summary": preflight_summary,
        "rule_based_selected_backend": rule_based_backend,
        "rule_based_selected_action": rule_based_action,
    }


def build_llm_decision_prompt(observation: Dict[str, Any]) -> str:
    """Build a prompt instructing the LLM to emit JSON only (no Markdown)."""

    schema = {
        "decision_type": "candidate_map_model_building",
        "selected_candidate_index": 0,
        "selected_backend": "modelangelo|emodelx|mica|human_review",
        "selected_action": "run_modelangelo|run_emodelx|run_mica|human_review",
        "confidence": 0.0,
        "decision_reason": "short rationale",
        "risk_flags": [],
        "fallback_allowed": True,
    }

    rules = [
        "Pick the next model-building action for the cryo-EM candidate maps.",
        "Prefer high qc_score, available preflight, sequence/AF3 priors when present.",
        "ModelAngelo is the default strong baseline whenever a sequence is available.",
        "EModelX is a candidate when no sequence is available (seq-free).",
        "MICA is only allowed when both af3_dir and its preflight are available.",
        "If inputs are insufficient or candidate map quality is too low, choose human_review.",
        "Never fabricate a map_path that is not in the observation candidate list.",
        "Never select a backend whose preflight is unavailable, unless decision_type is human_review.",
    ]

    output_rules = [
        "Output strictly JSON. Do NOT wrap output in Markdown code fences.",
        "Do not add any prose before or after the JSON.",
        "selected_action must match selected_backend (modelangelo->run_modelangelo, "
        "emodelx->run_emodelx, mica->run_mica, human_review->human_review).",
        "selected_candidate_index must reference a candidate present in candidate_maps.",
        "confidence must be a float in [0.0, 1.0].",
        "decision_reason must be a non-empty short string.",
    ]

    sections = [
        "[Task]",
        "You are an LLM supervisor for a cryo-EM candidate-map model-building router.",
        "",
        "[Decision Rules]",
        *(f"- {r}" for r in rules),
        "",
        "[Output Rules]",
        *(f"- {r}" for r in output_rules),
        "",
        "[Output JSON schema example]",
        json.dumps(schema, indent=2, ensure_ascii=False),
        "",
        "[Observation]",
        json.dumps(observation, indent=2, ensure_ascii=False, default=str),
        "",
        "[Response]",
        "Return only JSON. No Markdown.",
    ]
    return "\n".join(sections)


def parse_llm_decision(text: str) -> Dict[str, Any]:
    """Parse a JSON decision string. Reject Markdown code fences."""

    if not isinstance(text, str):
        raise ValueError("LLM response must be a string")
    s = text.strip()
    if not s:
        raise ValueError("empty LLM response")
    if "```" in s:
        raise ValueError("Markdown code fences are not allowed in LLM response")
    try:
        decoded = json.loads(s)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON from LLM: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ValueError("LLM JSON must be an object")
    return decoded


def validate_llm_decision(decision: Any, observation: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a parsed LLM decision against the observation."""

    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(decision, dict):
        return {
            "valid": False,
            "validated_decision": None,
            "errors": ["decision must be a JSON object"],
            "warnings": [],
        }

    decision_type = str(decision.get("decision_type") or "").strip()
    if not decision_type:
        decision_type = DECISION_TYPE
        warnings.append("decision_type missing; defaulted to candidate_map_model_building")
    elif decision_type != DECISION_TYPE:
        warnings.append(f"unexpected decision_type {decision_type!r}")

    backend = str(decision.get("selected_backend") or "").strip().lower()
    action = str(decision.get("selected_action") or "").strip().lower()
    if backend not in ALLOWED_BACKENDS:
        errors.append(f"selected_backend must be one of {ALLOWED_BACKENDS}; got {backend!r}")
    expected_action = BACKEND_TO_ACTION.get(backend)
    if expected_action is not None and action != expected_action:
        errors.append(
            f"selected_action {action!r} does not match selected_backend {backend!r} "
            f"(expected {expected_action!r})"
        )

    raw_idx = decision.get("selected_candidate_index")
    try:
        idx_int = int(raw_idx)
    except (TypeError, ValueError):
        idx_int = None
        errors.append("selected_candidate_index must be an integer")

    candidates = observation.get("candidate_maps") or []
    candidate_map: Dict[int, Dict[str, Any]] = {}
    for c in candidates:
        if isinstance(c, dict) and c.get("candidate_index") is not None:
            try:
                candidate_map[int(c["candidate_index"])] = c
            except (TypeError, ValueError):
                continue
    if idx_int is not None and idx_int not in candidate_map:
        errors.append(
            f"selected_candidate_index {idx_int} not present in candidate_maps "
            f"({sorted(candidate_map.keys())})"
        )

    if (
        idx_int is not None
        and idx_int in candidate_map
        and backend != "human_review"
    ):
        cand = candidate_map[idx_int]
        pf = cand.get("preflight") if isinstance(cand.get("preflight"), dict) else {}
        if not pf.get("available"):
            errors.append(
                f"preflight for backend {backend!r} on candidate {idx_int} is unavailable"
            )
        if not cand.get("map_path"):
            errors.append(f"selected candidate {idx_int} has no map_path")

    confidence = decision.get("confidence")
    try:
        conf_f = float(confidence)
    except (TypeError, ValueError):
        conf_f = None
        errors.append("confidence must be a number in [0, 1]")
    else:
        if not (0.0 <= conf_f <= 1.0):
            errors.append(f"confidence {conf_f} out of range [0, 1]")

    reason = decision.get("decision_reason")
    if not isinstance(reason, str) or not reason.strip():
        errors.append("decision_reason must be a non-empty short string")
        reason_clean = ""
    else:
        reason_clean = reason.strip()
        if len(reason_clean) > 500:
            warnings.append("decision_reason longer than 500 chars; truncating in validated_decision")
            reason_clean = reason_clean[:500]

    risk_flags_raw = decision.get("risk_flags") or []
    if not isinstance(risk_flags_raw, list):
        warnings.append("risk_flags must be a list; coerced to []")
        risk_flags = []
    else:
        risk_flags = [str(x) for x in risk_flags_raw]

    fallback_allowed = bool(decision.get("fallback_allowed", True))

    if errors:
        return {
            "valid": False,
            "validated_decision": None,
            "errors": errors,
            "warnings": warnings,
        }

    validated = {
        "decision_type": DECISION_TYPE,
        "selected_candidate_index": idx_int,
        "selected_backend": backend,
        "selected_action": action,
        "confidence": float(conf_f) if conf_f is not None else 0.0,
        "decision_reason": reason_clean,
        "risk_flags": risk_flags,
        "fallback_allowed": fallback_allowed,
    }
    return {
        "valid": True,
        "validated_decision": validated,
        "errors": [],
        "warnings": warnings,
    }


def _rule_based_decision_from_observation(observation: Dict[str, Any]) -> Dict[str, Any]:
    """Build a fallback decision from rule-based ranking baked into the observation."""

    candidates = observation.get("candidate_maps") or []
    selected_idx = observation.get("selected_candidate_index")
    chosen: Optional[Dict[str, Any]] = None
    if selected_idx is not None:
        for c in candidates:
            if isinstance(c, dict) and c.get("candidate_index") == selected_idx:
                chosen = c
                break
    if chosen is None and candidates:
        chosen = candidates[0]

    if chosen is None:
        return {
            "decision_type": DECISION_TYPE,
            "selected_candidate_index": -1,
            "selected_backend": "human_review",
            "selected_action": "human_review",
            "confidence": 0.5,
            "decision_reason": "no candidate maps available; defer to human review",
            "risk_flags": ["no_candidates"],
            "fallback_allowed": False,
        }

    backend = str(chosen.get("recommended_backend") or "").strip().lower() or "modelangelo"
    if backend not in ALLOWED_BACKENDS:
        backend = "human_review"
    action = BACKEND_TO_ACTION.get(backend) or "human_review"
    pf = chosen.get("preflight") if isinstance(chosen.get("preflight"), dict) else {}
    risk_flags: List[str] = []
    if not pf.get("available") and backend != "human_review":
        risk_flags.append("preflight_unavailable")
        backend = "human_review"
        action = "human_review"
    if not chosen.get("map_path") and backend != "human_review":
        risk_flags.append("missing_map_path")
        backend = "human_review"
        action = "human_review"

    qc_score = chosen.get("qc_score")
    return {
        "decision_type": DECISION_TYPE,
        "selected_candidate_index": int(chosen.get("candidate_index"))
        if chosen.get("candidate_index") is not None
        else 0,
        "selected_backend": backend,
        "selected_action": action,
        "confidence": 0.6,
        "decision_reason": (
            f"rule-based fallback: highest qc_score candidate "
            f"(qc_score={qc_score}, backend={backend})"
        ),
        "risk_flags": risk_flags,
        "fallback_allowed": False,
    }


def _mock_llm_response(observation: Dict[str, Any]) -> str:
    """Deterministic mock LLM response that mimics rule-based selection."""

    candidates = observation.get("candidate_maps") or []
    selected_idx = observation.get("selected_candidate_index")
    chosen: Optional[Dict[str, Any]] = None
    if selected_idx is not None:
        for c in candidates:
            if isinstance(c, dict) and c.get("candidate_index") == selected_idx:
                chosen = c
                break
    if chosen is None and candidates:
        chosen = candidates[0]

    if chosen is None:
        decision = {
            "decision_type": DECISION_TYPE,
            "selected_candidate_index": -1,
            "selected_backend": "human_review",
            "selected_action": "human_review",
            "confidence": 0.4,
            "decision_reason": "mock LLM: no candidate maps available; defer to human review",
            "risk_flags": ["no_candidates"],
            "fallback_allowed": True,
        }
        return json.dumps(decision)

    backend = str(chosen.get("recommended_backend") or "").strip().lower() or "modelangelo"
    if backend not in ALLOWED_BACKENDS:
        backend = "human_review"
    pf = chosen.get("preflight") if isinstance(chosen.get("preflight"), dict) else {}
    risk_flags: List[str] = []
    if not pf.get("available") and backend != "human_review":
        risk_flags.append("preflight_unavailable")
        backend = "human_review"
    action = BACKEND_TO_ACTION.get(backend) or "human_review"
    qc_score = chosen.get("qc_score")
    decision = {
        "decision_type": DECISION_TYPE,
        "selected_candidate_index": int(chosen.get("candidate_index"))
        if chosen.get("candidate_index") is not None
        else 0,
        "selected_backend": backend,
        "selected_action": action,
        "confidence": 0.85,
        "decision_reason": (
            f"mock LLM: highest qc_score candidate "
            f"(qc_score={qc_score}, backend={backend})"
        ),
        "risk_flags": risk_flags,
        "fallback_allowed": True,
    }
    return json.dumps(decision)


def llm_supervised_route(
    state: Any,
    llm_response_text: Optional[str] = None,
    provider: str = "mock",
) -> Dict[str, Any]:
    """Run LLM-supervised routing for candidate-map model building.

    - ``provider="mock"`` and no ``llm_response_text``: use the deterministic
      mock LLM (selects rule-based highest-qc candidate).
    - ``llm_response_text`` provided: parse + validate; fall back to rule-based
      if validation fails.
    """

    observation = build_llm_observation(state)

    raw_response: Optional[str]
    decision: Optional[Dict[str, Any]] = None
    parse_error: Optional[str] = None

    if llm_response_text is None:
        if provider == "mock":
            raw_response = _mock_llm_response(observation)
        else:
            raw_response = _mock_llm_response(observation)
    else:
        raw_response = llm_response_text

    try:
        decision = parse_llm_decision(raw_response or "")
    except ValueError as exc:
        parse_error = str(exc)
        decision = None

    if decision is None:
        validation = {
            "valid": False,
            "validated_decision": None,
            "errors": [parse_error or "could not parse LLM response"],
            "warnings": [],
        }
    else:
        validation = validate_llm_decision(decision, observation)

    fallback_used = False
    fallback_reason: Optional[str] = None
    final_decision = validation.get("validated_decision")

    if not validation.get("valid") or final_decision is None:
        fallback_used = True
        fallback_reason = "; ".join(validation.get("errors") or []) or "invalid LLM decision"
        final_decision = _rule_based_decision_from_observation(observation)

    return {
        "mode": "llm_supervised",
        "provider": provider,
        "observation": observation,
        "raw_llm_response": raw_response,
        "decision": final_decision,
        "validation": validation,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "selected_candidate_index": final_decision.get("selected_candidate_index"),
        "selected_backend": final_decision.get("selected_backend"),
        "selected_action": final_decision.get("selected_action"),
        "decision_reason": final_decision.get("decision_reason"),
        "risk_flags": list(final_decision.get("risk_flags") or []),
    }
