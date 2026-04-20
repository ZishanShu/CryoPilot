"""Experiment protocol layer for paper-grade baseline comparisons.

A protocol describes *how* the agent behaves during a single experiment case:
- fixed_workflow: run exactly one allowed backend, no rerouting
- rule_only_agent: use policy / router but no LLM supervision
- llm_supervised_agent: collect LLM supervision payloads at key stop points

Protocols do NOT execute anything directly. They are parsed into a
``ExperimentProtocol`` dataclass consumed by ``baseline_runner``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

SUPPORTED_PROTOCOL_KINDS: Tuple[str, ...] = (
    "fixed_workflow",
    "rule_only_agent",
    "llm_supervised_agent",
)

SUPPORTED_ROUTING_MODES: Tuple[str, ...] = (
    "fixed",
    "rule",
    "rule_with_llm",
)

SUPPORTED_STOP_POINTS: Tuple[str, ...] = (
    "preflight",
    "execution",
    "evaluation",
    "diff",
)


@dataclass
class ExperimentProtocol:
    """Normalized protocol config used by the baseline runner."""

    protocol_name: str
    protocol_kind: str
    allowed_backends: List[str] = field(default_factory=list)
    routing_mode: str = "fixed"
    llm_supervision_enabled: bool = False
    fallback_policy: str = "none"
    stop_points: List[str] = field(default_factory=list)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    recovery_policy: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProtocolValidationError(ValueError):
    """Raised when a protocol config is malformed."""


def _as_str_list(value: Any, *, field_name: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ProtocolValidationError(
                    f"{field_name} must be a list of non-empty strings"
                )
            out.append(item.strip())
        return out
    raise ProtocolValidationError(f"{field_name} must be a list of strings")


def parse_protocol(raw: Mapping[str, Any]) -> ExperimentProtocol:
    """Parse a raw mapping into a validated ``ExperimentProtocol``."""
    if not isinstance(raw, Mapping):
        raise ProtocolValidationError("protocol must be a mapping")

    name = str(raw.get("protocol_name") or "").strip()
    if not name:
        raise ProtocolValidationError("protocol_name is required")

    kind = str(raw.get("protocol_kind") or "").strip()
    if kind not in SUPPORTED_PROTOCOL_KINDS:
        raise ProtocolValidationError(
            f"protocol_kind must be one of {SUPPORTED_PROTOCOL_KINDS}, got {kind!r}"
        )

    allowed = _as_str_list(raw.get("allowed_backends"), field_name="allowed_backends")
    if kind == "fixed_workflow" and len(allowed) != 1:
        raise ProtocolValidationError(
            "fixed_workflow protocol requires exactly one allowed_backend"
        )
    if kind != "fixed_workflow" and not allowed:
        raise ProtocolValidationError(
            "rule_only_agent / llm_supervised_agent require at least one allowed_backend"
        )

    routing_mode = str(raw.get("routing_mode") or "").strip() or (
        "fixed" if kind == "fixed_workflow" else "rule"
    )
    if routing_mode not in SUPPORTED_ROUTING_MODES:
        raise ProtocolValidationError(
            f"routing_mode must be one of {SUPPORTED_ROUTING_MODES}"
        )

    llm_supervised = bool(raw.get("llm_supervision_enabled", False))
    if kind == "llm_supervised_agent" and not llm_supervised:
        llm_supervised = True
    if kind == "fixed_workflow" and llm_supervised:
        raise ProtocolValidationError(
            "fixed_workflow protocol must have llm_supervision_enabled=false"
        )

    fallback_policy = str(raw.get("fallback_policy") or "none").strip() or "none"
    if fallback_policy not in {"none", "registry_default", "router"}:
        raise ProtocolValidationError(
            "fallback_policy must be one of: none | registry_default | router"
        )

    stop_points = _as_str_list(raw.get("stop_points"), field_name="stop_points")
    for sp in stop_points:
        if sp not in SUPPORTED_STOP_POINTS:
            raise ProtocolValidationError(
                f"stop_point {sp!r} not in {SUPPORTED_STOP_POINTS}"
            )

    success_criteria = raw.get("success_criteria") or {}
    if not isinstance(success_criteria, Mapping):
        raise ProtocolValidationError("success_criteria must be a mapping")

    recovery_policy = raw.get("recovery_policy") or {}
    if not isinstance(recovery_policy, Mapping):
        raise ProtocolValidationError("recovery_policy must be a mapping")

    notes = raw.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise ProtocolValidationError("notes must be a string if provided")

    return ExperimentProtocol(
        protocol_name=name,
        protocol_kind=kind,
        allowed_backends=allowed,
        routing_mode=routing_mode,
        llm_supervision_enabled=llm_supervised,
        fallback_policy=fallback_policy,
        stop_points=list(stop_points),
        success_criteria=dict(success_criteria),
        recovery_policy=dict(recovery_policy),
        notes=notes,
    )


def load_protocol(path: Path) -> ExperimentProtocol:
    """Load a protocol from YAML or JSON file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"protocol file not found: {p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif p.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ProtocolValidationError(f"cannot parse {p}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ProtocolValidationError(f"{p} must deserialize to a mapping")
    return parse_protocol(data)


def load_protocols_from_dir(dir_path: Path) -> Dict[str, ExperimentProtocol]:
    """Load every *.yaml / *.yml / *.json file in a directory as a protocol."""
    d = Path(dir_path)
    out: Dict[str, ExperimentProtocol] = {}
    if not d.is_dir():
        return out
    for child in sorted(d.iterdir()):
        if not child.is_file():
            continue
        if child.suffix.lower() not in {".yaml", ".yml", ".json"}:
            continue
        proto = load_protocol(child)
        if proto.protocol_name in out:
            raise ProtocolValidationError(
                f"duplicate protocol_name {proto.protocol_name!r} in {d}"
            )
        out[proto.protocol_name] = proto
    return out


def default_protocols() -> Dict[str, ExperimentProtocol]:
    """Return the three minimal protocols required by the paper baseline set."""
    return {
        "fixed_workflow_topaz": parse_protocol(
            {
                "protocol_name": "fixed_workflow_topaz",
                "protocol_kind": "fixed_workflow",
                "allowed_backends": ["topaz"],
                "routing_mode": "fixed",
                "llm_supervision_enabled": False,
                "fallback_policy": "none",
                "stop_points": [],
                "success_criteria": {"require_non_empty_output": True},
                "recovery_policy": {"retry_on_transient_error": False},
                "notes": "Baseline 1: run only topaz, no rerouting, no LLM.",
            }
        ),
        "rule_only_agent": parse_protocol(
            {
                "protocol_name": "rule_only_agent",
                "protocol_kind": "rule_only_agent",
                "allowed_backends": ["topaz", "cryodrgn", "emodelx", "mica"],
                "routing_mode": "rule",
                "llm_supervision_enabled": False,
                "fallback_policy": "registry_default",
                "stop_points": ["preflight"],
                "success_criteria": {"require_non_empty_output": True},
                "recovery_policy": {"retry_on_transient_error": False},
                "notes": "Baseline 2: routed by registry/policy, no LLM.",
            }
        ),
        "llm_supervised_agent": parse_protocol(
            {
                "protocol_name": "llm_supervised_agent",
                "protocol_kind": "llm_supervised_agent",
                "allowed_backends": ["topaz", "cryodrgn", "emodelx", "mica"],
                "routing_mode": "rule_with_llm",
                "llm_supervision_enabled": True,
                "fallback_policy": "router",
                "stop_points": ["preflight", "execution", "evaluation", "diff"],
                "success_criteria": {"require_non_empty_output": True},
                "recovery_policy": {"retry_on_transient_error": True},
                "notes": "Baseline 3: LLM supervision at every stop point.",
            }
        ),
    }


def resolve_protocol(
    name_or_path: str,
    *,
    search_dirs: Optional[List[Path]] = None,
) -> ExperimentProtocol:
    """Resolve a protocol by name (looking in directories/defaults) or file path."""
    candidate = Path(name_or_path)
    if candidate.is_file():
        return load_protocol(candidate)

    for d in search_dirs or []:
        for suffix in (".yaml", ".yml", ".json"):
            p = Path(d) / f"{name_or_path}{suffix}"
            if p.is_file():
                return load_protocol(p)

    defaults = default_protocols()
    if name_or_path in defaults:
        return defaults[name_or_path]

    raise ProtocolValidationError(
        f"could not resolve protocol {name_or_path!r} in "
        f"{[str(d) for d in (search_dirs or [])]} or defaults"
    )
