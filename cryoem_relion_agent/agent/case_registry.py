"""Case registry for paper-grade experiment runs.

A *case* is a concrete dataset + input state the experiment runner is asked to
execute.  Two kinds are supported in this first version:

- workflow_case: the full pick -> class2D -> ... pipeline domain
- map_level_case: starts from a map + sequence (model_building / routing)

Cases do NOT describe protocol behavior. The runner combines ``ExperimentCase``
with ``ExperimentProtocol`` to produce a result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

SUPPORTED_CASE_TYPES: Tuple[str, ...] = (
    "workflow_case",
    "map_level_case",
)

SUPPORTED_TARGET_STAGES: Tuple[str, ...] = (
    "picking",
    "extraction",
    "class2d",
    "heterogeneity",
    "postprocess",
    "model_building",
)

SUPPORTED_EVALUATION_MODES: Tuple[str, ...] = (
    "none",
    "picking_coords",
    "model_building_metrics",
    "success_only",
)


@dataclass
class ExperimentCase:
    """Normalized case entry used by the baseline runner."""

    case_id: str
    case_type: str
    dataset_name: str
    dataset_root: Optional[str] = None
    input_state: Dict[str, Any] = field(default_factory=dict)
    available_inputs: Dict[str, Any] = field(default_factory=dict)
    target_stage: str = "picking"
    evaluation_mode: str = "none"
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CaseValidationError(ValueError):
    """Raised when a case config is malformed."""


def _require_str(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CaseValidationError(f"{key} is required and must be a non-empty string")
    return value.strip()


def parse_case(raw: Mapping[str, Any]) -> ExperimentCase:
    """Parse a raw mapping into a validated ``ExperimentCase``."""
    if not isinstance(raw, Mapping):
        raise CaseValidationError("case must be a mapping")

    case_id = _require_str(raw, "case_id")
    case_type = _require_str(raw, "case_type")
    if case_type not in SUPPORTED_CASE_TYPES:
        raise CaseValidationError(
            f"case_type must be one of {SUPPORTED_CASE_TYPES}, got {case_type!r}"
        )
    dataset_name = _require_str(raw, "dataset_name")

    dataset_root = raw.get("dataset_root")
    if dataset_root is not None and not isinstance(dataset_root, str):
        raise CaseValidationError("dataset_root must be a string if provided")

    input_state = raw.get("input_state") or {}
    if not isinstance(input_state, Mapping):
        raise CaseValidationError("input_state must be a mapping")

    available_inputs = raw.get("available_inputs") or {}
    if not isinstance(available_inputs, Mapping):
        raise CaseValidationError("available_inputs must be a mapping")

    target_stage = str(raw.get("target_stage") or "").strip() or (
        "model_building" if case_type == "map_level_case" else "picking"
    )
    if target_stage not in SUPPORTED_TARGET_STAGES:
        raise CaseValidationError(
            f"target_stage must be one of {SUPPORTED_TARGET_STAGES}, got {target_stage!r}"
        )

    evaluation_mode = str(raw.get("evaluation_mode") or "").strip() or "none"
    if evaluation_mode not in SUPPORTED_EVALUATION_MODES:
        raise CaseValidationError(
            f"evaluation_mode must be one of {SUPPORTED_EVALUATION_MODES}"
        )

    notes = raw.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise CaseValidationError("notes must be a string if provided")

    if case_type == "map_level_case" and target_stage not in {
        "model_building",
        "postprocess",
    }:
        raise CaseValidationError(
            "map_level_case requires target_stage in {model_building, postprocess}"
        )

    return ExperimentCase(
        case_id=case_id,
        case_type=case_type,
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        input_state=dict(input_state),
        available_inputs=dict(available_inputs),
        target_stage=target_stage,
        evaluation_mode=evaluation_mode,
        notes=notes,
    )


def load_case(path: Path) -> ExperimentCase:
    """Load a case from YAML or JSON file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"case file not found: {p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif p.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, Mapping):
        raise CaseValidationError(f"{p} must deserialize to a mapping")
    return parse_case(data)


def load_cases_from_dir(dir_path: Path) -> Dict[str, ExperimentCase]:
    """Load every *.yaml / *.yml / *.json file in a directory as a case."""
    d = Path(dir_path)
    out: Dict[str, ExperimentCase] = {}
    if not d.is_dir():
        return out
    for child in sorted(d.iterdir()):
        if not child.is_file():
            continue
        if child.suffix.lower() not in {".yaml", ".yml", ".json"}:
            continue
        case = load_case(child)
        if case.case_id in out:
            raise CaseValidationError(
                f"duplicate case_id {case.case_id!r} in {d}"
            )
        out[case.case_id] = case
    return out


def default_cases() -> Dict[str, ExperimentCase]:
    """Minimal set of built-in cases for protocol tests and smoke runs."""
    return {
        "cryoppp_lite_topaz_smoke": parse_case(
            {
                "case_id": "cryoppp_lite_topaz_smoke",
                "case_type": "workflow_case",
                "dataset_name": "cryoppp_lite",
                "dataset_root": None,
                "input_state": {"stage": "picking"},
                "available_inputs": {"micrographs_star": False},
                "target_stage": "picking",
                "evaluation_mode": "picking_coords",
                "notes": "Success-or-failure registry entry for CryoPPP_Lite Topaz path.",
            }
        ),
        "mid3d_map_only_model_building": parse_case(
            {
                "case_id": "mid3d_map_only_model_building",
                "case_type": "map_level_case",
                "dataset_name": "real_mid3d_project",
                "dataset_root": None,
                "input_state": {"has_map": True, "has_sequence": True},
                "available_inputs": {"map_path": True, "sequence_path": True},
                "target_stage": "model_building",
                "evaluation_mode": "success_only",
                "notes": "Map-level routing registry entry (ModelAngelo / EModelX / MICA).",
            }
        ),
    }


def resolve_case(
    name_or_path: str,
    *,
    search_dirs: Optional[List[Path]] = None,
) -> ExperimentCase:
    """Resolve a case by case_id (directories/defaults) or file path."""
    candidate = Path(name_or_path)
    if candidate.is_file():
        return load_case(candidate)

    for d in search_dirs or []:
        for suffix in (".yaml", ".yml", ".json"):
            p = Path(d) / f"{name_or_path}{suffix}"
            if p.is_file():
                return load_case(p)

    defaults = default_cases()
    if name_or_path in defaults:
        return defaults[name_or_path]

    raise CaseValidationError(
        f"could not resolve case {name_or_path!r} in "
        f"{[str(d) for d in (search_dirs or [])]} or defaults"
    )
