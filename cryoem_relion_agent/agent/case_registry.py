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
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

SUPPORTED_CASE_TYPES: Tuple[str, ...] = (
    "workflow_case",
    "map_level_case",
)

SUPPORTED_VARIABLES: Tuple[str, ...] = (
    "PROJECT_ROOT",
    "DATA_ROOT",
    "RUN_ROOT",
    "CASE_ROOT",
)

_VAR_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")

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
    """Minimal set of built-in cases for protocol tests and smoke runs.

    Includes the four standard failure-set cases (F1..F4).  Values stay
    independent of machine paths; use ``materialize_case`` with variables
    to bind them to a concrete environment.
    """
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
        "F1_cryoppp_lite_topaz_input_format_failure": parse_case(
            {
                "case_id": "F1_cryoppp_lite_topaz_input_format_failure",
                "case_type": "workflow_case",
                "dataset_name": "cryoppp_lite",
                "dataset_root": None,
                "input_state": {
                    "stage": "picking",
                    "expected_failure_phase": "execution",
                    "expected_failure_kind": "backend_runtime_error",
                    "expected_root_cause_guess": "input_format_incompatibility",
                },
                "available_inputs": {"micrographs_star": True},
                "target_stage": "picking",
                "evaluation_mode": "picking_coords",
                "notes": "F1 real failure sample: Topaz runtime input-format incompatibility.",
            }
        ),
        "F2_topaz_preflight_blocked_missing_executable": parse_case(
            {
                "case_id": "F2_topaz_preflight_blocked_missing_executable",
                "case_type": "workflow_case",
                "dataset_name": "cryoppp_lite",
                "dataset_root": None,
                "input_state": {
                    "stage": "picking",
                    "expected_failure_phase": "preflight",
                    "expected_failure_kind": "preflight_blocked",
                    "expected_root_cause_guess": "missing_model_or_dependency",
                },
                "available_inputs": {
                    "micrographs_star": True,
                    "topaz_executable_present": False,
                },
                "target_stage": "picking",
                "evaluation_mode": "none",
                "notes": "F2 preflight-blocked Topaz: executable missing.",
            }
        ),
        "F3_topaz_empty_output_case": parse_case(
            {
                "case_id": "F3_topaz_empty_output_case",
                "case_type": "workflow_case",
                "dataset_name": "cryoppp_lite",
                "dataset_root": None,
                "input_state": {
                    "stage": "picking",
                    "expected_failure_phase": "execution",
                    "expected_failure_kind": "empty_output_failure",
                    "expected_root_cause_guess": "unknown",
                },
                "available_inputs": {
                    "micrographs_star": True,
                    "synthetic_empty_output": True,
                },
                "target_stage": "picking",
                "evaluation_mode": "picking_coords",
                "notes": "F3 empty-output execution: rc=0 but picks.star is header-only.",
            }
        ),
        "T1_tutorial5_map_emodelx_stopA": parse_case(
            {
                "case_id": "T1_tutorial5_map_emodelx_stopA",
                "case_type": "map_level_case",
                "dataset_name": "tutorial5",
                "dataset_root": None,
                "input_state": {
                    "has_map": True,
                    "has_sequence": False,
                    "emodelx_protocol": "seq_free",
                },
                "available_inputs": {
                    "map_path": True,
                    "emodelx_repo_dir": True,
                    "output_dir": True,
                    "emodelx_protocol": "seq_free",
                },
                "target_stage": "model_building",
                "evaluation_mode": "success_only",
                "notes": "Tutorial5 map routed to EModelX seq_free (no sequence).",
            }
        ),
        "F4_mica_unavailable_missing_af3_dir": parse_case(
            {
                "case_id": "F4_mica_unavailable_missing_af3_dir",
                "case_type": "map_level_case",
                "dataset_name": "real_mid3d_project",
                "dataset_root": None,
                "input_state": {
                    "has_map": True,
                    "has_sequence": True,
                    "expected_failure_phase": "preflight",
                    "expected_failure_kind": "preflight_blocked",
                    "expected_root_cause_guess": "missing_model_or_dependency",
                    "af3_dir_present": False,
                },
                "available_inputs": {"map_path": True, "sequence_path": True, "af3_dir": False},
                "target_stage": "model_building",
                "evaluation_mode": "success_only",
                "notes": "F4 MICA unavailable: missing af3_dir; advice should route to EModelX/ModelAngelo.",
            }
        ),
    }


class CaseMaterializationError(ValueError):
    """Raised when variable expansion for a case fails."""


def _expand_string(
    value: str,
    variables: Mapping[str, str],
    *,
    field_path: str,
) -> str:
    """Replace ``${VAR}`` markers in ``value`` using ``variables``.

    Raises ``CaseMaterializationError`` when a referenced variable is missing;
    we never silently leave placeholders in materialized cases.
    """
    missing: List[str] = []

    def _sub(match: "re.Match[str]") -> str:
        name = match.group(1)
        if name not in variables or variables[name] is None:
            missing.append(name)
            return match.group(0)
        return str(variables[name])

    expanded = _VAR_PATTERN.sub(_sub, value)
    if missing:
        raise CaseMaterializationError(
            f"missing variable(s) {sorted(set(missing))} required to expand "
            f"{field_path}={value!r}"
        )
    return expanded


def _expand_value(
    value: Any,
    variables: Mapping[str, str],
    *,
    field_path: str,
) -> Any:
    if isinstance(value, str):
        if "${" not in value:
            return value
        return _expand_string(value, variables, field_path=field_path)
    if isinstance(value, Mapping):
        return {
            k: _expand_value(v, variables, field_path=f"{field_path}.{k}")
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [
            _expand_value(v, variables, field_path=f"{field_path}[{i}]")
            for i, v in enumerate(value)
        ]
    return value


def _normalized_variables(
    variables: Optional[Mapping[str, Any]],
    *,
    include_env: bool = False,
) -> Dict[str, str]:
    """Return a clean ``{name: str_value}`` mapping restricted to supported vars.

    Unsupported keys are tolerated (ignored) to allow callers to pass a
    superset, but the returned mapping only contains non-empty string values.
    """
    out: Dict[str, str] = {}
    if include_env:
        for name in SUPPORTED_VARIABLES:
            env_val = os.environ.get(name)
            if env_val:
                out[name] = str(env_val)
    for name, value in (variables or {}).items():
        if value is None:
            continue
        if not isinstance(name, str):
            continue
        sval = str(value).strip()
        if not sval:
            continue
        out[name] = sval
    return out


def materialize_case(
    case: ExperimentCase,
    variables: Optional[Mapping[str, Any]] = None,
    *,
    include_env: bool = False,
) -> ExperimentCase:
    """Return a new ``ExperimentCase`` with ``${VAR}`` placeholders expanded.

    Expansion covers ``dataset_root``, ``input_state``, ``available_inputs``,
    and ``notes``.  ``case_id``, ``case_type``, ``dataset_name``,
    ``target_stage``, ``evaluation_mode`` are left untouched.
    """
    vars_map = _normalized_variables(variables, include_env=include_env)

    dataset_root = case.dataset_root
    if dataset_root is not None:
        dataset_root = _expand_string(
            dataset_root, vars_map, field_path="dataset_root"
        )

    input_state = _expand_value(
        dict(case.input_state or {}),
        vars_map,
        field_path="input_state",
    )
    available_inputs = _expand_value(
        dict(case.available_inputs or {}),
        vars_map,
        field_path="available_inputs",
    )

    notes = case.notes
    if isinstance(notes, str) and "${" in notes:
        notes = _expand_string(notes, vars_map, field_path="notes")

    return ExperimentCase(
        case_id=case.case_id,
        case_type=case.case_type,
        dataset_name=case.dataset_name,
        dataset_root=dataset_root,
        input_state=input_state,
        available_inputs=available_inputs,
        target_stage=case.target_stage,
        evaluation_mode=case.evaluation_mode,
        notes=notes,
    )


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
