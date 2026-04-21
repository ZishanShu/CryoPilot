"""Case ``available_inputs`` -> backend-contract input-field mapping.

The case registry uses semantic / experiment-layer field names (e.g.
``micrographs_star``) while backend preflight + executors consume the input
contract defined in :mod:`agent.backend_registry` (e.g.
``micrographs_star_path``).  This tiny helper bridges the two without changing
any backend executor.

Rules:

* Same-name contract fields present in ``available_inputs`` pass through.
* Semantic aliases (see ``SEMANTIC_TO_CONTRACT``) are translated.
* When both are present, the contract-name wins (no silent alias-override).
* Values that are ``None`` / ``False`` / empty string are dropped — they are
  flags-of-absence at the case layer and must not appear as backend inputs.
* Unknown keys are ignored; we never leak unrelated case metadata into the
  backend params.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

SEMANTIC_TO_CONTRACT: Dict[str, Dict[str, str]] = {
    "topaz": {
        "micrographs_star": "micrographs_star_path",
        "ctf_star": "ctf_micrographs_star",
    },
    "cryodrgn": {
        "particles_star": "particles_star_path",
        "pose_pkl": "pose_pkl_path",
        "ctf_pkl": "ctf_pkl_path",
        "particles_mrcs": "particles_mrcs_path",
    },
    "modelangelo": {
        "map": "map_path",
        "sequence": "sequence_path",
    },
    "emodelx": {
        "map": "map_path",
        "sequence": "sequence_path",
        "repo_dir": "emodelx_repo_dir",
    },
    "mica": {
        "map": "map_path",
        "sequence": "sequence_path",
        "repo_dir": "mica_repo_dir",
    },
}


def _contract_fields(backend: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Return (required_fields, optional_fields) from the backend registry."""
    from agent.backend_registry import (
        HETEROGENEITY_INPUT_SPECS,
        MODEL_BUILDING_INPUT_SPECS,
        PICKING_INPUT_SPECS,
    )

    for table in (
        PICKING_INPUT_SPECS,
        HETEROGENEITY_INPUT_SPECS,
        MODEL_BUILDING_INPUT_SPECS,
    ):
        spec = table.get(backend)
        if spec:
            return (
                tuple(spec.get("required_fields", ()) or ()),
                tuple(spec.get("optional_fields", ()) or ()),
            )
    return (), ()


def _is_usable_value(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, bool):
        return False
    if isinstance(v, str) and not v.strip():
        return False
    return True


def map_case_inputs_to_backend_params(
    *,
    available_inputs: Mapping[str, Any],
    backend: str,
) -> Dict[str, Any]:
    """Translate ``case.available_inputs`` into backend-contract input params.

    See module docstring for the full rule set.
    """
    required, optional = _contract_fields(backend)
    valid = set(required) | set(optional)
    aliases = SEMANTIC_TO_CONTRACT.get(backend, {})

    out: Dict[str, Any] = {}

    for semantic_key, contract_key in aliases.items():
        if contract_key not in valid:
            continue
        val = available_inputs.get(semantic_key)
        if _is_usable_value(val):
            out[contract_key] = val

    for key, value in available_inputs.items():
        if key in valid and _is_usable_value(value):
            out[key] = value

    return out


def supported_backends() -> Tuple[str, ...]:
    """List of backends that have mapping rules registered."""
    return tuple(SEMANTIC_TO_CONTRACT.keys())
