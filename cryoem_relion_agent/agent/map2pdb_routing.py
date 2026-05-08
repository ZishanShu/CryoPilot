"""Stage 10 – Map-to-PDB (model-building) routing gate.

Resolves a candidate model-building backend (ModelAngelo / EModelX / MICA)
based on a map quality gate decision and per-backend availability checks.

This gate never executes a backend.  When ``mode == 'real'`` and all
preconditions hold, it records the selected backend and writes a
``real_run_allowed = True`` decision; the actual backend invocation is
handled elsewhere (or kept disabled in this pipeline).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


VALID_MODES = ("none", "plan-only", "dry-run", "real")
DEFAULT_BACKENDS = ("modelangelo", "emodelx", "mica")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _normalise_mode(mode: str) -> str:
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode {mode!r}; expected one of {VALID_MODES}")
    return mode


def _backend_available(name: str) -> bool:
    """Best-effort availability check for a model-building backend."""
    name = name.lower()
    if name == "modelangelo":
        return shutil.which("model_angelo") is not None
    if name == "emodelx":
        return shutil.which("emodelx") is not None or shutil.which("EModelX") is not None
    if name == "mica":
        return shutil.which("mica") is not None
    return False


def _select_backend(
    candidates: Iterable[str],
    *,
    sequence_provided: bool,
    availability: Dict[str, bool],
) -> Optional[str]:
    cand_list = [c for c in candidates]
    if sequence_provided and "modelangelo" in cand_list and availability.get("modelangelo"):
        return "modelangelo"
    if (not sequence_provided) and "emodelx" in cand_list and availability.get("emodelx"):
        return "emodelx"
    for c in cand_list:
        if availability.get(c):
            return c
    return None


def stage10_map2pdb_routing(
    out_dir: Path,
    *,
    map_path: Optional[Path],
    map_exists: bool,
    map_source: str,
    map_quality_gate: Dict[str, Any],
    sequence: Optional[Path],
    backends: Optional[List[str]] = None,
    mode: str = "dry-run",
    require_fsc: bool = True,
    allow_without_fsc: bool = False,
    availability_overrides: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """Decide a map-to-model backend without executing it."""
    mode = _normalise_mode(mode)
    backends = list(backends) if backends else list(DEFAULT_BACKENDS)
    availability = {b: _backend_available(b) for b in backends}
    if availability_overrides:
        availability.update(availability_overrides)

    sequence_provided = bool(sequence and Path(sequence).exists())
    fsc_available = bool(map_quality_gate.get("fsc_available"))
    allow_real = bool(map_quality_gate.get("allow_map2pdb_real"))
    map_quality_status = map_quality_gate.get("map_quality_status", "unvalidated")

    routing: Dict[str, Any] = {
        "stage": "10_map2pdb_routing",
        "input_map": str(map_path) if map_path else None,
        "map_source": map_source,
        "map_quality_status": map_quality_status,
        "map2pdb_mode": mode,
        "candidate_backends": backends,
        "backend_availability": availability,
        "sequence_provided": sequence_provided,
        "sequence": str(sequence) if sequence else None,
        "selected_backend": None,
        "real_run_allowed": False,
        "decision": None,
        "reason": None,
    }

    if mode == "none":
        routing["decision"] = "SKIP_MAP2PDB"
        routing["reason"] = "map2pdb-mode=none"
        _persist(out_dir, routing)
        return routing

    if not map_exists or map_path is None:
        routing["decision"] = "BLOCK_MAP2PDB_NO_MAP"
        routing["reason"] = "No reconstruction or postprocess map available"
        _persist(out_dir, routing)
        return routing

    # Determine candidate backend.
    selected = _select_backend(
        backends, sequence_provided=sequence_provided, availability=availability,
    )
    routing["selected_backend"] = selected
    if selected is None:
        routing["decision"] = "BLOCK_MAP2PDB_NO_AVAILABLE_BACKEND"
        routing["reason"] = "No model-building backend executable found on PATH"
        _persist(out_dir, routing)
        return routing

    qc_required = require_fsc and not fsc_available
    if mode == "real":
        if qc_required and not allow_real:
            routing["decision"] = "STOP_MAP2PDB_QC_REQUIRED"
            routing["reason"] = "FSC/resolution/postprocess evidence missing"
            routing["real_run_allowed"] = False
        elif not allow_real:
            routing["decision"] = "STOP_MAP2PDB_QC_REQUIRED"
            routing["reason"] = "Map quality gate did not allow real map2pdb run"
            routing["real_run_allowed"] = False
        else:
            routing["decision"] = "ALLOW_MAP2PDB_REAL"
            routing["real_run_allowed"] = True
    else:  # plan-only / dry-run
        if qc_required and not allow_without_fsc:
            routing["decision"] = "STOP_MAP2PDB_QC_REQUIRED"
            routing["reason"] = "FSC/resolution/postprocess evidence missing"
        else:
            routing["decision"] = "ALLOW_MAP2PDB_DRY_RUN"
            routing["reason"] = "Routing recorded; no model produced"

    _persist(out_dir, routing)
    return routing


def _persist(out_dir: Path, routing: Dict[str, Any]) -> None:
    _write_json(out_dir / "map2pdb_routing.json", routing)
    md = ["# Stage 10 – Map-to-PDB routing", "", "| key | value |", "|---|---|"]
    for k in ("input_map", "map_source", "map_quality_status", "map2pdb_mode",
              "selected_backend", "real_run_allowed", "decision", "reason",
              "sequence_provided", "sequence"):
        md.append(f"| {k} | {routing.get(k)} |")
    md.append("")
    md.append("## Candidate backends")
    md.append("")
    md.append("| backend | available |")
    md.append("|---|---|")
    for b in routing.get("candidate_backends", []):
        md.append(f"| {b} | {routing.get('backend_availability', {}).get(b)} |")
    _write_text(out_dir / "map2pdb_routing.md", "\n".join(md) + "\n")
