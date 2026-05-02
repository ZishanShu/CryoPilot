"""Stage 11 – Model validation gate.

Plans validation of a built atomic model.  Does not fake metrics.  When no
model file exists, the gate records ``SKIP_MODEL_VALIDATION_NO_MODEL``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def stage11_model_validation_gate(
    out_dir: Path,
    *,
    model_path: Optional[Path],
    map_path: Optional[Path],
    sequence: Optional[Path],
) -> Dict[str, Any]:
    model_exists = bool(model_path and Path(model_path).exists())

    if not model_exists:
        result: Dict[str, Any] = {
            "stage": "11_model_validation",
            "status": "skipped",
            "decision": "SKIP_MODEL_VALIDATION_NO_MODEL",
            "model_path": str(model_path) if model_path else None,
            "model_exists": False,
            "metrics": None,
        }
        _write_json(out_dir / "model_validation.json", result)
        _write_text(out_dir / "model_validation.md",
                    "# Stage 11 – Model validation\n\n"
                    "Skipped: no model produced upstream.\n"
                    f"\n- decision: {result['decision']}\n")
        return result

    planned_metrics = {
        "model_file_exists": True,
        "sequence_coverage": "planned" if sequence else "unavailable",
        "map_model_correlation": "planned" if map_path else "unavailable",
        "geometry_validation": "planned",
    }
    result = {
        "stage": "11_model_validation",
        "status": "planned",
        "decision": "PLAN_MODEL_VALIDATION",
        "model_path": str(model_path),
        "model_exists": True,
        "map_path": str(map_path) if map_path else None,
        "sequence": str(sequence) if sequence else None,
        "metrics": planned_metrics,
        "note": "Validation metrics are planned only; not computed.",
    }
    _write_json(out_dir / "model_validation.json", result)
    md = ["# Stage 11 – Model validation", "", "| key | value |", "|---|---|"]
    for k in ("status", "decision", "model_path", "map_path", "sequence"):
        md.append(f"| {k} | {result.get(k)} |")
    md.append("")
    md.append("## Planned metrics")
    md.append("")
    for k, v in planned_metrics.items():
        md.append(f"- {k}: {v}")
    _write_text(out_dir / "model_validation.md", "\n".join(md) + "\n")
    return result
