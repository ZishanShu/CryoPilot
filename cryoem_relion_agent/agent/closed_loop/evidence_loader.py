"""Load baseline evidence from a same-dataset workflow run root."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EvidenceCase:
    case_id: str
    dataset_id: str
    run_root: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    stage_status: Dict[str, str] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    rule_decisions_by_stage: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    llm_assessments_by_stage: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    llm_overall: Dict[str, Any] = field(default_factory=dict)
    available_large_artifact_manifest: Dict[str, Any] = field(default_factory=dict)
    logs: Dict[str, str] = field(default_factory=dict)
    workflow_summary_raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


CANONICAL_STAGES = ("particle_picking", "particle_extraction", "reconstruction", "map2pdb")

_RULE_TO_CANON = {
    "particle_picking": "particle_picking",
    "picking": "particle_picking",
    "particle_extraction": "particle_extraction",
    "extraction": "particle_extraction",
    "reconstruction": "reconstruction",
    "map2pdb": "map2pdb",
}


def _read_json(p: Path) -> Optional[Dict[str, Any]]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
    if not p.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def load_evidence_case(baseline_root: str | Path, case_id: Optional[str] = None) -> EvidenceCase:
    root = Path(baseline_root)
    if not root.exists():
        raise FileNotFoundError(f"baseline root not found: {root}")

    workflow_summary = _read_json(root / "workflow_summary.json") or {}
    dataset_id = workflow_summary.get("dataset_id", "UNKNOWN")
    if case_id is None:
        case_id = workflow_summary.get("project_id", root.name)

    rule_rows = _read_jsonl(root / "rule_based" / "stage_decisions.jsonl")
    rule_by_stage: Dict[str, Dict[str, Any]] = {}
    for r in rule_rows:
        stage = _RULE_TO_CANON.get(r.get("stage", ""), r.get("stage", ""))
        if stage:
            rule_by_stage[stage] = r

    llm_stagewise = _read_json(root / "llm_supervised" / "llm_stagewise_result.json") or {}
    llm_by_stage: Dict[str, Dict[str, Any]] = {}
    for a in llm_stagewise.get("stage_assessments", []) or []:
        s = _RULE_TO_CANON.get(a.get("stage", ""), a.get("stage", ""))
        if s:
            llm_by_stage[s] = a
    llm_overall = llm_stagewise.get("overall_assessment", {}) or {}

    if not llm_overall:
        # Fallback: parse llm_supervised_result.json (raw_text JSON).
        sup = _read_json(root / "llm_supervised" / "llm_supervised_result.json") or {}
        sup_inline = workflow_summary.get("llm_supervised", {})
        raw = (sup or {}).get("raw_text") or sup_inline.get("raw_text")
        if raw:
            try:
                parsed = json.loads(raw)
                llm_overall = {
                    "accept_backend_execution": parsed.get("accept_backend_execution"),
                    "accept_for_scientific_claim": parsed.get("accept_for_downstream"),
                    "risk_level": parsed.get("risk_level"),
                    "summary": parsed.get("main_issue"),
                    "recommended_next_action": parsed.get("recommended_next_action"),
                }
            except Exception:
                pass

    artifact_manifest = _read_json(root / "artifact_manifest.json") or {}

    return EvidenceCase(
        case_id=case_id,
        dataset_id=dataset_id,
        run_root=str(root),
        metrics=workflow_summary.get("metrics", {}) or {},
        stage_status=workflow_summary.get("stage_status", {}) or {},
        artifacts=workflow_summary.get("artifacts", {}) or {},
        rule_decisions_by_stage=rule_by_stage,
        llm_assessments_by_stage=llm_by_stage,
        llm_overall=llm_overall,
        available_large_artifact_manifest=artifact_manifest,
        logs=workflow_summary.get("logs", {}) or {},
        workflow_summary_raw=workflow_summary,
    )
