"""LLM-supervised picker fallback / model-selection stage.

When a Topaz threshold sweep fails F1 QC, this stage retrieves alternative
picker candidates, detects their availability, picks one according to a fixed
priority order, and writes a closed-loop evidence bundle (retrieval, plan,
comparison, final selection, and a top-level report).

The stage is intentionally rule-based and stdlib-only.  No raw MRC/MRCS or GPU
is required: availability is inferred from filesystem presence of executables,
repos, and ground-truth coordinate paths supplied via ``BackendPaths``.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# Fixed retrieval order.  Anything new must be appended here AND given an
# availability check below.
CANDIDATE_ORDER: List[str] = [
    "relion_log_autopick",
    "cryotransformer_picker",
    "cryolo_picker",
    "topaz_supervised_retrain",
    "template_picker",
]

# Status values used in candidate dicts.
STATUS_AVAILABLE = "available"
STATUS_PARTIAL = "partial"
STATUS_UNAVAILABLE = "unavailable"

FINAL_ACTION_RUN = "RUN_ALTERNATIVE_PICKER"
FINAL_ACTION_PLAN_ONLY = "PLAN_ONLY_ALTERNATIVE_PICKER"
FINAL_ACTION_STOP = "STOP_PLAN_ONLY_NO_VIABLE_PICKER"


@dataclass
class BackendPaths:
    """Filesystem locations used for availability detection.

    All fields are optional; missing paths are treated as unavailable.
    """

    relion_autopick: Optional[Path] = None
    relion_autopick_mpi: Optional[Path] = None
    topaz: Optional[Path] = None
    cryolo_predict: Optional[Path] = None
    cryotransformer_repo: Optional[Path] = None
    template_picker_script: Optional[Path] = None
    gt_coords_dir: Optional[Path] = None  # for topaz supervised retrain

    def _which(self, name: str) -> Optional[Path]:
        found = shutil.which(name)
        return Path(found) if found else None


def _exists(p: Optional[Path]) -> bool:
    return p is not None and Path(p).exists()


# ---------------------------- availability detection ------------------------


def detect_relion_log_autopick(paths: BackendPaths) -> Dict[str, Any]:
    has_serial = _exists(paths.relion_autopick) or paths._which("relion_autopick") is not None
    has_mpi = (
        _exists(paths.relion_autopick_mpi)
        or paths._which("relion_autopick_mpi") is not None
    )
    available = has_serial or has_mpi
    return {
        "name": "relion_log_autopick",
        "status": STATUS_AVAILABLE if available else STATUS_UNAVAILABLE,
        "kind": "rule_based_log_picker",
        "evidence": {
            "relion_autopick_present": has_serial,
            "relion_autopick_mpi_present": has_mpi,
        },
        "notes": (
            "RELION LoG/autopick available; can produce real coordinates and "
            "be evaluated with the existing F1 evaluator."
            if available
            else "Neither relion_autopick nor relion_autopick_mpi found."
        ),
    }


def detect_cryotransformer(paths: BackendPaths) -> Dict[str, Any]:
    repo = paths.cryotransformer_repo
    repo_exists = _exists(repo)
    predict_py = (Path(repo) / "predict.py") if repo_exists else None
    has_predict = _exists(predict_py)
    if repo_exists and has_predict:
        status = STATUS_PARTIAL
        notes = (
            "Local CryoTransformer repo and predict.py present, but inference "
            "command is not verified.  Treat as plan-only until a successful "
            "smoke run is recorded."
        )
    else:
        status = STATUS_UNAVAILABLE
        notes = "CryoTransformer repo or predict.py missing."
    return {
        "name": "cryotransformer_picker",
        "status": status,
        "kind": "deep_learning_picker",
        "evidence": {
            "repo_path": str(repo) if repo else None,
            "repo_exists": repo_exists,
            "predict_py_present": has_predict,
        },
        "notes": notes,
    }


def detect_cryolo(paths: BackendPaths) -> Dict[str, Any]:
    predict = paths.cryolo_predict
    has_predict = _exists(predict) or paths._which("cryolo_predict.py") is not None
    return {
        "name": "cryolo_picker",
        "status": STATUS_AVAILABLE if has_predict else STATUS_UNAVAILABLE,
        "kind": "deep_learning_picker",
        "evidence": {"cryolo_predict_present": has_predict},
        "notes": (
            "crYOLO predict script present."
            if has_predict
            else "cryolo_predict.py missing; crYOLO is unavailable.  This is "
            "non-fatal: the stage will skip crYOLO and continue."
        ),
    }


def detect_topaz_retrain(paths: BackendPaths) -> Dict[str, Any]:
    has_topaz = _exists(paths.topaz) or paths._which("topaz") is not None
    has_gt = _exists(paths.gt_coords_dir)
    available = has_topaz and has_gt
    return {
        "name": "topaz_supervised_retrain",
        "status": STATUS_AVAILABLE if available else STATUS_UNAVAILABLE,
        "kind": "deep_learning_picker_retrain",
        "evidence": {
            "topaz_present": has_topaz,
            "gt_coords_dir_present": has_gt,
            "gt_coords_dir": str(paths.gt_coords_dir) if paths.gt_coords_dir else None,
        },
        "notes": (
            "topaz binary and GT coordinates available for supervised retraining."
            if available
            else "topaz binary or GT coordinate paths missing."
        ),
    }


def detect_template_picker(paths: BackendPaths) -> Dict[str, Any]:
    script = paths.template_picker_script
    has_script = _exists(script)
    return {
        "name": "template_picker",
        "status": STATUS_AVAILABLE if has_script else STATUS_UNAVAILABLE,
        "kind": "template_matching_picker",
        "evidence": {"template_picker_script_present": has_script},
        "notes": (
            "Template picker script present."
            if has_script
            else "Template picker script missing."
        ),
    }


_DETECTORS = {
    "relion_log_autopick": detect_relion_log_autopick,
    "cryotransformer_picker": detect_cryotransformer,
    "cryolo_picker": detect_cryolo,
    "topaz_supervised_retrain": detect_topaz_retrain,
    "template_picker": detect_template_picker,
}


def retrieve_alternative_picker_candidates(paths: BackendPaths) -> List[Dict[str, Any]]:
    """Return all candidates in fixed retrieval order, with availability filled in."""
    return [_DETECTORS[name](paths) for name in CANDIDATE_ORDER]


# ------------------------------ selection logic -----------------------------


def select_alternative_picker(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pick the first viable candidate per the documented order.

    Selection rules (per task spec):

    - relion_log_autopick if available -> RUN
    - cryotransformer_picker if available or partial -> RUN if available else PLAN_ONLY
    - cryolo_picker if available -> RUN
    - topaz_supervised_retrain if available -> RUN
    - template_picker if available -> RUN
    - otherwise STOP plan-only
    """
    by_name = {c["name"]: c for c in candidates}
    for name in CANDIDATE_ORDER:
        cand = by_name.get(name)
        if cand is None:
            continue
        status = cand.get("status")
        if name == "cryotransformer_picker":
            if status == STATUS_AVAILABLE:
                return _selection(cand, FINAL_ACTION_RUN)
            if status == STATUS_PARTIAL:
                return _selection(cand, FINAL_ACTION_PLAN_ONLY)
            continue
        if status == STATUS_AVAILABLE:
            return _selection(cand, FINAL_ACTION_RUN)
    return {
        "selected": None,
        "final_action": FINAL_ACTION_STOP,
        "reason": "No viable alternative picker is available; preserving STOP.",
    }


def _selection(cand: Dict[str, Any], action: str) -> Dict[str, Any]:
    return {
        "selected": cand["name"],
        "final_action": action,
        "reason": f"Selected {cand['name']} ({cand['status']}) per priority order.",
        "candidate": cand,
    }


# ------------------------------- output writing -----------------------------


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _candidates_md(candidates: List[Dict[str, Any]]) -> str:
    lines = ["# Alternative picker candidates", "", "| order | name | status | kind | notes |", "|---|---|---|---|---|"]
    for i, c in enumerate(candidates, 1):
        lines.append(
            f"| {i} | {c['name']} | {c['status']} | {c.get('kind','')} | {c.get('notes','').replace(chr(10), ' ')} |"
        )
    return "\n".join(lines) + "\n"


def _rerun_plan(selection: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "trigger": {
            "previous_backend": evidence.get("selected_for_backend_execution"),
            "selection_metric": evidence.get("selection_metric"),
            "selected_micro_f1": evidence.get("selected_micro_f1"),
            "qc_threshold_f1": evidence.get("qc_threshold_f1"),
            "picking_qc_passed": evidence.get("picking_qc_passed"),
        },
        "selection": selection,
        "executes": (
            selection.get("final_action") == FINAL_ACTION_RUN
            and selection.get("selected") is not None
        ),
        "plan_only": selection.get("final_action") in (FINAL_ACTION_PLAN_ONLY, FINAL_ACTION_STOP),
    }


def _rerun_plan_md(plan: Dict[str, Any]) -> str:
    sel = plan.get("selection", {})
    trig = plan.get("trigger", {})
    lines = [
        "# Alternative picker rerun plan",
        "",
        f"- previous_backend: {trig.get('previous_backend')}",
        f"- selection_metric: {trig.get('selection_metric')}",
        f"- selected_micro_f1: {trig.get('selected_micro_f1')}",
        f"- qc_threshold_f1: {trig.get('qc_threshold_f1')}",
        f"- picking_qc_passed: {trig.get('picking_qc_passed')}",
        "",
        f"- selected: {sel.get('selected')}",
        f"- final_action: {sel.get('final_action')}",
        f"- reason: {sel.get('reason')}",
        f"- executes: {plan.get('executes')}",
        f"- plan_only: {plan.get('plan_only')}",
    ]
    return "\n".join(lines) + "\n"


def _comparison_rows(
    evidence: Dict[str, Any], candidates: List[Dict[str, Any]], selection: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """One row for the failed Topaz baseline plus one row per candidate."""
    selected_name = selection.get("selected")
    rows: List[Dict[str, Any]] = []
    rows.append(
        {
            "model": evidence.get("selected_for_backend_execution") or "topaz_threshold_sweep",
            "kind": "topaz_threshold_sweep",
            "status": "failed_qc",
            "best_f1": evidence.get("selected_micro_f1"),
            "qc_threshold_f1": evidence.get("qc_threshold_f1"),
            "selected": False,
            "notes": "Baseline that failed picking F1 QC.",
        }
    )
    for c in candidates:
        rows.append(
            {
                "model": c["name"],
                "kind": c.get("kind", ""),
                "status": c["status"],
                "best_f1": None,
                "qc_threshold_f1": evidence.get("qc_threshold_f1"),
                "selected": c["name"] == selected_name,
                "notes": c.get("notes", ""),
            }
        )
    return rows


def _comparison_md(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "# Picker model selection comparison",
        "",
        "| model | kind | status | best_f1 | qc_threshold_f1 | selected |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['kind']} | {r['status']} | {r['best_f1']} | "
            f"{r['qc_threshold_f1']} | {r['selected']} |"
        )
    return "\n".join(lines) + "\n"


def _final_selection_md(final: Dict[str, Any]) -> str:
    lines = [
        "# Final picker selection",
        "",
        f"- selected: {final.get('selected')}",
        f"- final_action: {final.get('final_action')}",
        f"- reason: {final.get('reason')}",
        f"- previous_backend: {final.get('previous_backend')}",
        f"- previous_micro_f1: {final.get('previous_micro_f1')}",
        f"- qc_threshold_f1: {final.get('qc_threshold_f1')}",
        "",
        "Note: this stage does not claim full autonomous structure determination.",
    ]
    return "\n".join(lines) + "\n"


def _report_md(
    evidence: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    selection: Dict[str, Any],
    final: Dict[str, Any],
) -> str:
    lines = [
        "# Picker model selection report",
        "",
        "## Trigger",
        f"- previous_backend: {evidence.get('selected_for_backend_execution')}",
        f"- selected_micro_f1: {evidence.get('selected_micro_f1')}",
        f"- qc_threshold_f1: {evidence.get('qc_threshold_f1')}",
        f"- picking_qc_passed: {evidence.get('picking_qc_passed')}",
        "",
        "## Retrieved candidates",
        _candidates_md(candidates),
        "## Selection",
        f"- selected: {selection.get('selected')}",
        f"- final_action: {selection.get('final_action')}",
        f"- reason: {selection.get('reason')}",
        "",
        "## Notes",
        "- Rule-based baseline behavior is unchanged by this stage.",
        "- crYOLO unavailability is non-fatal.",
        "- CryoTransformer is plan-only unless an inference smoke run is verified.",
        "- This stage does not claim full autonomous structure determination.",
    ]
    return "\n".join(lines) + "\n"


# ------------------------------- top-level API ------------------------------


def run_picker_fallback_stage(
    evidence: Dict[str, Any],
    paths: BackendPaths,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run the picker fallback / model-selection stage and persist all artifacts.

    Returns the in-memory bundle (also persisted to ``output_dir``).
    """
    if evidence.get("picking_qc_passed", True):
        bundle = {
            "skipped": True,
            "reason": "picking_qc_passed=true; fallback stage not triggered.",
            "evidence": evidence,
        }
        _write_json(Path(output_dir) / "picker_model_selection_skipped.json", bundle)
        return bundle

    candidates = retrieve_alternative_picker_candidates(paths)
    selection = select_alternative_picker(candidates)
    plan = _rerun_plan(selection, evidence)
    rows = _comparison_rows(evidence, candidates, selection)
    final = {
        "selected": selection.get("selected"),
        "final_action": selection.get("final_action"),
        "reason": selection.get("reason"),
        "previous_backend": evidence.get("selected_for_backend_execution"),
        "previous_micro_f1": evidence.get("selected_micro_f1"),
        "qc_threshold_f1": evidence.get("qc_threshold_f1"),
    }

    out = Path(output_dir)
    _write_json(out / "retrieval" / "alternative_picker_candidates.json", candidates)
    _write_text(out / "retrieval" / "alternative_picker_candidates.md", _candidates_md(candidates))
    _write_json(out / "selection" / "alternative_picker_rerun_plan.json", plan)
    _write_text(out / "selection" / "alternative_picker_rerun_plan.md", _rerun_plan_md(plan))

    csv_path = out / "comparison" / "picker_model_selection_comparison.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "kind", "status", "best_f1", "qc_threshold_f1", "selected", "notes"]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    _write_text(out / "comparison" / "picker_model_selection_comparison.md", _comparison_md(rows))
    _write_json(out / "comparison" / "final_picker_selection.json", final)
    _write_text(out / "comparison" / "final_picker_selection.md", _final_selection_md(final))
    _write_text(
        out / "picker_model_selection_report.md",
        _report_md(evidence, candidates, selection, final),
    )

    return {
        "candidates": candidates,
        "selection": selection,
        "plan": plan,
        "comparison_rows": rows,
        "final": final,
        "output_dir": str(out),
    }
