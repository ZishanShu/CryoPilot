"""Tests for the LLM-supervised picker fallback / model-selection stage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.picker_fallback import (
    BackendPaths,
    CANDIDATE_ORDER,
    FINAL_ACTION_PLAN_ONLY,
    FINAL_ACTION_RUN,
    FINAL_ACTION_STOP,
    STATUS_AVAILABLE,
    STATUS_PARTIAL,
    STATUS_UNAVAILABLE,
    detect_cryolo,
    detect_cryotransformer,
    detect_relion_log_autopick,
    detect_template_picker,
    detect_topaz_retrain,
    retrieve_alternative_picker_candidates,
    run_picker_fallback_stage,
    select_alternative_picker,
)


FAILED_TOPAZ_EVIDENCE = {
    "selected_for_backend_execution": "topaz_t-4",
    "selection_metric": "picking_micro_f1",
    "selected_micro_f1": 0.031848,
    "qc_threshold_f1": 0.2,
    "picking_qc_passed": False,
    "final_action": "FALLBACK_TO_ALTERNATIVE_PICKER_OR_RETRAIN",
}


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("", encoding="utf-8")
    return p


def _make_paths(
    tmp_path: Path,
    *,
    relion: bool = False,
    topaz: bool = False,
    gt: bool = False,
    cryolo: bool = False,
    cryotransformer: str = "missing",  # missing | partial
    template: bool = False,
) -> BackendPaths:
    paths = BackendPaths()
    if relion:
        paths.relion_autopick = _touch(tmp_path / "bin" / "relion_autopick")
    if topaz:
        paths.topaz = _touch(tmp_path / "bin" / "topaz")
    if gt:
        gt_dir = tmp_path / "gt_coords"
        gt_dir.mkdir(parents=True, exist_ok=True)
        paths.gt_coords_dir = gt_dir
    if cryolo:
        paths.cryolo_predict = _touch(tmp_path / "bin" / "cryolo_predict.py")
    if cryotransformer == "partial":
        repo = tmp_path / "cryotransformer"
        repo.mkdir(parents=True, exist_ok=True)
        _touch(repo / "predict.py")
        _touch(repo / "train.py")
        _touch(repo / "README.md")
        paths.cryotransformer_repo = repo
    if template:
        paths.template_picker_script = _touch(tmp_path / "bin" / "template_picker.py")
    return paths


# ----------------------------- retrieval / order ----------------------------


def test_retrieval_returns_all_candidates_in_fixed_order(tmp_path):
    cands = retrieve_alternative_picker_candidates(_make_paths(tmp_path))
    assert [c["name"] for c in cands] == CANDIDATE_ORDER


# ----------------------------- availability checks --------------------------


def test_relion_log_autopick_available_when_binary_present(tmp_path):
    p = _make_paths(tmp_path, relion=True)
    assert detect_relion_log_autopick(p)["status"] == STATUS_AVAILABLE


def test_relion_log_autopick_unavailable_when_missing(tmp_path):
    assert detect_relion_log_autopick(_make_paths(tmp_path))["status"] == STATUS_UNAVAILABLE


def test_cryotransformer_partial_when_repo_and_predict_present(tmp_path):
    p = _make_paths(tmp_path, cryotransformer="partial")
    d = detect_cryotransformer(p)
    assert d["status"] == STATUS_PARTIAL
    assert d["evidence"]["predict_py_present"] is True


def test_cryotransformer_unavailable_when_repo_missing(tmp_path):
    assert detect_cryotransformer(_make_paths(tmp_path))["status"] == STATUS_UNAVAILABLE


def test_cryolo_unavailable_when_predict_missing_and_does_not_crash(tmp_path):
    d = detect_cryolo(_make_paths(tmp_path))
    assert d["status"] == STATUS_UNAVAILABLE
    # Missing crYOLO must never raise; the whole retrieval still succeeds.
    cands = retrieve_alternative_picker_candidates(_make_paths(tmp_path))
    by_name = {c["name"]: c for c in cands}
    assert by_name["cryolo_picker"]["status"] == STATUS_UNAVAILABLE


def test_topaz_supervised_retrain_requires_topaz_and_gt(tmp_path):
    assert detect_topaz_retrain(_make_paths(tmp_path, topaz=True))["status"] == STATUS_UNAVAILABLE
    assert detect_topaz_retrain(_make_paths(tmp_path, gt=True))["status"] == STATUS_UNAVAILABLE
    assert (
        detect_topaz_retrain(_make_paths(tmp_path, topaz=True, gt=True))["status"]
        == STATUS_AVAILABLE
    )


def test_template_picker_availability(tmp_path):
    assert detect_template_picker(_make_paths(tmp_path))["status"] == STATUS_UNAVAILABLE
    assert (
        detect_template_picker(_make_paths(tmp_path, template=True))["status"] == STATUS_AVAILABLE
    )


# ----------------------------- selection order ------------------------------


def test_selection_prefers_relion_when_available(tmp_path):
    paths = _make_paths(
        tmp_path, relion=True, topaz=True, gt=True, cryotransformer="partial", template=True
    )
    sel = select_alternative_picker(retrieve_alternative_picker_candidates(paths))
    assert sel["selected"] == "relion_log_autopick"
    assert sel["final_action"] == FINAL_ACTION_RUN


def test_selection_falls_to_cryotransformer_partial_as_plan_only(tmp_path):
    paths = _make_paths(tmp_path, cryotransformer="partial", topaz=True, gt=True, template=True)
    sel = select_alternative_picker(retrieve_alternative_picker_candidates(paths))
    assert sel["selected"] == "cryotransformer_picker"
    assert sel["final_action"] == FINAL_ACTION_PLAN_ONLY


def test_selection_falls_to_topaz_retrain_when_no_higher_priority(tmp_path):
    paths = _make_paths(tmp_path, topaz=True, gt=True, template=True)
    sel = select_alternative_picker(retrieve_alternative_picker_candidates(paths))
    assert sel["selected"] == "topaz_supervised_retrain"
    assert sel["final_action"] == FINAL_ACTION_RUN


def test_selection_falls_to_template_when_only_template(tmp_path):
    paths = _make_paths(tmp_path, template=True)
    sel = select_alternative_picker(retrieve_alternative_picker_candidates(paths))
    assert sel["selected"] == "template_picker"
    assert sel["final_action"] == FINAL_ACTION_RUN


def test_selection_stops_when_nothing_viable(tmp_path):
    sel = select_alternative_picker(retrieve_alternative_picker_candidates(_make_paths(tmp_path)))
    assert sel["selected"] is None
    assert sel["final_action"] == FINAL_ACTION_STOP


# ----------------------------- STOP preservation ----------------------------


def test_stop_preserved_when_f1_below_threshold_and_nothing_available(tmp_path):
    bundle = run_picker_fallback_stage(
        FAILED_TOPAZ_EVIDENCE, _make_paths(tmp_path), tmp_path / "out"
    )
    assert bundle["selection"]["final_action"] == FINAL_ACTION_STOP
    final = json.loads(
        (tmp_path / "out" / "comparison" / "final_picker_selection.json").read_text()
    )
    assert final["final_action"] == FINAL_ACTION_STOP
    assert final["previous_micro_f1"] == FAILED_TOPAZ_EVIDENCE["selected_micro_f1"]


def test_skips_when_qc_passed(tmp_path):
    ev = dict(FAILED_TOPAZ_EVIDENCE, picking_qc_passed=True)
    bundle = run_picker_fallback_stage(ev, _make_paths(tmp_path), tmp_path / "out")
    assert bundle.get("skipped") is True


# ----------------------------- output artifacts -----------------------------


def test_run_writes_all_required_artifacts_for_partial_cryotransformer(tmp_path):
    paths = _make_paths(tmp_path, cryotransformer="partial", topaz=True, gt=True)
    out = tmp_path / "out"
    run_picker_fallback_stage(FAILED_TOPAZ_EVIDENCE, paths, out)

    expected = [
        out / "retrieval" / "alternative_picker_candidates.json",
        out / "retrieval" / "alternative_picker_candidates.md",
        out / "selection" / "alternative_picker_rerun_plan.json",
        out / "selection" / "alternative_picker_rerun_plan.md",
        out / "comparison" / "picker_model_selection_comparison.csv",
        out / "comparison" / "picker_model_selection_comparison.md",
        out / "comparison" / "final_picker_selection.json",
        out / "comparison" / "final_picker_selection.md",
        out / "picker_model_selection_report.md",
    ]
    for p in expected:
        assert p.exists(), f"missing expected artifact: {p}"

    plan = json.loads((out / "selection" / "alternative_picker_rerun_plan.json").read_text())
    assert plan["selection"]["selected"] == "cryotransformer_picker"
    assert plan["plan_only"] is True
    assert plan["executes"] is False
