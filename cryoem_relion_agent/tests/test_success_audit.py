"""Tests for agent.success_audit."""

from __future__ import annotations

import json
import unittest

from agent.success_audit import (
    SUCCESS_AUDIT_VERSION,
    build_success_audit,
    classify_success,
    render_success_audit_markdown,
)


def _case_result_success() -> dict:
    return {
        "benchmark_id": "bench_case_ok",
        "category": "picking",
        "backend": "topaz",
        "status": "completed",
        "execution_mode": "real",
        "runtime_seconds": 1.23,
        "return_code": 0,
        "command": "topaz extract --...",
        "stderr": "",
        "stdout": "",
        "artifacts": ["out/picks.star"],
        "preflight": {"available": True, "execution_mode": "real"},
        "extra": {
            "dataset_id": "10005",
            "smoke_run_checks": {
                "preflight_ok": True,
                "execution_ok": True,
                "non_empty_output": True,
                "smoke_run_success": True,
            },
            "picking_evaluation": {
                "evaluation_available": True,
                "predicted_count": 120,
                "gt_count": 100,
                "precision": 0.8,
                "recall": 0.9,
                "f1": 0.85,
            },
        },
    }


def _latest_payload() -> dict:
    return {
        "run_metadata": {"run_id": "bench_run_ok"},
        "previous_run_metadata": {"previous_run_id": "bench_prev"},
        "regression_summary": {"unchanged_cases": ["bench_case_ok"]},
        "benchmark_results": [],
    }


class TestClassifySuccess(unittest.TestCase):
    def test_smoke_run_success_classified_high_evidence(self) -> None:
        cls = classify_success(_case_result_success())
        self.assertEqual(cls["success_kind"], "smoke_run_success")
        self.assertEqual(cls["evidence_strength"], "high")
        self.assertTrue(cls["evaluation_available"])

    def test_plain_completed_status_is_execution_completed(self) -> None:
        cr = _case_result_success()
        cr["extra"]["smoke_run_checks"] = {}
        cr["extra"]["picking_evaluation"] = {}
        cls = classify_success(cr)
        self.assertEqual(cls["success_kind"], "execution_completed")


class TestBuildSuccessAudit(unittest.TestCase):
    def test_audit_has_all_top_level_sections(self) -> None:
        audit = build_success_audit(
            _latest_payload(),
            {"case_result": _case_result_success()},
            protocol={"protocol_name": "rule_only_agent"},
            case={"case_id": "cryoppp_lite_topaz_smoke"},
        )
        for key in (
            "audit_metadata",
            "run_identity",
            "execution_context",
            "success_classification",
            "evidence",
        ):
            self.assertIn(key, audit)
        self.assertEqual(
            audit["audit_metadata"]["audit_version"], SUCCESS_AUDIT_VERSION
        )
        self.assertEqual(audit["run_identity"]["protocol_name"], "rule_only_agent")
        self.assertEqual(audit["run_identity"]["case_id"], "cryoppp_lite_topaz_smoke")

    def test_audit_is_json_serializable(self) -> None:
        audit = build_success_audit(
            _latest_payload(), {"case_result": _case_result_success()}
        )
        json.dumps(audit)

    def test_render_markdown_includes_backend_and_status(self) -> None:
        audit = build_success_audit(
            _latest_payload(), {"case_result": _case_result_success()}
        )
        md = render_success_audit_markdown(audit)
        self.assertIn("Success", md)
        self.assertIn("topaz", md)
        self.assertIn("completed", md)


if __name__ == "__main__":
    unittest.main()
