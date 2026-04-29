from __future__ import annotations

import json
import unittest
from pathlib import Path

from agent.failure_audit import (
    AUDIT_VERSION,
    build_failure_audit,
    classify_failure,
    render_failure_audit_markdown,
)


def _safe_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except (OSError, PermissionError):
        return False


def _minimal_latest(benchmark_id: str = "bench_case_1", run_id: str = "bench_run_1") -> dict:
    return {
        "run_metadata": {"run_id": run_id, "benchmark_count": 1},
        "previous_run_metadata": {"previous_run_id": "bench_prev"},
        "regression_summary": {"unchanged_cases": [benchmark_id]},
        "benchmark_results": [],
    }


class TestFailureAuditClassification(unittest.TestCase):
    def test_topaz_png_runtime_classifies_execution_and_input_format(self) -> None:
        case = {
            "benchmark_id": "cryoppp_topaz_smoke_10005",
            "category": "picking",
            "backend": "topaz",
            "status": "failed",
            "execution_mode": "real",
            "return_code": 1,
            "command": "topaz extract",
            "stderr": "AttributeError: 'PngImageFile' object has no attribute 'shape'",
            "stdout": "",
            "blocked_reason": "",
            "preflight": {"available": True, "execution_mode": "real"},
            "extra": {
                "dataset_id": "10005",
                "smoke_run_checks": {
                    "preflight_ok": True,
                    "execution_ok": False,
                    "non_empty_output": True,
                    "smoke_run_success": False,
                },
                "llm_supervision_payloads": {
                    "preflight": {"selected_backend": "topaz"},
                    "execution": {"return_code": 1},
                    "evaluation": {"f1": 0.0},
                },
            },
        }
        latest = _minimal_latest()
        fc = classify_failure(case, latest)
        self.assertEqual(fc["failure_phase"], "execution")
        self.assertEqual(fc["failure_kind"], "backend_runtime_error")
        self.assertEqual(fc["root_cause_guess"], "input_format_incompatibility")
        self.assertIn(fc["confidence"], ("high", "medium"))

    def test_preflight_blocked(self) -> None:
        case = {
            "benchmark_id": "x1",
            "status": "preflight_blocked",
            "execution_mode": "real",
            "preflight": {"available": False},
            "extra": {"smoke_run_checks": {}},
        }
        fc = classify_failure(case, None)
        self.assertEqual(fc["failure_phase"], "preflight")
        self.assertEqual(fc["failure_kind"], "preflight_blocked")

    def test_empty_output_failure(self) -> None:
        case = {
            "benchmark_id": "x2",
            "status": "failed",
            "return_code": 0,
            "stderr": "",
            "preflight": {"available": True},
            "extra": {
                "smoke_run_checks": {
                    "preflight_ok": True,
                    "execution_ok": True,
                    "non_empty_output": False,
                    "smoke_run_success": False,
                }
            },
        }
        fc = classify_failure(case, None)
        self.assertEqual(fc["failure_kind"], "empty_output_failure")


class TestFailureAuditBuild(unittest.TestCase):
    def test_build_has_required_top_level_keys(self) -> None:
        bid = "cryoppp_topaz_smoke_10005"
        latest = _minimal_latest(benchmark_id=bid)
        latest["benchmark_results"] = [
            {
                "benchmark_id": bid,
                "backend": "topaz",
                "return_code": 1,
                "blocked_reason": "PngImageFile",
            }
        ]
        smoke = {
            "case_result": {
                "benchmark_id": bid,
                "backend": "topaz",
                "category": "picking",
                "status": "failed",
                "execution_mode": "real",
                "command": "topaz extract",
                "return_code": 1,
                "runtime_seconds": 1.0,
                "stderr": "PngImageFile no attribute shape",
                "stdout": "",
                "preflight": {"available": True},
                "extra": {
                    "dataset_id": "10005",
                    "smoke_run_checks": {"preflight_ok": True, "execution_ok": False},
                    "llm_supervision_payloads": {
                        "preflight": {"selected_backend": "topaz"},
                        "execution": {},
                        "evaluation": {},
                    },
                },
            },
            "llm_diff_regression_payload": {"previous_run_id": "p1", "current_run_id": "bench_run_1"},
        }
        audit = build_failure_audit(latest, smoke)
        for k in (
            "audit_metadata",
            "run_identity",
            "execution_context",
            "failure_classification",
            "evidence",
            "supervision_inputs",
            "suggested_actions",
            "human_summary",
        ):
            self.assertIn(k, audit)
        self.assertEqual(audit["audit_metadata"]["audit_version"], AUDIT_VERSION)
        self.assertTrue(audit["human_summary"])
        self.assertTrue(isinstance(audit["suggested_actions"], list))
        self.assertLessEqual(len(audit["suggested_actions"]), 3)
        self.assertIn("diff_regression", audit["supervision_inputs"])

    def test_markdown_sections(self) -> None:
        audit = build_failure_audit(
            _minimal_latest(),
            {
                "case_result": {
                    "benchmark_id": "b",
                    "backend": "topaz",
                    "category": "picking",
                    "status": "failed",
                    "execution_mode": "real",
                    "command": "x",
                    "return_code": 1,
                    "stderr": "err",
                    "preflight": {"available": True},
                    "extra": {
                        "smoke_run_checks": {},
                        "llm_supervision_payloads": {},
                    },
                }
            },
        )
        md = render_failure_audit_markdown(audit)
        for title in (
            "# Smoke Run Failure Audit",
            "## Run Identity",
            "## Failure Classification",
            "## Key Evidence",
            "## Suggested Actions",
            "## Human Summary",
        ):
            self.assertIn(title, md)


@unittest.skipUnless(
    _safe_is_file(Path(__file__).resolve().parents[1].joinpath(".cryoem_agent/benchmarks/latest.json")),
    "repo benchmark latest.json not present",
)
class TestFailureAuditRealFixtures(unittest.TestCase):
    def test_real_latest_and_smoke_sample(self) -> None:
        root = Path(__file__).resolve().parents[1]
        latest_path = root / ".cryoem_agent" / "benchmarks" / "latest.json"
        smoke_path = (
            Path("/root/share/szs/CryoAgent/real_runs/cryoppp_topaz_smoke/10005/outputs/reports/smoke_run_summary.json")
        )
        if not _safe_is_file(smoke_path):
            self.skipTest("real smoke summary not in workspace")
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        audit = build_failure_audit(latest, smoke)
        fc = audit["failure_classification"]
        self.assertEqual(fc["failure_phase"], "execution")
        self.assertEqual(fc["failure_kind"], "backend_runtime_error")
        self.assertEqual(fc["root_cause_guess"], "input_format_incompatibility")
