"""Regression tests for build_failure_audit None/non-dict robustness.

T1 / EModelX / rule_only_agent exposed a crash where ``pf_result.get(...)`` in
``build_failure_audit`` assumed every payload / preflight block was a dict.
For model-building cases that never populate the picking-style preflight or
that arrive with ``preflight=None`` inside supervision payloads, the audit
call would raise ``AttributeError: 'NoneType' object has no attribute 'get'``.

These tests pin down the defensive coercion so the audit stays usable even
when evidence pieces are missing, without changing its output schema.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.failure_audit import (
    attach_audit_sources,
    audit_dict_to_json_bytes,
    build_failure_audit,
    render_failure_audit_markdown,
)


class TestBuildFailureAuditHandlesNonePayloads(unittest.TestCase):
    def test_none_preflight_and_none_supervision_payloads(self) -> None:
        smoke_summary = {
            "case_result": {
                "benchmark_id": "T1_tutorial5_map_emodelx_stopA",
                "backend": "emodelx",
                "category": "model_building",
                "status": "failed",
                "return_code": 1,
                "stderr": "some stderr",
                "preflight": None,
                "extra": {"llm_supervision_payloads": None},
            },
        }
        audit = build_failure_audit({}, smoke_summary)
        self.assertIn("execution_context", audit)
        self.assertIn("evidence", audit)
        # preflight_available falls back to False when no source provides it.
        self.assertFalse(audit["execution_context"]["preflight_available"])

    def test_supervision_payload_preflight_is_none(self) -> None:
        smoke_summary = {
            "case_result": {
                "benchmark_id": "T1_tutorial5_map_emodelx_stopA",
                "backend": "emodelx",
                "category": "model_building",
                "status": "failed",
                "return_code": 1,
                "stderr": "boom",
                "preflight": {"available": True, "execution_mode": "real"},
                "extra": {
                    "llm_supervision_payloads": {
                        "preflight": None,
                        "execution": None,
                        "evaluation": None,
                    }
                },
            },
        }
        audit = build_failure_audit({}, smoke_summary)
        self.assertTrue(audit["execution_context"]["preflight_available"])
        self.assertEqual(audit["execution_context"]["preflight_execution_mode"], "real")

    def test_non_dict_preflight_and_non_dict_extra(self) -> None:
        # Even degenerate shapes (preflight=False, extra=None) must not crash.
        smoke_summary = {
            "case_result": {
                "benchmark_id": "x",
                "backend": "emodelx",
                "status": "failed",
                "return_code": 2,
                "preflight": False,
                "extra": None,
            }
        }
        audit = build_failure_audit({}, smoke_summary)
        self.assertEqual(audit["run_identity"]["backend"], "emodelx")
        self.assertEqual(audit["execution_context"]["return_code"], 2)

    def test_missing_smoke_run_checks_and_picking_evaluation(self) -> None:
        smoke_summary = {
            "case_result": {
                "benchmark_id": "T1_seq_free",
                "backend": "emodelx",
                "category": "model_building",
                "status": "failed",
                "preflight": {"available": True},
                "extra": {},  # no smoke_run_checks, no picking_evaluation
            }
        }
        audit = build_failure_audit({}, smoke_summary)
        self.assertEqual(audit["evidence"]["smoke_run_checks"], {})
        self.assertIsNone(audit["evidence"]["picking_evaluation"])

    def test_latest_payload_none_safe(self) -> None:
        smoke_summary = {
            "case_result": {
                "benchmark_id": "x",
                "backend": "emodelx",
                "status": "failed",
                "preflight": None,
            }
        }
        audit = build_failure_audit(None, smoke_summary)  # type: ignore[arg-type]
        self.assertIn("failure_classification", audit)

    def test_fully_empty_smoke_summary(self) -> None:
        # No case_result at all.
        audit = build_failure_audit({}, {})
        self.assertIn("run_identity", audit)
        self.assertIsNone(audit["run_identity"]["backend"])

    def test_audit_remains_json_serializable(self) -> None:
        smoke_summary = {
            "case_result": {
                "benchmark_id": "T1",
                "backend": "emodelx",
                "category": "model_building",
                "status": "failed",
                "return_code": 1,
                "preflight": None,
                "extra": {"llm_supervision_payloads": None},
            }
        }
        audit = build_failure_audit({}, smoke_summary)
        # Must round-trip through json without errors.
        blob = audit_dict_to_json_bytes(audit)
        loaded = json.loads(blob.decode("utf-8"))
        self.assertEqual(loaded["run_identity"]["backend"], "emodelx")

    def test_markdown_render_does_not_crash_on_minimal_audit(self) -> None:
        smoke_summary = {
            "case_result": {
                "benchmark_id": "T1",
                "backend": "emodelx",
                "status": "failed",
                "preflight": None,
            }
        }
        audit = build_failure_audit({}, smoke_summary)
        md = render_failure_audit_markdown(audit)
        self.assertIn("Smoke Run Failure Audit", md)


class TestBuildFailureAuditWithBaselineRunnerShape(unittest.TestCase):
    """End-to-end: baseline_runner._write_audit feeds this exact smoke_summary."""

    def test_baseline_runner_style_smoke_summary_with_none_fields(self) -> None:
        # Shape mirrors what baseline_runner._write_audit passes today.
        case_result = {
            "benchmark_id": "fixed_workflow_emodelx__T1_tutorial5_map_emodelx_stopA",
            "category": "model_building",
            "backend": "emodelx",
            "status": "failed",
            "execution_mode": "real",
            "runtime_seconds": 0.0,
            "command": None,
            "artifacts": [],
            "blocked_reason": None,
            "preflight": None,  # ← the crash trigger in the real log
            "selection_decision": {"selected_backend": "emodelx"},
            "stdout": "",
            "stderr": "",
            "return_code": None,
            "extra": {
                "protocol_name": "rule_only_agent",
                "case_id": "T1_tutorial5_map_emodelx_stopA",
                "dataset_name": "tutorial5",
                "target_stage": "model_building",
                "evaluation_mode": "success_only",
                "selection_decision": {"selected_backend": "emodelx"},
                # NOTE: no llm_supervision_payloads key at all in this case
            },
        }
        audit = build_failure_audit({}, {"case_result": case_result})
        self.assertEqual(audit["run_identity"]["backend"], "emodelx")
        self.assertEqual(
            audit["run_identity"]["benchmark_id"],
            "fixed_workflow_emodelx__T1_tutorial5_map_emodelx_stopA",
        )
        self.assertIsNone(audit["execution_context"]["command"])
        self.assertFalse(audit["execution_context"]["preflight_available"])

    def test_writes_audit_json_and_md_to_tempdir(self) -> None:
        case_result = {
            "benchmark_id": "T1",
            "backend": "emodelx",
            "status": "failed",
            "preflight": None,
            "extra": {"llm_supervision_payloads": None},
        }
        audit = build_failure_audit({}, {"case_result": case_result})
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "failure_audit.json").write_bytes(audit_dict_to_json_bytes(audit))
            (root / "failure_audit.md").write_text(
                render_failure_audit_markdown(audit), encoding="utf-8"
            )
            self.assertTrue((root / "failure_audit.json").is_file())
            self.assertTrue((root / "failure_audit.md").is_file())
            self.assertGreater((root / "failure_audit.json").stat().st_size, 0)
            self.assertGreater((root / "failure_audit.md").stat().st_size, 0)


class TestBuildFailureAuditKeepsSchema(unittest.TestCase):
    """Make sure defensive coercion did not drop or rename any top-level key."""

    def test_top_level_keys_unchanged(self) -> None:
        audit = build_failure_audit({}, {"case_result": {"status": "failed"}})
        expected_keys = {
            "audit_metadata",
            "run_identity",
            "execution_context",
            "failure_classification",
            "evidence",
            "supervision_inputs",
            "suggested_actions",
            "human_summary",
        }
        self.assertEqual(set(audit.keys()), expected_keys)

    def test_evidence_keys_unchanged(self) -> None:
        audit = build_failure_audit({}, {"case_result": {"status": "failed"}})
        ev = audit["evidence"]
        expected = {
            "blocked_reason",
            "stderr_head",
            "stdout_head",
            "artifact_paths",
            "smoke_run_checks",
            "picking_evaluation",
            "regression_summary",
            "key_signals",
        }
        self.assertEqual(set(ev.keys()), expected)


class TestAttachAuditSourcesStillWorks(unittest.TestCase):
    def test_attach_source_paths_preserves_structure(self) -> None:
        audit = build_failure_audit({}, {"case_result": {"status": "failed"}})
        attach_audit_sources(audit, latest_path="/x/latest.json", smoke_summary_path=None)
        self.assertEqual(audit["audit_metadata"]["source_latest_json"], "/x/latest.json")
        self.assertIsNone(audit["audit_metadata"]["source_smoke_summary"])


if __name__ == "__main__":
    unittest.main()
