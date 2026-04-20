"""Tests for agent.failure_benchmark (E2 runner)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.case_registry import default_cases, materialize_case
from agent.failure_benchmark import (
    ADVICE_SOURCES,
    build_advice,
    evaluate_case_under_source,
    persist_failure_benchmark,
    render_failure_benchmark_markdown,
    run_failure_benchmark,
    synthesize_case_result,
)


_FAILURE_CASES = (
    "F1_cryoppp_lite_topaz_input_format_failure",
    "F2_topaz_preflight_blocked_missing_executable",
    "F3_topaz_empty_output_case",
    "F4_mica_unavailable_missing_af3_dir",
)


def _load_all_failure_cases():
    d = default_cases()
    return [d[cid] for cid in _FAILURE_CASES]


class TestSynthesizeCaseResult(unittest.TestCase):
    def test_f1_maps_to_backend_runtime_error(self) -> None:
        d = default_cases()
        cr = synthesize_case_result(d["F1_cryoppp_lite_topaz_input_format_failure"])
        self.assertEqual(cr["status"], "failed")
        self.assertEqual(cr["return_code"], 1)
        self.assertIn("PngImageFile", cr["stderr"])

    def test_f2_maps_to_preflight_blocked(self) -> None:
        d = default_cases()
        cr = synthesize_case_result(d["F2_topaz_preflight_blocked_missing_executable"])
        self.assertEqual(cr["status"], "preflight_blocked")
        self.assertFalse(cr["preflight"]["available"])

    def test_f3_maps_to_empty_output_failure(self) -> None:
        d = default_cases()
        cr = synthesize_case_result(d["F3_topaz_empty_output_case"])
        self.assertEqual(cr["return_code"], 0)
        self.assertFalse(cr["extra"]["smoke_run_checks"]["non_empty_output"])

    def test_f4_maps_to_mica_preflight_blocked(self) -> None:
        d = default_cases()
        cr = synthesize_case_result(d["F4_mica_unavailable_missing_af3_dir"])
        self.assertEqual(cr["backend"], "mica")
        self.assertEqual(cr["category"], "model_building")
        self.assertEqual(cr["status"], "preflight_blocked")


class TestBuildAdvice(unittest.TestCase):
    def test_none_advice_is_empty(self) -> None:
        d = default_cases()
        case = d["F1_cryoppp_lite_topaz_input_format_failure"]
        cr = synthesize_case_result(case)
        classification = {
            "failure_phase": "execution",
            "failure_kind": "backend_runtime_error",
            "root_cause_guess": "input_format_incompatibility",
        }
        self.assertEqual(build_advice("none", classification=classification, case=case, case_result=cr), [])

    def test_rule_only_advice_for_input_format(self) -> None:
        d = default_cases()
        case = d["F1_cryoppp_lite_topaz_input_format_failure"]
        cr = synthesize_case_result(case)
        classification = {
            "failure_phase": "execution",
            "failure_kind": "backend_runtime_error",
            "root_cause_guess": "input_format_incompatibility",
        }
        advice = build_advice("rule_only", classification=classification, case=case, case_result=cr)
        self.assertTrue(len(advice) >= 1)
        for item in advice:
            self.assertIn("action_id", item)
            self.assertIn("actionable_steps", item)

    def test_llm_advice_extends_rules_and_adds_mica_reroute(self) -> None:
        d = default_cases()
        case = d["F4_mica_unavailable_missing_af3_dir"]
        cr = synthesize_case_result(case)
        classification = {
            "failure_phase": "preflight",
            "failure_kind": "preflight_blocked",
            "root_cause_guess": "missing_model_or_dependency",
        }
        advice = build_advice("llm", classification=classification, case=case, case_result=cr)
        aids = {a.get("action_id") for a in advice}
        self.assertIn("reroute_to_emodelx", aids)
        self.assertIn("install_missing_dependency", aids)


class TestEvaluateCaseUnderSource(unittest.TestCase):
    def test_fields_all_populated(self) -> None:
        d = default_cases()
        case = d["F1_cryoppp_lite_topaz_input_format_failure"]
        result = evaluate_case_under_source(case=case, advice_source="rule_only")
        rd = result.to_dict()
        for key in (
            "case_id",
            "advice_source",
            "initial_status",
            "failure_phase",
            "failure_kind",
            "root_cause_guess",
            "suggested_actions",
            "actionable",
            "post_action_state",
            "recovered",
            "intervention_count",
            "notes",
        ):
            self.assertIn(key, rd)

    def test_none_source_never_recovers(self) -> None:
        d = default_cases()
        for cid in _FAILURE_CASES:
            r = evaluate_case_under_source(case=d[cid], advice_source="none")
            self.assertFalse(r.recovered, cid)
            self.assertEqual(r.intervention_count, 1, cid)

    def test_llm_recovers_on_mica_reroute(self) -> None:
        d = default_cases()
        r = evaluate_case_under_source(
            case=d["F4_mica_unavailable_missing_af3_dir"], advice_source="llm"
        )
        self.assertTrue(r.recovered)
        self.assertTrue(r.actionable)


class TestRunFailureBenchmark(unittest.TestCase):
    def test_full_run_has_all_sources_and_summary(self) -> None:
        cases = _load_all_failure_cases()
        payload = run_failure_benchmark(cases)
        self.assertEqual(payload["benchmark_name"], "e2_failure_set")
        self.assertEqual(set(payload["advice_sources"]), set(ADVICE_SOURCES))
        self.assertEqual(len(payload["results"]), len(cases) * len(ADVICE_SOURCES))
        summary = payload["summary"]
        self.assertEqual(summary["total_cases"], len(cases))
        for key in (
            "actionable_advice_rate_by_source",
            "recovered_by_source",
            "recovery_lift",
            "intervention_reduction",
            "time_to_recovery",
        ):
            self.assertIn(key, summary)
        # time_to_recovery should be flagged unavailable in v1.
        self.assertFalse(summary["time_to_recovery"]["available"])
        # Payload must be JSON-friendly.
        json.dumps(payload)

    def test_recovery_lift_nonnegative_for_advice_sources(self) -> None:
        payload = run_failure_benchmark(_load_all_failure_cases())
        lift = payload["summary"]["recovery_lift"]
        self.assertEqual(lift["none"], 0.0)
        self.assertGreaterEqual(lift["rule_only"], 0.0)
        self.assertGreaterEqual(lift["llm"], lift["rule_only"])

    def test_intervention_reduction_nonnegative(self) -> None:
        payload = run_failure_benchmark(_load_all_failure_cases())
        red = payload["summary"]["intervention_reduction"]
        self.assertEqual(red["none"], 0)
        self.assertGreaterEqual(red["rule_only"], 0)
        self.assertGreaterEqual(red["llm"], red["rule_only"])

    def test_results_contain_expected_keys(self) -> None:
        payload = run_failure_benchmark(_load_all_failure_cases())
        for row in payload["results"]:
            for key in (
                "case_id",
                "advice_source",
                "initial_status",
                "failure_phase",
                "failure_kind",
                "root_cause_guess",
                "suggested_actions",
                "actionable",
                "post_action_state",
                "recovered",
                "intervention_count",
            ):
                self.assertIn(key, row)


class TestRenderAndPersist(unittest.TestCase):
    def test_markdown_lists_all_sources_and_cases(self) -> None:
        payload = run_failure_benchmark(_load_all_failure_cases())
        md = render_failure_benchmark_markdown(payload)
        self.assertIn("E2 Failure-Set Benchmark", md)
        for source in ADVICE_SOURCES:
            self.assertIn(source, md)
        for cid in _FAILURE_CASES:
            self.assertIn(cid, md)
        self.assertIn("Time-to-recovery", md)

    def test_persist_writes_latest_and_history(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = run_failure_benchmark(_load_all_failure_cases())
            info = persist_failure_benchmark(project_root=root, payload=payload)
            for key in ("latest_json", "latest_md", "history_json", "history_md"):
                self.assertTrue(Path(info[key]).is_file())
            # latest.json should round-trip to identical cases list.
            loaded = json.loads(Path(info["latest_json"]).read_text(encoding="utf-8"))
            self.assertEqual(loaded["cases"], payload["cases"])


class TestFailureBenchmarkAcceptsMaterializedCases(unittest.TestCase):
    def test_materialized_cases_still_runnable(self) -> None:
        d = default_cases()
        vars_map = {
            "PROJECT_ROOT": "/p",
            "DATA_ROOT": "/d",
            "RUN_ROOT": "/r",
            "CASE_ROOT": "/c",
        }
        cases = []
        for cid in _FAILURE_CASES:
            # Defaults have no placeholders; materialize is a no-op, still valid.
            cases.append(materialize_case(d[cid], variables=vars_map))
        payload = run_failure_benchmark(cases)
        self.assertEqual(payload["summary"]["total_cases"], len(cases))


if __name__ == "__main__":
    unittest.main()
