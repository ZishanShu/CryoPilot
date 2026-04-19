from __future__ import annotations

import json
import unittest

from agent.llm_supervision_payloads import (
    build_diff_regression_supervisor_input,
    build_evaluation_supervisor_input,
    build_execution_supervisor_input,
    build_preflight_supervisor_input,
)


class TestLLMSupervisionPayloads(unittest.TestCase):
    def test_all_payloads_are_json_friendly(self) -> None:
        preflight = build_preflight_supervisor_input(
            selected_backend="topaz",
            selection_reason="preferred_backend_selected",
            preflight_result={
                "available": True,
                "execution_mode": "real",
                "missing_inputs": [],
                "missing_dependencies": [],
            },
        )
        execution = build_execution_supervisor_input(
            {
                "command": "topaz predict ...",
                "return_code": 0,
                "stdout": "ok",
                "stderr": "",
                "status": "completed",
                "output_dir": "/tmp/out",
                "artifacts": ["/tmp/out/pred.txt"],
            }
        )
        evaluation = build_evaluation_supervisor_input(
            evaluation={
                "predicted_count": 10,
                "gt_count": 12,
                "matched_count": 8,
                "precision": 0.8,
                "recall": 0.66,
                "f1": 0.72,
            },
            dataset_id="ds1",
            particle_diameter_angstrom=150.0,
            topaz_threshold=-6.0,
        )
        diff = build_diff_regression_supervisor_input(
            {
                "regression_summary": {"improved_cases": ["a"]},
                "benchmark_diff": {"case_diffs": [], "selection_diffs": {}},
                "run_metadata": {"run_id": "r2"},
                "previous_run_metadata": {"previous_run_id": "r1"},
            }
        )
        for payload in (preflight, execution, evaluation, diff):
            json.dumps(payload)
        self.assertIn("selected_backend", preflight)
        self.assertIn("benchmark_result_status", execution)
        self.assertIn("f1", evaluation)
        self.assertIn("current_run_id", diff)


if __name__ == "__main__":
    unittest.main()

