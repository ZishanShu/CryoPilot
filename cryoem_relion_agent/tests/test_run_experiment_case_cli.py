"""Tests for the run_experiment_case CLI variable-override behavior."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from scripts.run_experiment_case import main as run_experiment_main


def _write_case_with_placeholders(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "case_id": "cli_var_case",
                "case_type": "workflow_case",
                "dataset_name": "cryoppp_lite",
                "dataset_root": "${DATA_ROOT}/cryoppp_lite",
                "input_state": {"stage": "picking", "run_out": "${RUN_ROOT}/out"},
                "available_inputs": {"cases_here": "${CASE_ROOT}/me"},
                "target_stage": "picking",
                "evaluation_mode": "none",
                "notes": "cli test ${PROJECT_ROOT}",
            }
        ),
        encoding="utf-8",
    )


class TestRunExperimentCaseCLI(unittest.TestCase):
    def test_cli_expands_root_variables_into_experiment_result(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases_dir = root / "configs" / "cases"
            cases_dir.mkdir(parents=True)
            _write_case_with_placeholders(cases_dir / "cli_var_case.yaml")
            data_root = root / "data"
            run_root = root / "runs"
            case_root = root / "cases_wd"
            output_dir = root / "out"

            rc = run_experiment_main(
                [
                    "--project-root",
                    str(root),
                    "--protocol",
                    "fixed_workflow_topaz",
                    "--case",
                    "cli_var_case",
                    "--case-dir",
                    str(cases_dir),
                    "--data-root",
                    str(data_root),
                    "--run-root",
                    str(run_root),
                    "--case-root",
                    str(case_root),
                    "--output-dir",
                    str(output_dir),
                    "--dry-run",
                    "--no-persist",
                    "--no-audit",
                ]
            )
            self.assertIn(rc, (0, 1, 2))
            result_json = json.loads((output_dir / "experiment_result.json").read_text(encoding="utf-8"))
            case_result_extra = result_json["case_result"]["extra"]
            self.assertIn("cases_here", str(case_result_extra))
            # dataset_root is not in extra directly, verify via case_id
            self.assertEqual(result_json["case_id"], "cli_var_case")
            # Expanded path should appear somewhere in the serialized payload.
            serialized = json.dumps(result_json)
            self.assertIn(str(data_root.resolve()), serialized)
            self.assertIn(str(run_root.resolve()), serialized)
            self.assertIn(str(case_root.resolve()), serialized)

    def test_cli_errors_cleanly_when_variable_missing(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases_dir = root / "configs" / "cases"
            cases_dir.mkdir(parents=True)
            _write_case_with_placeholders(cases_dir / "cli_var_case.yaml")

            with self.assertRaises(SystemExit) as ctx:
                run_experiment_main(
                    [
                        "--project-root",
                        str(root),
                        "--protocol",
                        "fixed_workflow_topaz",
                        "--case",
                        "cli_var_case",
                        "--case-dir",
                        str(cases_dir),
                        # Intentionally omit --data-root / --run-root / --case-root.
                        "--dry-run",
                        "--no-persist",
                        "--no-audit",
                    ]
                )
            msg = str(ctx.exception)
            self.assertIn("materialization failed", msg)


if __name__ == "__main__":
    unittest.main()
