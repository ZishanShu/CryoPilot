"""Tests for case_registry parameterization (variable expansion / materialize)."""

from __future__ import annotations

import json
import os
import unittest

from agent.case_registry import (
    CaseMaterializationError,
    SUPPORTED_VARIABLES,
    default_cases,
    materialize_case,
    parse_case,
    resolve_case,
)


def _case_with_placeholders() -> "parse_case":
    return parse_case(
        {
            "case_id": "c_var",
            "case_type": "workflow_case",
            "dataset_name": "cryoppp_lite",
            "dataset_root": "${DATA_ROOT}/cryoppp_lite",
            "input_state": {
                "stage": "picking",
                "nested": {
                    "path": "${RUN_ROOT}/out",
                    "list": ["${PROJECT_ROOT}/x", "literal"],
                },
            },
            "available_inputs": {
                "micrographs_star": "${RUN_ROOT}/m.star",
                "project_root": "${PROJECT_ROOT}",
            },
            "notes": "for ${CASE_ROOT} only",
        }
    )


class TestSupportedVariables(unittest.TestCase):
    def test_four_supported_variables_exposed(self) -> None:
        self.assertEqual(
            set(SUPPORTED_VARIABLES),
            {"PROJECT_ROOT", "DATA_ROOT", "RUN_ROOT", "CASE_ROOT"},
        )


class TestMaterializeCase(unittest.TestCase):
    def test_expand_all_four_root_variables(self) -> None:
        case = _case_with_placeholders()
        materialized = materialize_case(
            case,
            variables={
                "PROJECT_ROOT": "/p",
                "DATA_ROOT": "/d",
                "RUN_ROOT": "/r",
                "CASE_ROOT": "/c",
            },
        )
        self.assertEqual(materialized.dataset_root, "/d/cryoppp_lite")
        self.assertEqual(materialized.input_state["nested"]["path"], "/r/out")
        self.assertEqual(
            materialized.input_state["nested"]["list"], ["/p/x", "literal"]
        )
        self.assertEqual(materialized.available_inputs["micrographs_star"], "/r/m.star")
        self.assertEqual(materialized.available_inputs["project_root"], "/p")
        self.assertEqual(materialized.notes, "for /c only")

    def test_missing_variable_raises_clear_error(self) -> None:
        case = _case_with_placeholders()
        with self.assertRaises(CaseMaterializationError) as ctx:
            materialize_case(case, variables={"DATA_ROOT": "/d"})
        self.assertIn("RUN_ROOT", str(ctx.exception))

    def test_materialized_case_is_json_serializable(self) -> None:
        case = _case_with_placeholders()
        materialized = materialize_case(
            case,
            variables={
                "PROJECT_ROOT": "/p",
                "DATA_ROOT": "/d",
                "RUN_ROOT": "/r",
                "CASE_ROOT": "/c",
            },
        )
        json.dumps(materialized.to_dict())

    def test_no_placeholders_passthrough(self) -> None:
        case = parse_case(
            {
                "case_id": "plain",
                "case_type": "workflow_case",
                "dataset_name": "d",
                "dataset_root": "/plain",
                "input_state": {"stage": "picking"},
            }
        )
        materialized = materialize_case(case, variables={})
        self.assertEqual(materialized.dataset_root, "/plain")
        self.assertEqual(materialized.input_state["stage"], "picking")

    def test_include_env_reads_environment_variables(self) -> None:
        case = _case_with_placeholders()
        saved = {k: os.environ.get(k) for k in SUPPORTED_VARIABLES}
        try:
            os.environ["PROJECT_ROOT"] = "/envp"
            os.environ["DATA_ROOT"] = "/envd"
            os.environ["RUN_ROOT"] = "/envr"
            os.environ["CASE_ROOT"] = "/envc"
            materialized = materialize_case(case, variables=None, include_env=True)
            self.assertEqual(materialized.dataset_root, "/envd/cryoppp_lite")
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_explicit_overrides_win_over_env(self) -> None:
        case = _case_with_placeholders()
        saved = os.environ.get("DATA_ROOT")
        try:
            os.environ["DATA_ROOT"] = "/env_value"
            os.environ["PROJECT_ROOT"] = "/envp"
            os.environ["RUN_ROOT"] = "/envr"
            os.environ["CASE_ROOT"] = "/envc"
            materialized = materialize_case(
                case,
                variables={"DATA_ROOT": "/override"},
                include_env=True,
            )
            self.assertEqual(materialized.dataset_root, "/override/cryoppp_lite")
        finally:
            if saved is None:
                os.environ.pop("DATA_ROOT", None)
            else:
                os.environ["DATA_ROOT"] = saved


class TestFailureSetRegistry(unittest.TestCase):
    def test_all_four_failure_cases_in_defaults(self) -> None:
        d = default_cases()
        for cid in (
            "F1_cryoppp_lite_topaz_input_format_failure",
            "F2_topaz_preflight_blocked_missing_executable",
            "F3_topaz_empty_output_case",
            "F4_mica_unavailable_missing_af3_dir",
        ):
            self.assertIn(cid, d)

    def test_failure_cases_resolvable_by_name(self) -> None:
        case = resolve_case("F1_cryoppp_lite_topaz_input_format_failure")
        self.assertEqual(case.input_state.get("expected_failure_phase"), "execution")
        self.assertEqual(
            case.input_state.get("expected_root_cause_guess"),
            "input_format_incompatibility",
        )

    def test_f4_is_map_level_and_model_building(self) -> None:
        case = resolve_case("F4_mica_unavailable_missing_af3_dir")
        self.assertEqual(case.case_type, "map_level_case")
        self.assertEqual(case.target_stage, "model_building")
        self.assertFalse(case.available_inputs.get("af3_dir"))


class TestMicaSuccessRegistry(unittest.TestCase):
    def test_m1_resolves_and_exposes_mica_inputs(self) -> None:
        case = resolve_case("M1_mica_real_success_stopA")
        self.assertEqual(case.case_type, "map_level_case")
        self.assertEqual(case.target_stage, "model_building")
        for key in (
            "map_path",
            "sequence_path",
            "af3_dir",
            "mica_repo_dir",
            "mica_python",
            "mica_weights_dir",
        ):
            self.assertIn(key, case.available_inputs)


if __name__ == "__main__":
    unittest.main()
