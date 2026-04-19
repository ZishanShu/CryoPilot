"""Tests for agent.case_registry case loader/parser."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from agent.case_registry import (
    CaseValidationError,
    ExperimentCase,
    default_cases,
    load_case,
    load_cases_from_dir,
    parse_case,
    resolve_case,
)


class TestParseCase(unittest.TestCase):
    def test_parse_workflow_case(self) -> None:
        case = parse_case(
            {
                "case_id": "c1",
                "case_type": "workflow_case",
                "dataset_name": "cryoppp_lite",
                "target_stage": "picking",
                "evaluation_mode": "picking_coords",
            }
        )
        self.assertIsInstance(case, ExperimentCase)
        self.assertEqual(case.case_type, "workflow_case")
        self.assertEqual(case.target_stage, "picking")

    def test_parse_map_level_case_defaults_target_stage(self) -> None:
        case = parse_case(
            {
                "case_id": "m1",
                "case_type": "map_level_case",
                "dataset_name": "mid3d",
            }
        )
        self.assertEqual(case.target_stage, "model_building")

    def test_missing_required_fields_rejected(self) -> None:
        with self.assertRaises(CaseValidationError):
            parse_case({"case_type": "workflow_case", "dataset_name": "d"})
        with self.assertRaises(CaseValidationError):
            parse_case({"case_id": "x", "dataset_name": "d"})

    def test_invalid_target_stage_for_map_level_rejected(self) -> None:
        with self.assertRaises(CaseValidationError):
            parse_case(
                {
                    "case_id": "m2",
                    "case_type": "map_level_case",
                    "dataset_name": "d",
                    "target_stage": "picking",
                }
            )

    def test_invalid_evaluation_mode_rejected(self) -> None:
        with self.assertRaises(CaseValidationError):
            parse_case(
                {
                    "case_id": "c2",
                    "case_type": "workflow_case",
                    "dataset_name": "d",
                    "evaluation_mode": "bogus",
                }
            )

    def test_to_dict_json_friendly(self) -> None:
        case = parse_case(
            {
                "case_id": "c3",
                "case_type": "workflow_case",
                "dataset_name": "d",
            }
        )
        json.dumps(case.to_dict())


class TestLoadCase(unittest.TestCase):
    def test_load_from_yaml(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "case_id": "cyaml",
                        "case_type": "workflow_case",
                        "dataset_name": "d",
                    }
                ),
                encoding="utf-8",
            )
            case = load_case(path)
            self.assertEqual(case.case_id, "cyaml")

    def test_load_from_dir_rejects_duplicate_ids(self) -> None:
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "a.yaml").write_text(
                yaml.safe_dump(
                    {
                        "case_id": "dup",
                        "case_type": "workflow_case",
                        "dataset_name": "d",
                    }
                ),
                encoding="utf-8",
            )
            (d / "b.yaml").write_text(
                yaml.safe_dump(
                    {
                        "case_id": "dup",
                        "case_type": "workflow_case",
                        "dataset_name": "d",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(CaseValidationError):
                load_cases_from_dir(d)


class TestResolveCase(unittest.TestCase):
    def test_default_cases_present(self) -> None:
        d = default_cases()
        self.assertIn("cryoppp_lite_topaz_smoke", d)
        self.assertIn("mid3d_map_only_model_building", d)

    def test_resolve_case_from_defaults(self) -> None:
        case = resolve_case("cryoppp_lite_topaz_smoke")
        self.assertEqual(case.case_type, "workflow_case")

    def test_resolve_case_missing_raises(self) -> None:
        with self.assertRaises(CaseValidationError):
            resolve_case("no_such_case_xyz")


if __name__ == "__main__":
    unittest.main()
