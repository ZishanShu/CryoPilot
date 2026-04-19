from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.run_cryoppp_topaz_smoke import _build_case, _evaluate_and_enrich


class TestCryoPPPSmokeRunCaseBuild(unittest.TestCase):
    def test_case_uses_derived_particle_diameter(self) -> None:
        manifest = {
            "dataset_id": "d1",
            "output_paths": {
                "micrographs_star_path": "/tmp/m.star",
                "topaz_output_dir": "/tmp/out",
            },
        }
        case = _build_case(
            manifest,
            particle_diameter_angstrom=120.0,
            topaz_executable=None,
            topaz_model_path=None,
            topaz_threshold=-6.0,
            topaz_device="0",
        )
        self.assertEqual(case["inputs"]["particle_diameter_angstrom"], 120.0)
        self.assertEqual(case["backend"], "topaz")

    def test_case_override_values_take_precedence(self) -> None:
        manifest = {
            "dataset_id": "d1",
            "output_paths": {
                "micrographs_star_path": "/tmp/m.star",
                "topaz_output_dir": "/tmp/out",
            },
        }
        case = _build_case(
            manifest,
            particle_diameter_angstrom=160.0,
            topaz_executable="/opt/topaz",
            topaz_model_path="/opt/model.sav",
            topaz_threshold=-3.2,
            topaz_device="cpu",
        )
        self.assertEqual(case["inputs"]["topaz_executable"], "/opt/topaz")
        self.assertEqual(case["inputs"]["topaz_model_path"], "/opt/model.sav")
        self.assertEqual(case["inputs"]["topaz_threshold"], -3.2)
        self.assertEqual(case["inputs"]["topaz_device"], "cpu")


class TestCryoPPPSmokeRunIntegrationBits(unittest.TestCase):
    def test_evaluate_and_success_flags(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred = root / "pred.txt"
            pred.write_text("10 10\n20 20\n", encoding="utf-8")
            gt = root / "gt.txt"
            gt.write_text("10 10\n40 40\n", encoding="utf-8")
            result = {
                "status": "completed",
                "preflight": {"available": True},
                "topaz_output_path": str(pred),
                "selection_decision": {"selection_reason": "preferred_backend_selected"},
                "stdout": "ok",
                "stderr": "",
                "return_code": 0,
                "artifacts": [str(pred)],
                "generated_paths": [str(pred)],
            }
            manifest = {"dataset_id": "d1", "gt_coordinate_files": [str(gt)]}
            out = _evaluate_and_enrich(
                result,
                manifest=manifest,
                particle_diameter_angstrom=150.0,
                topaz_threshold=-6.0,
                match_radius_px=3.0,
            )
            checks = out["extra"]["smoke_run_checks"]
            self.assertTrue(checks["smoke_run_success"])
            self.assertEqual(out["extra"]["picking_evaluation"]["predicted_count"], 2)
            self.assertIn("llm_supervision_payloads", out["extra"])

    def test_preflight_blocked_does_not_mark_success(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt = root / "gt.txt"
            gt.write_text("10 10\n", encoding="utf-8")
            result = {
                "status": "preflight_blocked",
                "preflight": {"available": False, "blocked_reason": "missing_required_inputs"},
                "topaz_output_path": str(root / "missing.txt"),
                "selection_decision": {"selection_reason": "preferred_backend_selected"},
            }
            manifest = {"dataset_id": "d1", "gt_coordinate_files": [str(gt)]}
            out = _evaluate_and_enrich(
                result,
                manifest=manifest,
                particle_diameter_angstrom=150.0,
                topaz_threshold=-6.0,
                match_radius_px=3.0,
            )
            self.assertFalse(out["extra"]["smoke_run_checks"]["smoke_run_success"])


if __name__ == "__main__":
    unittest.main()

