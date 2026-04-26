from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import mrcfile
import numpy as np

from agent.candidate_map_qc import evaluate_candidate_map


class TestCandidateMapQC(unittest.TestCase):
    def test_missing_map_gets_failed_zero_score(self) -> None:
        metrics = evaluate_candidate_map("/tmp/this_file_does_not_exist_12345.mrc")
        self.assertEqual(metrics["qc_status"], "failed")
        self.assertEqual(float(metrics["qc_score"]), 0.0)
        self.assertFalse(metrics["file_exists"])

    def test_valid_synthetic_mrc_gets_positive_score(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "vol_valid.mrc"
            rng = np.random.default_rng(42)
            with mrcfile.new(str(path), overwrite=True) as mrc:
                mrc.set_data(rng.normal(0.0, 1.0, size=(16, 16, 16)).astype(np.float32))
            metrics = evaluate_candidate_map(path)
            self.assertIn(metrics["qc_status"], {"passed", "warning"})
            self.assertGreater(float(metrics["qc_score"]), 0.0)
            self.assertEqual(metrics["file_exists"], True)

    def test_near_constant_scores_lower_than_variable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            const_map = root / "const.mrc"
            var_map = root / "var.mrc"
            with mrcfile.new(str(const_map), overwrite=True) as mrc:
                mrc.set_data(np.ones((16, 16, 16), dtype=np.float32))
            rng = np.random.default_rng(7)
            with mrcfile.new(str(var_map), overwrite=True) as mrc:
                mrc.set_data(rng.normal(0.0, 1.0, size=(16, 16, 16)).astype(np.float32))
            const_metrics = evaluate_candidate_map(const_map)
            var_metrics = evaluate_candidate_map(var_map)
            self.assertLess(float(const_metrics["qc_score"]), float(var_metrics["qc_score"]))


if __name__ == "__main__":
    unittest.main()
