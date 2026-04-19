"""Tests for resolution normalization and extractor robustness."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from relion.qc_extractor import QCExtractor
from schemas.qc_metrics import QCMetrics


class TestResolutionNormalization(unittest.TestCase):
    """Verify invalid resolution inputs normalize to None."""

    def test_qc_metrics_zero_resolution_normalized_to_none(self) -> None:
        qc = QCMetrics(current_resolution=0.0)
        self.assertIsNone(qc.current_resolution)

    def test_qc_metrics_nan_string_normalized_to_none(self) -> None:
        qc = QCMetrics(current_resolution="nan")
        self.assertIsNone(qc.current_resolution)

    def test_qc_metrics_valid_resolution_preserved(self) -> None:
        qc = QCMetrics(current_resolution=3.8)
        self.assertEqual(qc.current_resolution, 3.8)

    def test_qc_extractor_with_invalid_resolution_does_not_crash(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Class2D/job001").mkdir(parents=True, exist_ok=True)
            (root / "Class2D/job001/classes.star").write_text("data_\ncompleted\n", encoding="utf-8")

            extractor = QCExtractor()
            # Simulate upstream resolution candidate that must be filtered out.
            extractor._collect_resolution_from_project = lambda _root, _parsed: 0.0  # type: ignore[method-assign]

            _, qc = extractor.extract(root)
            self.assertIsNone(qc.current_resolution)


if __name__ == "__main__":
    unittest.main()
