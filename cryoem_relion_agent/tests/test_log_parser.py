"""Tests for conservative log resolution parsing."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from relion.log_parser import LogParser


class TestLogParserResolution(unittest.TestCase):
    """Resolution only from lines with resolution semantics."""

    def setUp(self) -> None:
        self.parser = LogParser()

    def test_final_resolution_line(self) -> None:
        with TemporaryDirectory() as tmp:
            log = Path(tmp) / "run.log"
            log.write_text(
                "Some header\nFinal resolution = 3.8 A\n",
                encoding="utf-8",
            )
            out = self.parser.parse(log)
            self.assertAlmostEqual(float(out["resolution"]), 3.8, places=2)

    def test_iteration_noise_without_resolution_keyword(self) -> None:
        with TemporaryDirectory() as tmp:
            log = Path(tmp) / "run.log"
            log.write_text(
                "Refine3D/job029/ iteration 1\n",
                encoding="utf-8",
            )
            out = self.parser.parse(log)
            self.assertIsNone(out["resolution"])

    def test_fsc_correlation_line_not_used_alone(self) -> None:
        with TemporaryDirectory() as tmp:
            log = Path(tmp) / "run.log"
            log.write_text(
                "Correlation value 1.000000 in table\n",
                encoding="utf-8",
            )
            out = self.parser.parse(log)
            self.assertIsNone(out["resolution"])


if __name__ == "__main__":
    unittest.main()
