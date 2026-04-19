"""Tests for STAR resolution and micrograph parsing."""

from __future__ import annotations

import unittest

from relion.star_parser import (
    count_unique_micrographs_from_star_text,
    extract_resolution_from_star_text,
)


class TestStarParserResolution(unittest.TestCase):
    """Resolution must not come from FSC tables or column headers."""

    def test_postprocess_general_rlnFinalResolution(self) -> None:
        text = """
data_general
_rlnFinalResolution                               3.387915
"""
        self.assertAlmostEqual(extract_resolution_from_star_text(text), 3.387915, places=4)

    def test_fsc_table_does_not_yield_1_0(self) -> None:
        """FSC ``_rlnResolution`` / correlation 1.000000 must not become global resolution."""
        text = """
data_fsc
loop_
_rlnSpectralIndex #1
_rlnResolution #2
_rlnAngstromResolution #3
           0     0.001001   999.000000     1.000000
"""
        self.assertIsNone(extract_resolution_from_star_text(text))

    def test_no_resolution_context_returns_none(self) -> None:
        text = "job001 iteration 1 run #1\n"
        self.assertIsNone(extract_resolution_from_star_text(text))


class TestStarParserMicrographs(unittest.TestCase):
    """Unique micrograph names on particle tables."""

    def test_unique_micrograph_names_not_row_count(self) -> None:
        header = """
data_particles
loop_
_rlnImageName #1
_rlnMicrographName #2
"""
        rows = "\n".join(
            f"000001@stack.mrcs micrographs/{i % 123}.mrc"
            for i in range(10000)
        )
        text = header + rows
        # 10000 rows but only 123 unique micrograph IDs
        n = count_unique_micrographs_from_star_text(text)
        self.assertEqual(n, 123)


if __name__ == "__main__":
    unittest.main()
