"""Lightweight tests for stage smoke removal specs (no large project copies)."""

from __future__ import annotations

import unittest

from app.stage_smoke_layout import STAGE_SMOKE_FIXTURE_ORDER, STAGE_SMOKE_REMOVALS


class TestStageSmokeLayout(unittest.TestCase):
    def test_fixture_order_matches_removals_keys(self) -> None:
        self.assertEqual(list(STAGE_SMOKE_FIXTURE_ORDER), list(STAGE_SMOKE_REMOVALS.keys()))

    def test_removals_are_non_empty_tuples_of_str(self) -> None:
        for name, dirs in STAGE_SMOKE_REMOVALS.items():
            self.assertIsInstance(dirs, tuple, msg=name)
            self.assertGreater(len(dirs), 0, msg=name)
            for d in dirs:
                self.assertIsInstance(d, str, msg=f"{name}/{d}")
                self.assertTrue(d, msg=f"{name} empty string")

    def test_postprocess_ready_drops_postprocess_not_refine(self) -> None:
        r = STAGE_SMOKE_REMOVALS["postprocess_ready"]
        self.assertIn("PostProcess", r)
        self.assertNotIn("Refine3D", r)


if __name__ == "__main__":
    unittest.main()
