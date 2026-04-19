from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.eval.picking_eval import evaluate_picking_predictions


class TestPickingEval(unittest.TestCase):
    def test_metrics_are_computed(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred = root / "pred.txt"
            gt = root / "gt.txt"
            pred.write_text("10 10\n20 20\n30 30\n", encoding="utf-8")
            gt.write_text("10 10\n20 22\n90 90\n", encoding="utf-8")
            out = evaluate_picking_predictions(
                predicted_coord_file=pred,
                gt_coord_files=[gt],
                match_radius_px=3.0,
            )
            self.assertEqual(out["predicted_count"], 3)
            self.assertEqual(out["gt_count"], 3)
            self.assertEqual(out["matched_count"], 2)
            self.assertGreater(out["precision"], 0.0)
            self.assertGreater(out["recall"], 0.0)
            self.assertGreater(out["f1"], 0.0)
            json.dumps(out)


if __name__ == "__main__":
    unittest.main()

