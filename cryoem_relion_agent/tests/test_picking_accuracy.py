"""Tests for ``agent.picking_accuracy`` and the Topaz output converter."""

from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.picking_accuracy import (
    evaluate_picking_accuracy,
    evaluate_picking_accuracy_radius_sensitivity,
    write_radius_sensitivity_outputs,
)
from scripts.convert_topaz_output_to_pred_tsv import convert_topaz_output_to_pred_tsv


def _write_tsv(path: Path, rows, fieldnames) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return path


def _gt_rows(image, coords, split="test"):
    return [
        {"split": split, "image_name": image, "x_coord": x, "y_coord": y}
        for x, y in coords
    ]


def _pred_rows(image, coords):
    return [
        {"image_name": image, "x_coord": x, "y_coord": y, "score": "0.5"}
        for x, y in coords
    ]


class TestEvaluatePickingAccuracyExactMatch(unittest.TestCase):
    def test_perfect_match(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            coords = [(100.0, 200.0), (300.0, 400.0), (500.0, 600.0)]
            gt = _write_tsv(
                root / "gt.tsv",
                _gt_rows("imgA", coords),
                ["split", "image_name", "x_coord", "y_coord"],
            )
            pred = _write_tsv(
                root / "pred.tsv",
                _pred_rows("imgA", coords),
                ["image_name", "x_coord", "y_coord", "score"],
            )
            metrics = evaluate_picking_accuracy(
                gt_tsv=gt, pred_tsv=pred, split="test", matching_radius_px=10
            )
            self.assertEqual(metrics["n_gt"], 3)
            self.assertEqual(metrics["n_pred"], 3)
            self.assertEqual(metrics["n_matched"], 3)
            self.assertEqual(metrics["precision"], 1.0)
            self.assertEqual(metrics["recall"], 1.0)
            self.assertEqual(metrics["f1"], 1.0)
            self.assertFalse(metrics["empty_output"])


class TestEvaluatePickingAccuracyEmptyPrediction(unittest.TestCase):
    def test_zero_predictions(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt = _write_tsv(
                root / "gt.tsv",
                _gt_rows("imgA", [(1.0, 2.0), (3.0, 4.0)]),
                ["split", "image_name", "x_coord", "y_coord"],
            )
            pred = _write_tsv(root / "pred.tsv", [], ["image_name", "x_coord", "y_coord", "score"])
            metrics = evaluate_picking_accuracy(
                gt_tsv=gt, pred_tsv=pred, split="test", matching_radius_px=30
            )
            self.assertEqual(metrics["n_pred"], 0)
            self.assertEqual(metrics["n_matched"], 0)
            self.assertEqual(metrics["precision"], 0.0)
            self.assertEqual(metrics["recall"], 0.0)
            self.assertEqual(metrics["f1"], 0.0)
            self.assertTrue(metrics["empty_output"])


class TestPredictionDoesNotMatchTwice(unittest.TestCase):
    def test_one_pred_only_matches_one_gt(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt = _write_tsv(
                root / "gt.tsv",
                _gt_rows("imgA", [(100.0, 100.0), (105.0, 100.0)]),
                ["split", "image_name", "x_coord", "y_coord"],
            )
            # Single prediction sits between the two GTs; only one should match.
            pred = _write_tsv(
                root / "pred.tsv",
                _pred_rows("imgA", [(102.5, 100.0)]),
                ["image_name", "x_coord", "y_coord", "score"],
            )
            metrics = evaluate_picking_accuracy(
                gt_tsv=gt, pred_tsv=pred, split="test", matching_radius_px=10
            )
            self.assertEqual(metrics["n_pred"], 1)
            self.assertEqual(metrics["n_gt"], 2)
            self.assertEqual(metrics["n_matched"], 1)
            self.assertEqual(metrics["precision"], 1.0)
            self.assertEqual(metrics["recall"], 0.5)


class TestRadiusSensitivity(unittest.TestCase):
    def test_more_matches_at_larger_radius(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt_coords = [(0.0, 0.0), (100.0, 0.0), (200.0, 0.0)]
            pred_coords = [(15.0, 0.0), (125.0, 0.0), (235.0, 0.0)]  # offsets 15, 25, 35
            gt = _write_tsv(
                root / "gt.tsv",
                _gt_rows("imgA", gt_coords),
                ["split", "image_name", "x_coord", "y_coord"],
            )
            pred = _write_tsv(
                root / "pred.tsv",
                _pred_rows("imgA", pred_coords),
                ["image_name", "x_coord", "y_coord", "score"],
            )
            rows = evaluate_picking_accuracy_radius_sensitivity(
                gt_tsv=gt, pred_tsv=pred, split="test", radii_px=(20, 30, 40)
            )
            self.assertEqual(len(rows), 3)
            matched_at = {row["matching_radius_px"]: row["n_matched"] for row in rows}
            self.assertEqual(matched_at[20.0], 1)  # only the first (offset 15)
            self.assertEqual(matched_at[30.0], 2)  # offsets 15, 25
            self.assertEqual(matched_at[40.0], 3)  # all three

            csv_path, summary_path = write_radius_sensitivity_outputs(
                rows,
                csv_path=root / "by_r.csv",
                summary_json_path=root / "summary.json",
                primary_radius_px=30,
            )
            self.assertTrue(csv_path.is_file())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["primary_radius_px"], 30.0)
            self.assertEqual(len(summary["radii"]), 3)
            self.assertEqual(summary["primary_metrics"]["n_matched"], 2)


class TestSplitFilter(unittest.TestCase):
    def test_only_requested_split_evaluated(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt = _write_tsv(
                root / "gt.tsv",
                _gt_rows("imgA", [(10.0, 10.0)], split="train")
                + _gt_rows("imgB", [(20.0, 20.0)], split="test"),
                ["split", "image_name", "x_coord", "y_coord"],
            )
            pred = _write_tsv(
                root / "pred.tsv",
                _pred_rows("imgB", [(20.0, 20.0)]),
                ["image_name", "x_coord", "y_coord", "score"],
            )
            metrics = evaluate_picking_accuracy(
                gt_tsv=gt, pred_tsv=pred, split="test", matching_radius_px=5
            )
            self.assertEqual(metrics["n_gt"], 1)
            self.assertEqual(metrics["n_matched"], 1)


class TestTopazStarConverter(unittest.TestCase):
    def test_converts_topaz_star_to_pred_tsv(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            star = root / "picks.star"
            star.write_text(
                "data_\n"
                "loop_\n"
                "_rlnMicrographName #1\n"
                "_rlnCoordinateX #2\n"
                "_rlnCoordinateY #3\n"
                "_rlnAutopickFigureOfMerit #4\n"
                "/abs/imgA.mrc 100.5 200.25 -3.0\n"
                "/abs/imgA.mrc 300.0 400.0 -2.5\n"
                "imgB.mrc 50 60 -4.0\n",
                encoding="utf-8",
            )
            out = convert_topaz_output_to_pred_tsv(star, root / "pred.tsv")
            text = out.read_text(encoding="utf-8")
            header = text.splitlines()[0].split("\t")
            self.assertEqual(header, ["image_name", "x_coord", "y_coord", "score"])
            rows = [line.split("\t") for line in text.splitlines()[1:]]
            self.assertEqual(rows[0][0], "imgA")
            self.assertAlmostEqual(float(rows[0][1]), 100.5)
            self.assertAlmostEqual(float(rows[0][2]), 200.25)
            self.assertAlmostEqual(float(rows[0][3]), -3.0)
            self.assertEqual(rows[2][0], "imgB")


if __name__ == "__main__":
    unittest.main()
