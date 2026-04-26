"""Tests for picking QC gate (parsing, thresholds, GT evaluation)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.picking_qc import (
    DEFAULT_THRESHOLDS,
    count_micrographs_in_star,
    evaluate_picking_output,
    parse_topaz_picks_star,
)


_STAR_PICKS_HEADER = (
    "data_\n"
    "\n"
    "loop_\n"
    "_rlnMicrographName #1\n"
    "_rlnCoordinateX #2\n"
    "_rlnCoordinateY #3\n"
    "_rlnAutopickFigureOfMerit #4\n"
)


def _write_star(path: Path, lines: list[str]) -> None:
    body = _STAR_PICKS_HEADER + "\n".join(lines) + ("\n" if lines else "")
    path.write_text(body, encoding="utf-8")


def _write_micrographs_star(path: Path, names: list[str]) -> None:
    body = (
        "data_\n\nloop_\n"
        "_rlnMicrographName #1\n"
        + "\n".join(names)
        + "\n"
    )
    path.write_text(body, encoding="utf-8")


class TestParseTopazPicksStar(unittest.TestCase):
    def test_parses_small_star_picks(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "picks.star"
            _write_star(
                p,
                [
                    "mic_001.mrc 100.0 200.0 -3.5",
                    "mic_001.mrc 150.0 250.0 -3.1",
                    "mic_002.mrc 50.0 80.0 -4.2",
                ],
            )
            parsed = parse_topaz_picks_star(p)
            rows = parsed["rows"]
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["micrograph"], "mic_001.mrc")
            self.assertAlmostEqual(rows[0]["x"], 100.0)
            self.assertAlmostEqual(rows[0]["y"], 200.0)
            self.assertAlmostEqual(rows[2]["score"], -4.2)

    def test_parses_tsv_picks(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "picks.tsv"
            p.write_text(
                "image_name\tx_coord\ty_coord\tscore\n"
                "mic_a\t10\t20\t-1.0\n"
                "mic_a\t30\t40\t-2.0\n",
                encoding="utf-8",
            )
            parsed = parse_topaz_picks_star(p)
            self.assertEqual(len(parsed["rows"]), 2)
            self.assertEqual(parsed["rows"][0]["micrograph"], "mic_a")

    def test_header_only_picks_star_returns_empty_rows(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "picks.star"
            _write_star(p, [])
            parsed = parse_topaz_picks_star(p)
            self.assertEqual(parsed["rows"], [])


class TestNoGTQCRules(unittest.TestCase):
    def test_severe_under_picking_flag_when_few_total(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "picks.star"
            _write_star(p, ["m1.mrc 1 2 -1.0", "m1.mrc 3 4 -1.5"])
            qc = evaluate_picking_output(p)
            self.assertEqual(qc["qc_status"], "failed_qc")
            self.assertIn("severe_under_picking", qc["qc_flags"])
            self.assertEqual(qc["recommended_next_action"], "threshold_sweep_or_train_topaz")
            self.assertEqual(qc["n_picks"], 2)

    def test_empty_picks_file_flag(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "picks.star"
            _write_star(p, [])
            qc = evaluate_picking_output(p)
            self.assertIn("empty_picks_file", qc["qc_flags"])
            self.assertEqual(qc["qc_status"], "failed_qc")

    def test_passes_when_thresholds_relaxed(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "picks.star"
            _write_star(p, ["m1.mrc 1 2 -1.0", "m1.mrc 3 4 -1.5", "m2.mrc 5 6 -2.0"])
            qc = evaluate_picking_output(
                p,
                thresholds={
                    "min_total_picks": 1,
                    "min_mean_picks_per_micrograph": 1,
                    "max_zero_pick_micrograph_ratio": 1.0,
                },
            )
            self.assertEqual(qc["qc_status"], "passed")
            self.assertEqual(qc["recommended_next_action"], "continue_to_extract")
            self.assertEqual(qc["qc_flags"], [])

    def test_zero_pick_ratio_with_micrographs_star(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "picks.star"
            _write_star(
                p,
                [
                    "m1.mrc 1 2 -1.0",
                    "m1.mrc 3 4 -1.5",
                    "m2.mrc 5 6 -2.0",
                ],
            )
            mics = Path(tmp) / "micrographs.star"
            _write_micrographs_star(
                mics, ["m1.mrc", "m2.mrc", "m3.mrc", "m4.mrc"]
            )
            qc = evaluate_picking_output(p, micrographs_star_path=mics)
            self.assertEqual(qc["n_micrographs_total"], 4)
            self.assertEqual(qc["n_micrographs_with_picks"], 2)
            self.assertEqual(qc["zero_pick_micrograph_count"], 2)
            self.assertAlmostEqual(qc["zero_pick_micrograph_ratio"], 0.5)
            # 0.5 is not strictly greater than the default 0.5 max ratio, so
            # ``many_empty_micrographs`` should NOT fire here.
            self.assertNotIn("many_empty_micrographs", qc["qc_flags"])

    def test_many_empty_micrographs_flag(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "picks.star"
            _write_star(p, ["m1.mrc 1 2 -1.0"])
            mics = Path(tmp) / "micrographs.star"
            _write_micrographs_star(
                mics, [f"m{i}.mrc" for i in range(10)]
            )
            qc = evaluate_picking_output(p, micrographs_star_path=mics)
            self.assertGreater(qc["zero_pick_micrograph_ratio"], 0.5)
            self.assertIn("many_empty_micrographs", qc["qc_flags"])


class TestCountMicrographs(unittest.TestCase):
    def test_count_zero_when_no_loop(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "broken.star"
            p.write_text("nothing here\n", encoding="utf-8")
            self.assertEqual(count_micrographs_in_star(p), 0)


class TestGroundTruthEvaluation(unittest.TestCase):
    def test_precision_recall_f1_on_synthetic_example(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "picks.star"
            # Predictions perfectly aligned to GT for mic_001 -> tp=2.
            # mic_002 has 1 prediction far from any GT -> fp=1, fn=1.
            _write_star(
                p,
                [
                    "mic_001.mrc 100 100 -1.0",
                    "mic_001.mrc 200 200 -1.0",
                    "mic_002.mrc 9000 9000 -1.0",
                ],
            )
            gt_dir = Path(tmp) / "gt"
            gt_dir.mkdir()
            (gt_dir / "mic_001.csv").write_text(
                "x_coord,y_coord\n100,100\n200,200\n", encoding="utf-8"
            )
            (gt_dir / "mic_002.csv").write_text(
                "x_coord,y_coord\n10,10\n", encoding="utf-8"
            )
            qc = evaluate_picking_output(
                p,
                gt_coords_dir=gt_dir,
                matching_tolerance_px=20.0,
                thresholds={"min_total_picks": 1, "min_mean_picks_per_micrograph": 0.1},
            )
            ge = qc["ground_truth_evaluation"]
            self.assertEqual(ge["tp"], 2)
            self.assertEqual(ge["fp"], 1)
            self.assertEqual(ge["fn"], 1)
            self.assertAlmostEqual(ge["precision"], 2 / 3, places=4)
            self.assertAlmostEqual(ge["recall"], 2 / 3, places=4)
            self.assertAlmostEqual(ge["f1"], 2 / 3, places=4)
            # f1 well above 0.05 => no low_f1 flag.
            self.assertNotIn("low_f1_against_ground_truth", qc["qc_flags"])

    def test_low_f1_flag_added_when_below_threshold(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "picks.star"
            _write_star(p, ["mic_001.mrc 9000 9000 -1.0"])
            gt_dir = Path(tmp) / "gt"
            gt_dir.mkdir()
            (gt_dir / "mic_001.csv").write_text(
                "x_coord,y_coord\n100,100\n", encoding="utf-8"
            )
            qc = evaluate_picking_output(
                p,
                gt_coords_dir=gt_dir,
                matching_tolerance_px=20.0,
                thresholds={"min_total_picks": 1, "min_mean_picks_per_micrograph": 0.1},
            )
            self.assertEqual(qc["ground_truth_evaluation"]["f1"], 0.0)
            self.assertIn("low_f1_against_ground_truth", qc["qc_flags"])
            self.assertEqual(qc["qc_status"], "failed_qc")

    def test_coordinate_scale_maps_predictions_to_gt_space(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "picks.star"
            # Predictions in ds4 space (25 -> 100 after scale=4).
            _write_star(
                p,
                [
                    "mic_001.mrc 25 25 -1.0",
                    "mic_001.mrc 50 50 -1.0",
                ],
            )
            gt_dir = Path(tmp) / "gt"
            gt_dir.mkdir()
            (gt_dir / "mic_001.csv").write_text(
                "x_coord,y_coord\n100,100\n200,200\n", encoding="utf-8"
            )
            qc = evaluate_picking_output(
                p,
                gt_coords_dir=gt_dir,
                coordinate_scale=4.0,
                matching_tolerance_px=10.0,
                thresholds={"min_total_picks": 1, "min_mean_picks_per_micrograph": 0.1},
            )
            ge = qc["ground_truth_evaluation"]
            self.assertEqual(ge["tp"], 2)
            self.assertEqual(ge["fp"], 0)
            self.assertEqual(ge["fn"], 0)
            self.assertAlmostEqual(ge["precision"], 1.0)
            self.assertAlmostEqual(ge["recall"], 1.0)
            self.assertAlmostEqual(ge["f1"], 1.0)


class TestGTEvaluationOverAllMicrographs(unittest.TestCase):
    """Regression: GT recall/FN must count zero-pick micrographs too."""

    def test_recall_includes_zero_pick_micrographs(self) -> None:
        with TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks.star"
            # Predictions only for mic_a; mic_b and mic_c get nothing.
            _write_star(picks, ["mic_a.mrc 100 100 -1.0"])
            mics = Path(tmp) / "micrographs.star"
            _write_micrographs_star(mics, ["mic_a.mrc", "mic_b.mrc", "mic_c.mrc"])
            gt_dir = Path(tmp) / "gt"
            gt_dir.mkdir()
            (gt_dir / "mic_a.csv").write_text(
                "x_coord,y_coord\n100,100\n", encoding="utf-8"
            )
            (gt_dir / "mic_b.csv").write_text(
                "x_coord,y_coord\n10,10\n20,20\n", encoding="utf-8"
            )
            (gt_dir / "mic_c.csv").write_text(
                "x_coord,y_coord\n50,50\n60,60\n70,70\n", encoding="utf-8"
            )
            qc = evaluate_picking_output(
                picks,
                micrographs_star_path=mics,
                gt_coords_dir=gt_dir,
                matching_tolerance_px=20.0,
                thresholds={"min_total_picks": 1, "min_mean_picks_per_micrograph": 0.1},
            )
            ge = qc["ground_truth_evaluation"]
            self.assertEqual(ge["tp"], 1)
            self.assertEqual(ge["fp"], 0)
            self.assertEqual(ge["fn"], 5)
            self.assertEqual(ge["evaluated_micrographs"], 3)
            self.assertEqual(ge["gt_particles_total"], 6)
            self.assertEqual(ge["pred_particles_total"], 1)
            self.assertEqual(ge["gt_micrographs_available"], 3)
            # mic_a is the only one with both preds and GT.
            self.assertEqual(ge["matched_micrographs"], 1)
            self.assertAlmostEqual(ge["precision"], 1.0, places=4)
            self.assertAlmostEqual(ge["recall"], 1 / 6, places=4)
            # f1 = 2 * P * R / (P + R) = (2/6) / (7/6) = 2/7
            self.assertAlmostEqual(ge["f1"], 2 / 7, places=4)
            # f1 ≈ 0.286 is well above min_f1=0.05.
            self.assertNotIn("low_f1_against_ground_truth", qc["qc_flags"])

    def test_normalization_handles_full_paths_and_extensions(self) -> None:
        with TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks.star"
            # Topaz often emits image_name without extension.
            _write_star(picks, ["stack_0002_2x_SumCorr 100 100 -1.0"])
            mics = Path(tmp) / "micrographs.star"
            # micrographs.star carries a full-path .mrc.
            _write_micrographs_star(
                mics, ["/path/to/stack_0002_2x_SumCorr.mrc"]
            )
            gt_dir = Path(tmp) / "gt"
            gt_dir.mkdir()
            (gt_dir / "stack_0002_2x_SumCorr.csv").write_text(
                "x_coord,y_coord\n100,100\n", encoding="utf-8"
            )
            qc = evaluate_picking_output(
                picks,
                micrographs_star_path=mics,
                gt_coords_dir=gt_dir,
                matching_tolerance_px=10.0,
                thresholds={"min_total_picks": 1, "min_mean_picks_per_micrograph": 0.1},
            )
            ge = qc["ground_truth_evaluation"]
            self.assertEqual(ge["tp"], 1)
            self.assertEqual(ge["fp"], 0)
            self.assertEqual(ge["fn"], 0)
            self.assertEqual(ge["evaluated_micrographs"], 1)
            self.assertEqual(ge["matched_micrographs"], 1)

    def test_fallback_eval_set_unions_pred_and_gt_stems(self) -> None:
        """When micrographs.star is absent, eval set = preds ∪ GT csv stems."""
        with TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks.star"
            _write_star(picks, ["mic_a.mrc 100 100 -1.0"])
            gt_dir = Path(tmp) / "gt"
            gt_dir.mkdir()
            (gt_dir / "mic_a.csv").write_text(
                "x_coord,y_coord\n100,100\n", encoding="utf-8"
            )
            # GT-only micrograph with no predictions: must still contribute FN.
            (gt_dir / "mic_b.csv").write_text(
                "x_coord,y_coord\n50,50\n", encoding="utf-8"
            )
            qc = evaluate_picking_output(
                picks,
                gt_coords_dir=gt_dir,
                matching_tolerance_px=10.0,
                thresholds={"min_total_picks": 1, "min_mean_picks_per_micrograph": 0.1},
            )
            ge = qc["ground_truth_evaluation"]
            self.assertEqual(ge["evaluated_micrographs"], 2)
            self.assertEqual(ge["tp"], 1)
            self.assertEqual(ge["fp"], 0)
            self.assertEqual(ge["fn"], 1)
            self.assertEqual(ge["pred_particles_total"], 1)
            self.assertEqual(ge["gt_particles_total"], 2)
            self.assertEqual(ge["gt_micrographs_available"], 2)
            self.assertEqual(ge["matched_micrographs"], 1)


class TestSerializability(unittest.TestCase):
    def test_result_is_json_serializable(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "picks.star"
            _write_star(p, ["m1.mrc 1 2 -1.0"])
            qc = evaluate_picking_output(p)
            json.dumps(qc)  # Must not raise.
            # Defaults round-trip.
            self.assertEqual(qc["thresholds"]["min_f1"], DEFAULT_THRESHOLDS["min_f1"])


if __name__ == "__main__":
    unittest.main()
