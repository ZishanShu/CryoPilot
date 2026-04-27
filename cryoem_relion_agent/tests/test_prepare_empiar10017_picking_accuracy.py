"""Tests for the EMPIAR-10017 picking-accuracy prepare script."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from scripts.prepare_empiar10017_picking_accuracy import (
    _pair_micrographs_with_coords,
    _parse_coord_file,
    prepare_empiar10017_picking_accuracy,
)


def _make_dataset(root: Path, n_pairs: int) -> Path:
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    for i in range(n_pairs):
        stem = f"img_{i:03d}"
        (data / f"{stem}.mrc").write_bytes(b"\x00" * 16)
        coord_path = data / f"{stem}.coord"
        coord_path.write_text(
            f"# header line\n{10 + i}.0 {20 + i}.0\n{30 + i}.0 {40 + i}.0\n",
            encoding="utf-8",
        )
    return data


class TestParseCoordFile(unittest.TestCase):
    def test_skips_comments_and_non_numeric(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.coord"
            p.write_text("# comment\n\n1.5 2.5 99\nfoo bar\n3 4\n", encoding="utf-8")
            coords = _parse_coord_file(p)
            self.assertEqual(coords, [(1.5, 2.5), (3.0, 4.0)])


class TestPairing(unittest.TestCase):
    def test_pairs_only_when_both_files_exist(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            (data / "a.mrc").write_bytes(b"\x00")
            (data / "a.coord").write_text("1 2\n", encoding="utf-8")
            (data / "b.mrc").write_bytes(b"\x00")  # no coord
            (data / "c.coord").write_text("3 4\n", encoding="utf-8")  # no mrc
            pairs = _pair_micrographs_with_coords(data)
            self.assertEqual([m.stem for m, _c in pairs], ["a"])


class TestPrepareScript(unittest.TestCase):
    def test_three_pairs_generates_split_manifest_and_gt(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_dataset(root, n_pairs=3)
            run_root = root / "run"
            manifest = prepare_empiar10017_picking_accuracy(
                dataset_root=root,
                run_root=run_root,
                train_size=1,
                dev_size=1,
                test_size=1,
                random_seed=42,
            )
            input_dir = run_root / "input"
            self.assertTrue((input_dir / "split_manifest.json").is_file())
            self.assertTrue((input_dir / "gt_coordinates.tsv").is_file())
            for label in ("train", "dev", "test"):
                star = input_dir / f"micrographs_{label}.star"
                self.assertTrue(star.is_file(), f"missing {star}")
                txt = star.read_text(encoding="utf-8")
                self.assertIn("data_micrographs", txt)
                self.assertIn("_rlnMicrographName #1", txt)
            self.assertEqual(manifest["n_paired"], 3)
            self.assertEqual(manifest["splits"]["train"]["count"], 1)
            self.assertEqual(manifest["splits"]["dev"]["count"], 1)
            self.assertEqual(manifest["splits"]["test"]["count"], 1)
            self.assertEqual(manifest["random_seed"], 42)

            saved = json.loads((input_dir / "split_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["n_paired"], 3)

            gt_text = (input_dir / "gt_coordinates.tsv").read_text(encoding="utf-8")
            header = gt_text.splitlines()[0].split("\t")
            for col in ("split", "image_name", "micrograph_path", "coord_path", "x_coord", "y_coord"):
                self.assertIn(col, header)
            # each .coord has 2 numeric rows; 3 micrographs -> 6 GT rows
            self.assertEqual(len(gt_text.strip().splitlines()) - 1, 6)

    def test_case_yaml_contains_required_sections(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_dataset(root, n_pairs=4)
            run_root = root / "run"
            prepare_empiar10017_picking_accuracy(
                dataset_root=root,
                run_root=run_root,
                train_size=2,
                dev_size=1,
                test_size=1,
                random_seed=42,
            )
            case = yaml.safe_load((run_root / "input" / "case.yaml").read_text(encoding="utf-8"))
            self.assertEqual(case["case_id"], "P1_topaz_empiar10017_picking_accuracy")
            self.assertEqual(case["case_type"], "picking_accuracy")
            self.assertEqual(case["evaluation"]["matching_radius_px"], 30)
            self.assertEqual(case["evaluation"]["radius_sensitivity_px"], [20, 30, 40])
            self.assertIn("split_manifest", case["input"])
            self.assertIn("gt_coordinates", case["input"])

    def test_seed_makes_split_deterministic(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_dataset(root, n_pairs=10)
            run_a = root / "ra"
            run_b = root / "rb"
            m_a = prepare_empiar10017_picking_accuracy(
                dataset_root=root, run_root=run_a, train_size=4, dev_size=2, test_size=2, random_seed=42
            )
            m_b = prepare_empiar10017_picking_accuracy(
                dataset_root=root, run_root=run_b, train_size=4, dev_size=2, test_size=2, random_seed=42
            )
            for label in ("train", "dev", "test"):
                names_a = [item["image_name"] for item in m_a["splits"][label]["items"]]
                names_b = [item["image_name"] for item in m_b["splits"][label]["items"]]
                self.assertEqual(names_a, names_b)

    def test_oversize_split_raises(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_dataset(root, n_pairs=2)
            with self.assertRaises(ValueError):
                prepare_empiar10017_picking_accuracy(
                    dataset_root=root, run_root=root / "r", train_size=2, dev_size=1, test_size=1
                )


if __name__ == "__main__":
    unittest.main()
