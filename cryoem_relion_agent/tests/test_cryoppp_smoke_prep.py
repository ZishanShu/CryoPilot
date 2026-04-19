from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from agent.data_prep.cryoppp_smoke_prep import prepare_cryoppp_topaz_smoke


class TestCryoPPPSmokePrep(unittest.TestCase):
    def test_prepare_outputs_manifest_and_star(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset_a"
            mg_dir = dataset / "motion_corrected_micrographs"
            gt_dir = dataset / "ground_truth" / "particle_coordinates"
            meta_dir = dataset / "metadata"
            mg_dir.mkdir(parents=True)
            gt_dir.mkdir(parents=True)
            meta_dir.mkdir(parents=True)
            (meta_dir / "meta.json").write_text('{"pixel_size_angstrom": 1.5, "particle_diameter_px": 100}', encoding="utf-8")
            for idx in range(4):
                name = f"mic_{idx:03d}.mrc"
                (mg_dir / name).write_text("m", encoding="utf-8")
                (gt_dir / f"mic_{idx:03d}.csv").write_text("10,10\n20,20\n", encoding="utf-8")
            manifest = prepare_cryoppp_topaz_smoke(
                dataset_root=dataset,
                run_root=root / "real_runs" / "cryoppp_topaz_smoke",
                dataset_id="ds1",
                sample_size=3,
                random_seed=42,
            )
            self.assertEqual(manifest["dataset_id"], "ds1")
            self.assertEqual(len(manifest["selected_micrographs"]), 3)
            self.assertEqual(len(manifest["gt_coordinate_files"]), 3)
            self.assertEqual(manifest["gt_overlap_count"], 4)
            self.assertEqual(manifest["requested_sample_size"], 3)
            self.assertEqual(manifest["effective_sample_size"], 3)
            self.assertTrue(manifest["gt_covered_sample"])
            self.assertEqual(len(manifest["matched_gt_coordinate_files"]), 3)
            star_path = Path(manifest["output_paths"]["micrographs_star_path"])
            self.assertTrue(star_path.is_file())
            self.assertIn("_rlnMicrographName", star_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["estimated_particle_diameter_angstrom"], 150.0)

    def test_fallback_to_micrographs_dir_and_default_gt_overlap_sampling(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset_b"
            mg_dir = dataset / "micrographs"
            gt_dir = dataset / "ground_truth" / "particle_coordinates"
            mg_dir.mkdir(parents=True)
            gt_dir.mkdir(parents=True)
            for idx in range(5):
                Image.new("L", (4, 4), color=idx + 10).save(
                    mg_dir / f"stack_{idx:04d}_2x_SumCorr.jpg",
                    format="JPEG",
                )
            # 仅给其中两个提供 GT，验证默认交集抽样
            (gt_dir / "stack_0001_2x_SumCorr.csv").write_text("10,10\n", encoding="utf-8")
            (gt_dir / "stack_0003_2x_SumCorr.csv").write_text("12,12\n", encoding="utf-8")

            manifest = prepare_cryoppp_topaz_smoke(
                dataset_root=dataset,
                run_root=root / "real_runs" / "cryoppp_topaz_smoke",
                dataset_id="ds2",
                sample_size=5,
                sampling_strategy="first_n",
            )
            self.assertEqual(manifest["requested_sample_size"], 5)
            self.assertEqual(manifest["gt_overlap_count"], 2)
            self.assertEqual(manifest["effective_sample_size"], 2)
            self.assertEqual(len(manifest["selected_micrographs"]), 2)
            self.assertEqual(len(manifest["gt_coordinate_files"]), 2)
            self.assertTrue(all(str(p).endswith(".csv") for p in manifest["gt_coordinate_files"]))
            gt_coords_dir = Path(manifest["output_paths"]["gt_coords_dir"])
            self.assertEqual(len(list(gt_coords_dir.glob("*.csv"))), 2)
            self.assertTrue(manifest["conversion_applied"])
            self.assertEqual(manifest["micrograph_source_format"], "jpg")
            self.assertEqual(manifest["micrograph_runtime_format"], "png")

    def test_sample_size_larger_than_overlap_auto_reduces(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset_c"
            mg_dir = dataset / "micrographs"
            gt_dir = dataset / "ground_truth" / "particle_coordinates"
            mg_dir.mkdir(parents=True)
            gt_dir.mkdir(parents=True)
            for idx in range(3):
                (mg_dir / f"mic_{idx:03d}.mrc").write_text("m", encoding="utf-8")
            (gt_dir / "mic_000.csv").write_text("1,1\n", encoding="utf-8")
            manifest = prepare_cryoppp_topaz_smoke(
                dataset_root=dataset,
                run_root=root / "real_runs" / "cryoppp_topaz_smoke",
                dataset_id="ds3",
                sample_size=10,
            )
            self.assertEqual(manifest["gt_overlap_count"], 1)
            self.assertEqual(manifest["effective_sample_size"], 1)
            self.assertEqual(len(manifest["selected_micrographs"]), 1)
            self.assertEqual(len(manifest["matched_gt_coordinate_files"]), 1)

    def test_jpg_micrographs_are_converted_to_png_and_star_points_to_png(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset_jpg"
            mg_dir = dataset / "micrographs"
            gt_dir = dataset / "ground_truth" / "particle_coordinates"
            mg_dir.mkdir(parents=True)
            gt_dir.mkdir(parents=True)
            for idx in range(2):
                img_path = mg_dir / f"stack_{idx:04d}_2x_SumCorr.jpg"
                Image.new("L", (4, 4), color=idx + 10).save(img_path, format="JPEG")
                (gt_dir / f"stack_{idx:04d}_2x_SumCorr.csv").write_text("1,1\n", encoding="utf-8")

            manifest = prepare_cryoppp_topaz_smoke(
                dataset_root=dataset,
                run_root=root / "real_runs" / "cryoppp_topaz_smoke",
                dataset_id="ds_jpg",
                sample_size=2,
                sampling_strategy="first_n",
            )
            self.assertTrue(manifest["conversion_applied"])
            self.assertEqual(manifest["micrograph_source_format"], "jpg")
            self.assertEqual(manifest["micrograph_runtime_format"], "png")
            self.assertEqual(len(manifest["converted_micrographs"]), 2)
            self.assertTrue(all(str(p).endswith(".png") for p in manifest["selected_micrographs"]))
            star_path = Path(manifest["output_paths"]["micrographs_star_path"])
            star_text = star_path.read_text(encoding="utf-8")
            self.assertIn(".png", star_text)
            self.assertNotIn(".jpg", star_text)
            self.assertEqual(len(manifest["gt_coordinate_files"]), 2)

    def test_non_jpg_input_keeps_runtime_format_unchanged(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset_png"
            mg_dir = dataset / "micrographs"
            gt_dir = dataset / "ground_truth" / "particle_coordinates"
            mg_dir.mkdir(parents=True)
            gt_dir.mkdir(parents=True)
            for idx in range(2):
                img_path = mg_dir / f"stack_{idx:04d}.png"
                Image.new("L", (4, 4), color=idx + 10).save(img_path, format="PNG")
                (gt_dir / f"stack_{idx:04d}.csv").write_text("1,1\n", encoding="utf-8")
            manifest = prepare_cryoppp_topaz_smoke(
                dataset_root=dataset,
                run_root=root / "real_runs" / "cryoppp_topaz_smoke",
                dataset_id="ds_png",
                sample_size=2,
                sampling_strategy="first_n",
            )
            self.assertFalse(manifest["conversion_applied"])
            self.assertEqual(manifest["micrograph_source_format"], "png")
            self.assertEqual(manifest["micrograph_runtime_format"], "png")
            self.assertEqual(manifest["converted_micrographs"], [])


if __name__ == "__main__":
    unittest.main()

