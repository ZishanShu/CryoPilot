from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from backends.reconstruction.cryodrgn_ai_backend import build_command, collect_outputs


class TestCryodrgnAiCommandBuilder(unittest.TestCase):
    def test_star_input_produces_preprocess_and_abinit_commands(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            star = root / "particles.star"
            star.write_text("data_particles", encoding="utf-8")
            run_dir = root / "run"
            manifest = build_command(
                {
                    "particles_star_or_mrcs": str(star),
                    "datadir": str(root),
                    "apix": 1.5,
                    "box_size": 256,
                    "downsample_box": 128,
                    "zdim": 12,
                    "cryodrgn_executable": "cryodrgn",
                },
                run_dir,
            )
            names = [cmd["name"] for cmd in manifest["commands"]]
            self.assertEqual(names, ["downsample", "parse_ctf_star", "abinit"])
            self.assertIn("--datadir", manifest["commands"][0]["argv"])
            self.assertIn("--Apix", manifest["commands"][1]["argv"])
            self.assertIn("--ctf", manifest["commands"][2]["argv"])
            self.assertTrue((run_dir / "preprocess" / "preprocess_manifest.json").is_file())

    def test_mrcs_ctf_input_produces_only_abinit_command(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            mrcs = root / "particles.mrcs"
            ctf = root / "ctf.pkl"
            mrcs.write_bytes(b"m")
            ctf.write_bytes(b"c")
            manifest = build_command(
                {
                    "particles_star_or_mrcs": str(mrcs),
                    "ctf_metadata": str(ctf),
                    "zdim": 8,
                    "cryodrgn_executable": "cryodrgn",
                },
                root / "run",
            )
            self.assertEqual([cmd["name"] for cmd in manifest["commands"]], ["abinit"])
            self.assertEqual(manifest["commands"][0]["argv"][2], str(mrcs))
            self.assertIn(str(ctf), manifest["commands"][0]["argv"])

    def test_collect_outputs_records_logs_pkl_and_mrc_files(self) -> None:
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "stdout.log").write_text("out", encoding="utf-8")
            (run_dir / "stderr.log").write_text("err", encoding="utf-8")
            (run_dir / "abinit").mkdir()
            (run_dir / "abinit" / "weights.pkl").write_bytes(b"w")
            (run_dir / "abinit" / "z.pkl").write_bytes(b"z")
            (run_dir / "abinit" / "vol_001.mrc").write_bytes(b"v")
            (run_dir / "command_manifest.json").write_text(json.dumps({}), encoding="utf-8")
            artifacts = collect_outputs(run_dir)
            self.assertEqual(artifacts["run_dir"], str(run_dir))
            self.assertTrue(artifacts["pkl_files"])
            self.assertTrue(artifacts["mrc_files"])
            self.assertTrue(artifacts["model_weights"])
            self.assertTrue(artifacts["latent_embeddings"])
            self.assertTrue(artifacts["sampled_volumes"])


if __name__ == "__main__":
    unittest.main()
