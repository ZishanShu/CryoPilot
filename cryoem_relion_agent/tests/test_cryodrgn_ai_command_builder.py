from __future__ import annotations

import json
import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from backends.reconstruction.cryodrgn_ai_backend import (
    build_command,
    build_eval_vol_command,
    collect_outputs,
    sample_representative_latents,
    update_state,
)


def _fake_cryodrgn(root: Path) -> str:
    exe = root / "cryodrgn"
    exe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    exe.chmod(0o755)
    return str(exe)


def _fake_abinit_help(cmd, **kwargs):
    """subprocess.run stub: abinit_het -h succeeds, other commands fail."""
    import subprocess
    proc = type("P", (), {"returncode": 0, "stdout": "help", "stderr": ""})()
    if len(cmd) >= 3 and cmd[1] in {"abinit", "abinit_het", "abinit_homo"} and cmd[2] == "-h":
        return proc
    proc2 = type("P", (), {"returncode": 1, "stdout": "", "stderr": ""})()
    return proc2


class TestCryodrgnAiCommandBuilder(unittest.TestCase):
    def test_star_input_produces_preprocess_and_abinit_commands(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            star = root / "particles.star"
            star.write_text("data_particles", encoding="utf-8")
            exe = _fake_cryodrgn(root)
            run_dir = root / "run"
            with patch("backends.reconstruction.cryodrgn_ai_backend.subprocess.run", side_effect=_fake_abinit_help):
                manifest = build_command(
                    {
                        "particles_star_or_mrcs": str(star),
                        "datadir": str(root),
                        "apix": 1.5,
                        "box_size": 256,
                        "downsample_box": 128,
                        "zdim": 12,
                        "cryodrgn_executable": exe,
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
            exe = _fake_cryodrgn(root)
            with patch("backends.reconstruction.cryodrgn_ai_backend.subprocess.run", side_effect=_fake_abinit_help):
                manifest = build_command(
                    {
                        "particles_star_or_mrcs": str(mrcs),
                        "ctf_metadata": str(ctf),
                        "zdim": 8,
                        "cryodrgn_executable": exe,
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
            # sampled_volumes/ subdir is where eval_vol writes its output
            sv_dir = run_dir / "sampled_volumes"
            sv_dir.mkdir()
            (sv_dir / "vol_000.mrc").write_bytes(b"v")
            (run_dir / "command_manifest.json").write_text(json.dumps({}), encoding="utf-8")
            artifacts = collect_outputs(run_dir)
            self.assertEqual(artifacts["run_dir"], str(run_dir))
            self.assertTrue(artifacts["pkl_files"])
            self.assertTrue(artifacts["mrc_files"])
            self.assertTrue(artifacts["model_weights"])
            self.assertTrue(artifacts["latent_embeddings"])
            self.assertTrue(artifacts["sampled_volumes"])

    # ------------------------------------------------------------------
    # New tests for eval_vol / representative latent / sampled volumes
    # ------------------------------------------------------------------

    def test_representative_z_generation_from_synthetic_z_pkl(self) -> None:
        """sample_representative_latents writes n_volumes rows to representative_z.txt."""
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            z_pkl = Path(tmp) / "z.pkl"
            rng = np.random.default_rng(42)
            z = rng.standard_normal((100, 8)).astype(np.float32)
            with open(z_pkl, "wb") as fh:
                pickle.dump(z, fh)
            rep_z_path = sample_representative_latents(run_dir, z_pkl, n_volumes=5)
            self.assertTrue(rep_z_path.is_file())
            loaded = np.loadtxt(str(rep_z_path))
            self.assertEqual(loaded.shape, (5, 8))

    def test_representative_z_fewer_samples_than_n_volumes(self) -> None:
        """When n_samples < n_volumes all rows are kept."""
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            z_pkl = Path(tmp) / "z.pkl"
            z = np.arange(12, dtype=float).reshape(3, 4)
            with open(z_pkl, "wb") as fh:
                pickle.dump(z, fh)
            rep_z_path = sample_representative_latents(run_dir, z_pkl, n_volumes=5)
            loaded = np.loadtxt(str(rep_z_path))
            self.assertEqual(loaded.shape, (3, 4))

    def test_eval_vol_command_uses_zfile_not_z(self) -> None:
        """build_eval_vol_command must use --zfile, not --z."""
        argv = build_eval_vol_command(
            "cryodrgn",
            "/run/abinit/weights.pkl",
            "/run/abinit/config.pkl",
            "/run/representative_z.txt",
            "/run/sampled_volumes",
        )
        self.assertIn("eval_vol", argv)
        self.assertIn("--zfile", argv)
        self.assertNotIn("--z", argv)
        zfile_idx = argv.index("--zfile")
        self.assertEqual(argv[zfile_idx + 1], "/run/representative_z.txt")

    def test_eval_vol_command_optional_flags(self) -> None:
        argv = build_eval_vol_command(
            "cryodrgn",
            "weights.pkl",
            "config.pkl",
            "rep_z.txt",
            "out/",
            apix=1.35,
            box_size=256,
            device="0",
        )
        self.assertIn("--Apix", argv)
        self.assertIn("1.35", argv)
        self.assertIn("--box", argv)
        self.assertIn("256", argv)
        self.assertIn("--device", argv)
        self.assertIn("0", argv)

    def test_dry_run_command_manifest_includes_eval_vol_when_sample_volumes_true(self) -> None:
        """When sample_volumes=True, build_command puts eval_vol in the manifest."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            mrcs = root / "particles.mrcs"
            ctf = root / "ctf.pkl"
            mrcs.write_bytes(b"m")
            ctf.write_bytes(b"c")
            exe = _fake_cryodrgn(root)
            with patch("backends.reconstruction.cryodrgn_ai_backend.subprocess.run", side_effect=_fake_abinit_help):
                manifest = build_command(
                    {
                        "particles_star_or_mrcs": str(mrcs),
                        "ctf_metadata": str(ctf),
                        "zdim": 8,
                        "cryodrgn_executable": exe,
                        "sample_volumes": True,
                        "n_sampled_volumes": 5,
                    },
                    root / "run",
                )
            names = [cmd["name"] for cmd in manifest["commands"]]
            self.assertIn("eval_vol", names)
            ev_cmd = next(c for c in manifest["commands"] if c["name"] == "eval_vol")
            self.assertIn("--zfile", ev_cmd["argv"])
            self.assertNotIn("--z", ev_cmd["argv"])

    def test_dry_run_manifest_no_eval_vol_when_sample_volumes_false(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            mrcs = root / "p.mrcs"
            ctf = root / "c.pkl"
            mrcs.write_bytes(b"m")
            ctf.write_bytes(b"c")
            exe = _fake_cryodrgn(root)
            with patch("backends.reconstruction.cryodrgn_ai_backend.subprocess.run", side_effect=_fake_abinit_help):
                manifest = build_command(
                    {"particles_star_or_mrcs": str(mrcs), "ctf_metadata": str(ctf), "cryodrgn_executable": exe},
                    root / "run",
                )
            names = [cmd["name"] for cmd in manifest["commands"]]
            self.assertNotIn("eval_vol", names)

    def test_collect_outputs_finds_sampled_volumes_subdir_mrc_files(self) -> None:
        """collect_outputs['sampled_volumes'] only includes files from sampled_volumes/ subdir."""
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "abinit").mkdir()
            (run_dir / "abinit" / "weights.pkl").write_bytes(b"w")
            (run_dir / "abinit" / "z.pkl").write_bytes(b"z")
            # These MRC files live outside sampled_volumes/ — should NOT appear in sampled_volumes
            (run_dir / "abinit" / "vol_raw.mrc").write_bytes(b"r")
            sv_dir = run_dir / "sampled_volumes"
            sv_dir.mkdir()
            (sv_dir / "vol_000.mrc").write_bytes(b"0")
            (sv_dir / "vol_001.mrc").write_bytes(b"1")
            (sv_dir / "vol_002.mrc").write_bytes(b"2")
            (run_dir / "command_manifest.json").write_text("{}", encoding="utf-8")
            artifacts = collect_outputs(run_dir)
            # sampled_volumes is scoped to sampled_volumes/ subdir
            self.assertEqual(len(artifacts["sampled_volumes"]), 3)
            for p in artifacts["sampled_volumes"]:
                self.assertIn("sampled_volumes", p)
            # mrc_files includes everything
            self.assertEqual(len(artifacts["mrc_files"]), 4)

    def test_update_state_writes_candidate_maps_with_metadata(self) -> None:
        """update_state writes candidate_maps as dicts with required fields."""
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            sv_dir = run_dir / "sampled_volumes"
            sv_dir.mkdir(parents=True)
            (run_dir / "abinit").mkdir(parents=True)
            (run_dir / "abinit" / "weights.pkl").write_bytes(b"w")
            rep_z = run_dir / "representative_z.txt"
            rep_z.write_text("0 0 0 0 0 0 0 0\n", encoding="utf-8")
            (sv_dir / "vol_000.mrc").write_bytes(b"0")
            (sv_dir / "vol_001.mrc").write_bytes(b"1")
            (run_dir / "command_manifest.json").write_text("{}", encoding="utf-8")
            artifacts = collect_outputs(run_dir)
            result = {
                "action_name": "run_cryodrgn_ai",
                "status": "completed",
                "output_artifacts": artifacts,
            }
            state: dict = {}
            state = update_state(state, result)
            self.assertIn("candidate_maps", state)
            maps = state["candidate_maps"]
            self.assertEqual(len(maps), 2)
            for m in maps:
                self.assertEqual(m["source_backend"], "cryodrgn_ai")
                self.assertIn("map_path", m)
                self.assertEqual(m["qc_status"], "pending")
                self.assertEqual(m["recommended_next_action"], "route_to_model_building")
                self.assertTrue(m["map_path"].endswith(".mrc"))


if __name__ == "__main__":
    unittest.main()
