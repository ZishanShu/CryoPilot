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
    run as backend_run,
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
            "/run/abinit/config.yaml",
            "/run/representative_z.txt",
            "/run/sampled_volumes",
        )
        self.assertIn("eval_vol", argv)
        self.assertIn("--zfile", argv)
        self.assertNotIn("--z", argv)
        zfile_idx = argv.index("--zfile")
        self.assertEqual(argv[zfile_idx + 1], "/run/representative_z.txt")

    def test_eval_vol_command_uses_dash_c_not_long_config(self) -> None:
        """cryoDRGN 3.4 eval_vol uses ``-c`` for the config flag."""
        argv = build_eval_vol_command(
            "cryodrgn",
            "/run/abinit/weights.pkl",
            "/run/abinit/config.yaml",
            "/run/representative_z.txt",
            "/run/sampled_volumes",
        )
        self.assertIn("-c", argv)
        c_idx = argv.index("-c")
        self.assertEqual(argv[c_idx + 1], "/run/abinit/config.yaml")
        # And must NOT use --config (long form) per spec.
        self.assertNotIn("--config", argv)

    def test_eval_vol_command_uses_config_yaml_not_pkl(self) -> None:
        """eval_vol argv must reference config.yaml; never config.pkl."""
        argv = build_eval_vol_command(
            "cryodrgn",
            "/run/abinit/weights.pkl",
            "/run/abinit/config.yaml",
            "/run/representative_z.txt",
            "/run/sampled_volumes",
        )
        self.assertNotIn("/run/abinit/config.pkl", argv)
        self.assertIn("/run/abinit/config.yaml", argv)

    def test_eval_vol_command_includes_prefix_vol(self) -> None:
        """Sampled volumes are written as vol_NNN.mrc via --prefix vol_."""
        argv = build_eval_vol_command(
            "cryodrgn", "w.pkl", "c.yaml", "rep_z.txt", "out/",
        )
        self.assertIn("--prefix", argv)
        prefix_idx = argv.index("--prefix")
        self.assertEqual(argv[prefix_idx + 1], "vol_")

    def test_eval_vol_command_optional_flags(self) -> None:
        argv = build_eval_vol_command(
            "cryodrgn",
            "weights.pkl",
            "config.yaml",
            "rep_z.txt",
            "out/",
            apix=1.35,
            box_size=256,
            device="0",
        )
        self.assertIn("--Apix", argv)
        self.assertIn("1.35", argv)
        # cryoDRGN 3.4 uses -D for box size, not --box.
        self.assertIn("-D", argv)
        self.assertNotIn("--box", argv)
        d_idx = argv.index("-D")
        self.assertEqual(argv[d_idx + 1], "256")
        self.assertIn("--device", argv)
        self.assertIn("0", argv)

    def test_dry_run_command_manifest_includes_eval_vol_when_sample_volumes_true(self) -> None:
        """When sample_volumes=True, build_command puts eval_vol in the manifest
        and the planned argv uses cryoDRGN 3.4 syntax (-c, -D, --zfile, config.yaml)."""
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
                        "eval_vol_apix": 1.2,
                        "eval_vol_box_size": 256,
                    },
                    root / "run",
                )
            names = [cmd["name"] for cmd in manifest["commands"]]
            self.assertIn("eval_vol", names)
            ev_cmd = next(c for c in manifest["commands"] if c["name"] == "eval_vol")
            argv = ev_cmd["argv"]
            self.assertIn("--zfile", argv)
            self.assertNotIn("--z", argv)
            self.assertIn("-c", argv)
            self.assertNotIn("--config", argv)
            self.assertIn("-D", argv)
            self.assertNotIn("--box", argv)
            self.assertIn("--prefix", argv)
            # config path must be config.yaml, never config.pkl.
            self.assertTrue(any("config.yaml" in str(x) for x in argv))
            self.assertFalse(any("config.pkl" in str(x) for x in argv))

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


# ---------------------------------------------------------------------------
# Closed-loop tests: run() refreshes artifacts after eval_vol; update_state
# populates the canonical last_execution dict.
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _make_closed_loop_subprocess(n_volumes: int = 5):
    """subprocess.run side_effect that materializes abinit + eval_vol outputs.

    - Help probes (``-h``) succeed.
    - abinit_het / abinit / abinit_homo: succeed and write weights.pkl, z.pkl,
      config.yaml under the ``-o`` directory.
    - eval_vol: succeed and write vol_NNN.mrc under the ``-o`` directory.
    """

    def _side_effect(cmd, **_kwargs):
        if not isinstance(cmd, list):
            cmd = list(cmd)
        if len(cmd) >= 3 and cmd[2] == "-h":
            return _FakeProc(0)
        sub = cmd[1] if len(cmd) > 1 else ""
        if sub in {"abinit", "abinit_het", "abinit_homo"}:
            if "-o" in cmd:
                out_dir = Path(cmd[cmd.index("-o") + 1])
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "weights.pkl").write_bytes(b"w")
                rng = np.random.default_rng(0)
                z = rng.standard_normal((50, 8)).astype(np.float32)
                with open(out_dir / "z.pkl", "wb") as fh:
                    pickle.dump(z, fh)
                (out_dir / "config.yaml").write_text("model: het\n", encoding="utf-8")
            return _FakeProc(0)
        if sub == "eval_vol":
            if "-o" in cmd:
                out_dir = Path(cmd[cmd.index("-o") + 1])
                out_dir.mkdir(parents=True, exist_ok=True)
                for i in range(n_volumes):
                    (out_dir / f"vol_{i:03d}.mrc").write_bytes(b"v")
            return _FakeProc(0)
        # Other preprocess/help cases: succeed silently.
        return _FakeProc(0)

    return _side_effect


class TestRunRefreshesArtifactsAfterEvalVol(unittest.TestCase):
    def test_run_returns_5_sampled_volumes_and_5_candidate_maps(self) -> None:
        """End-to-end run() with sample_volumes=True refreshes outputs after
        eval_vol so the result carries 5 .mrc paths and 5 candidate_maps."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            mrcs = root / "particles.mrcs"
            ctf = root / "ctf.pkl"
            mrcs.write_bytes(b"m")
            ctf.write_bytes(b"c")
            exe = _fake_cryodrgn(root)
            run_dir = root / "run"
            params = {
                "particles_star_or_mrcs": str(mrcs),
                "ctf_metadata": str(ctf),
                "cryodrgn_executable": exe,
                "execution_mode": "real",
                "sample_volumes": True,
                "n_sampled_volumes": 5,
                "eval_vol_apix": 1.2,
                "eval_vol_box_size": 256,
            }
            with patch(
                "backends.reconstruction.cryodrgn_ai_backend.subprocess.run",
                side_effect=_make_closed_loop_subprocess(n_volumes=5),
            ), patch(
                "backends.reconstruction.cryodrgn_ai_backend._gpu_available",
                return_value=True,
            ):
                result = backend_run(params, run_dir)

            self.assertEqual(result["status"], "completed")
            artifacts = result["output_artifacts"]
            self.assertEqual(len(artifacts["sampled_volumes"]), 5)
            self.assertEqual(len(artifacts["mrc_files"]), 5)
            self.assertTrue(artifacts.get("representative_z_path"))
            for p in artifacts["sampled_volumes"]:
                self.assertTrue(p.endswith(".mrc"))
                self.assertIn("sampled_volumes", p)

            self.assertIn("candidate_maps", result)
            self.assertEqual(len(result["candidate_maps"]), 5)
            for m in result["candidate_maps"]:
                self.assertEqual(m["source_backend"], "cryodrgn_ai")
                self.assertEqual(m["qc_status"], "pending")
                self.assertEqual(m["recommended_next_action"], "route_to_model_building")
                self.assertTrue(m["map_path"].endswith(".mrc"))

            ev = result.get("eval_vol_result") or {}
            self.assertEqual(ev.get("status"), "completed")
            ev_argv = ev.get("argv") or []
            self.assertIn("-c", ev_argv)
            self.assertIn("-D", ev_argv)
            self.assertIn("--zfile", ev_argv)
            self.assertIn("--prefix", ev_argv)
            self.assertFalse(any("config.pkl" in str(x) for x in ev_argv))


class TestUpdateStateClosedLoop(unittest.TestCase):
    def test_update_state_writes_last_execution_with_sampled_volumes(self) -> None:
        """update_state on dict state populates state['last_execution'] with
        candidate_maps, sampled_volumes, and output_artifacts.{sampled_volumes,mrc_files}."""
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            sv_dir = run_dir / "sampled_volumes"
            sv_dir.mkdir(parents=True)
            (run_dir / "abinit").mkdir(parents=True)
            (run_dir / "abinit" / "weights.pkl").write_bytes(b"w")
            (run_dir / "representative_z.txt").write_text("0 0 0\n", encoding="utf-8")
            for i in range(5):
                (sv_dir / f"vol_{i:03d}.mrc").write_bytes(b"v")
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
            self.assertEqual(len(state["candidate_maps"]), 5)

            le = state["last_execution"]
            self.assertEqual(len(le["candidate_maps"]), 5)
            self.assertEqual(len(le["sampled_volumes"]), 5)
            self.assertEqual(len(le["output_artifacts"]["sampled_volumes"]), 5)
            self.assertEqual(len(le["output_artifacts"]["mrc_files"]), 5)


if __name__ == "__main__":
    unittest.main()
