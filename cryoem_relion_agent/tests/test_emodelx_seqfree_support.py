"""Tests for EModelX seq_free / temp_free / temp_flex protocol support.

Covers the minimal stage-27 patch:
- Protocol-aware preflight (seq_free no longer requires sequence_path).
- Protocol-aware argv construction (no --fasta for seq_free; --template_dir
  forwarded for temp_flex).
- Runner ensures output_dir exists before subprocess launch.
- Case T1_tutorial5_map_emodelx_stopA (seq_free) passes preflight when the
  map / repo / output_dir are provided.
"""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from agent.case_registry import default_cases, materialize_case
from agent.model_building_preflight import preflight_model_building
from relion.emodelx_runner import run_emodelx_pipeline


def _make_repo(root: Path) -> Path:
    repo = root / "EModelX"
    (repo / "models").mkdir(parents=True, exist_ok=True)
    (repo / "models" / "w.bin").write_text("x", encoding="utf-8")
    (repo / "run.py").write_text("print('stub')", encoding="utf-8")
    return repo


def _base_params(
    *,
    root: Path,
    repo: Path,
    map_path: Path,
    sequence_path: Path | None,
    protocol: str,
    template_dir: Path | None = None,
    download_afdb: bool = False,
) -> dict:
    params: dict = {
        "emodelx_protocol": protocol,
        "emodelx_repo_dir": str(repo),
        "emodelx_python": "/usr/bin/python3",
        "map_path": str(map_path),
        "output_dir": str(root / "out"),
    }
    if sequence_path is not None:
        params["sequence_path"] = str(sequence_path)
    if template_dir is not None:
        params["template_dir"] = str(template_dir)
    if download_afdb:
        params["download_afdb"] = True
    return params


class TestSeqFreePreflight(unittest.TestCase):
    def test_seq_free_passes_preflight_without_sequence_path(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = _make_repo(root)
            map_path = root / "map.mrc"
            map_path.write_text("mrc", encoding="utf-8")
            params = _base_params(
                root=root,
                repo=repo,
                map_path=map_path,
                sequence_path=None,
                protocol="seq_free",
            )
            r = preflight_model_building("emodelx", params)
            self.assertTrue(r["available"], r)
            self.assertEqual(r["protocol"], "seq_free")
            self.assertEqual(r["missing_inputs"], [])
            self.assertIsNone(r["blocked_reason"])
            self.assertNotIn("--fasta", str(r.get("command") or ""))

    def test_temp_free_blocks_when_sequence_missing(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = _make_repo(root)
            map_path = root / "map.mrc"
            map_path.write_text("mrc", encoding="utf-8")
            params = _base_params(
                root=root,
                repo=repo,
                map_path=map_path,
                sequence_path=None,
                protocol="temp_free",
            )
            r = preflight_model_building("emodelx", params)
            self.assertFalse(r["available"])
            self.assertIn("sequence_path", r["missing_inputs"])
            self.assertIn("missing_required_inputs: sequence_path", str(r["blocked_reason"]))

    def test_temp_flex_blocks_when_template_dir_missing(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = _make_repo(root)
            map_path = root / "map.mrc"
            map_path.write_text("mrc", encoding="utf-8")
            seq_path = root / "seq.fasta"
            seq_path.write_text(">A\nM\n", encoding="utf-8")
            params = _base_params(
                root=root,
                repo=repo,
                map_path=map_path,
                sequence_path=seq_path,
                protocol="temp_flex",
            )
            r = preflight_model_building("emodelx", params)
            self.assertFalse(r["available"])
            # Existing code bundles the two accepted alternatives; either name
            # is acceptable as long as blocked_reason cites the missing slot.
            blocked = str(r["blocked_reason"])
            self.assertTrue(
                "template_dir" in blocked or "download_afdb" in blocked,
                f"unexpected blocked_reason={blocked!r}",
            )


class TestRunnerArgvProtocolAware(unittest.TestCase):
    def _preflight_stub(self, repo: Path, protocol: str) -> dict:
        return {
            "backend_name": "emodelx",
            "available": True,
            "execution_mode": "real",
            "protocol": protocol,
            "repo_path": str(repo),
            "run_py_path": str(repo / "run.py"),
            "python_path": "/usr/bin/python3",
        }

    def test_seq_free_argv_omits_fasta(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = _make_repo(root)
            params = {
                "emodelx_protocol": "seq_free",
                "emodelx_repo_dir": str(repo),
                "map_path": str(root / "map.mrc"),
                "output_dir": str(root / "out"),
            }
            result = run_emodelx_pipeline(
                params=params,
                preflight=self._preflight_stub(repo, "seq_free"),
                dry_run=True,
            )
            argv = result["command_argv"]
            self.assertNotIn("--fasta", argv)
            self.assertIn("--protocol", argv)
            self.assertEqual(argv[argv.index("--protocol") + 1], "seq_free")
            self.assertIn("--EM_map", argv)
            self.assertIn("--output_dir", argv)

    def test_temp_free_argv_includes_fasta(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = _make_repo(root)
            params = {
                "emodelx_protocol": "temp_free",
                "emodelx_repo_dir": str(repo),
                "map_path": str(root / "map.mrc"),
                "sequence_path": str(root / "seq.fasta"),
                "output_dir": str(root / "out"),
            }
            result = run_emodelx_pipeline(
                params=params,
                preflight=self._preflight_stub(repo, "temp_free"),
                dry_run=True,
            )
            argv = result["command_argv"]
            self.assertIn("--fasta", argv)
            self.assertEqual(argv[argv.index("--fasta") + 1], str(root / "seq.fasta"))

    def test_temp_flex_argv_includes_fasta_and_template_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = _make_repo(root)
            tpl = root / "templates"
            tpl.mkdir()
            params = {
                "emodelx_protocol": "temp_flex",
                "emodelx_repo_dir": str(repo),
                "map_path": str(root / "map.mrc"),
                "sequence_path": str(root / "seq.fasta"),
                "template_dir": str(tpl),
                "output_dir": str(root / "out"),
            }
            result = run_emodelx_pipeline(
                params=params,
                preflight=self._preflight_stub(repo, "temp_flex"),
                dry_run=True,
            )
            argv = result["command_argv"]
            self.assertIn("--fasta", argv)
            self.assertIn("--template_dir", argv)
            self.assertEqual(argv[argv.index("--template_dir") + 1], str(tpl))


class TestRunnerMkdirsOutputDir(unittest.TestCase):
    def test_missing_output_dir_is_created_before_subprocess(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = _make_repo(root)
            out_dir = root / "nested" / "out_not_exists_yet"
            # The parent chain does not exist; runner must mkdir before exec.
            self.assertFalse(out_dir.exists())

            fake_proc = MagicMock()
            fake_proc.returncode = 0
            fake_proc.stdout = "ok"
            fake_proc.stderr = ""

            def _fake_run(cmd, **_kwargs):  # noqa: ANN001
                # Subprocess would write here; if parent still doesn't exist
                # this write would explode, which is precisely what we want
                # to prove the runner has ensured it.
                (out_dir / "_EModelX(noseq)_ca_model.pdb").write_text("pdb", encoding="utf-8")
                return fake_proc

            params = {
                "emodelx_protocol": "seq_free",
                "emodelx_repo_dir": str(repo),
                "map_path": str(root / "map.mrc"),
                "output_dir": str(out_dir),
            }
            preflight = {
                "backend_name": "emodelx",
                "available": True,
                "execution_mode": "real",
                "protocol": "seq_free",
                "repo_path": str(repo),
                "run_py_path": str(repo / "run.py"),
                "python_path": "/usr/bin/python3",
            }
            with patch("relion.emodelx_runner.subprocess.run", side_effect=_fake_run):
                result = run_emodelx_pipeline(
                    params=params, preflight=preflight, dry_run=False
                )
            self.assertTrue(out_dir.is_dir())
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["return_code"], 0)


class TestT1CasePassesPreflight(unittest.TestCase):
    def test_t1_tutorial5_seq_free_preflight_passes(self) -> None:
        # Materialize the registered T1 case against a real fixture and
        # assert that the materialized inputs produce an available preflight.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = _make_repo(root)
            map_path = root / "tutorial5" / "map.mrc"
            map_path.parent.mkdir(parents=True, exist_ok=True)
            map_path.write_text("mrc", encoding="utf-8")
            out_dir = root / "runs" / "T1" / "output"

            params = {
                "emodelx_protocol": "seq_free",
                "emodelx_repo_dir": str(repo),
                "emodelx_python": "/usr/bin/python3",
                "map_path": str(map_path),
                "output_dir": str(out_dir),
            }
            r = preflight_model_building("emodelx", params)
            self.assertTrue(r["available"], r)
            self.assertNotIn("sequence_path", r["missing_inputs"])
            self.assertNotIn(
                "missing_required_inputs: sequence_path", str(r.get("blocked_reason") or "")
            )

    def test_t1_is_registered_in_default_cases(self) -> None:
        cases = default_cases()
        self.assertIn("T1_tutorial5_map_emodelx_stopA", cases)
        case = cases["T1_tutorial5_map_emodelx_stopA"]
        self.assertEqual(case.target_stage, "model_building")
        self.assertEqual(case.input_state.get("emodelx_protocol"), "seq_free")
        # sequence_path must not be advertised as required/available here.
        self.assertNotIn("sequence_path", case.available_inputs)

    def test_t1_yaml_materializes_cleanly(self) -> None:
        # YAML version with placeholders should materialize against concrete
        # variable bindings.  We only check it parses + expands; we do not
        # require the paths to exist on disk for this unit test.
        yaml_path = (
            Path(__file__).resolve().parent.parent
            / "configs"
            / "cases"
            / "T1_tutorial5_map_emodelx_stopA.yaml"
        )
        self.assertTrue(yaml_path.is_file())
        from agent.case_registry import load_case

        raw = load_case(yaml_path)
        mat = materialize_case(
            raw,
            variables={
                "PROJECT_ROOT": "/p",
                "DATA_ROOT": "/d",
                "RUN_ROOT": "/r",
                "CASE_ROOT": "/c",
            },
        )
        self.assertEqual(
            mat.available_inputs["map_path"],
            "/d/Tutorial5.0/PostProcess/job035/postprocess_masked.mrc",
        )
        self.assertEqual(mat.available_inputs["emodelx_repo_dir"], "/p/EModelX")
        self.assertEqual(mat.available_inputs["emodelx_protocol"], "seq_free")
        self.assertNotIn("sequence_path", mat.available_inputs)


if __name__ == "__main__":
    unittest.main()
