"""Input contracts and preflight for model-building backends."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.backend_registry import MODEL_BUILDING_BACKEND_METADATA, MODEL_BUILDING_INPUT_SPECS
from agent.model_building_preflight import preflight_model_building


class TestModelBuildingInputSpecs(unittest.TestCase):
    def test_each_backend_has_contract_in_registry(self) -> None:
        for bid in ("modelangelo", "emodelx", "mica"):
            self.assertIn(bid, MODEL_BUILDING_INPUT_SPECS)
            self.assertIn(bid, MODEL_BUILDING_BACKEND_METADATA)
            meta = MODEL_BUILDING_BACKEND_METADATA[bid]
            self.assertEqual(meta.get("required_fields"), list(MODEL_BUILDING_INPUT_SPECS[bid]["required_fields"]))

    def test_modelangelo_requires_map_and_sequence(self) -> None:
        spec = MODEL_BUILDING_INPUT_SPECS["modelangelo"]
        self.assertIn("map_path", spec["required_fields"])
        self.assertIn("sequence_path", spec["required_fields"])

    def test_emodelx_requires_real_execution_fields(self) -> None:
        spec = MODEL_BUILDING_INPUT_SPECS["emodelx"]
        self.assertIn("map_path", spec["required_fields"])
        self.assertIn("sequence_path", spec["required_fields"])
        self.assertIn("output_dir", spec["required_fields"])
        self.assertIn("emodelx_repo_dir", spec["required_fields"])
        for field in (
            "emodelx_python",
            "run_pulchra",
            "pulchra_path",
            "run_phenix",
            "phenix_act",
            "resolution",
        ):
            self.assertIn(field, spec["optional_fields"])

    def test_mica_requires_real_execution_fields(self) -> None:
        spec = MODEL_BUILDING_INPUT_SPECS["mica"]
        self.assertIn("map_path", spec["required_fields"])
        self.assertIn("sequence_path", spec["required_fields"])
        self.assertIn("output_dir", spec["required_fields"])
        self.assertIn("mica_repo_dir", spec["required_fields"])
        self.assertIn("af3_dir", spec["required_fields"])
        for field in (
            "mica_python",
            "run_pulchra",
            "pulchra_path",
            "run_phenix",
            "phenix_act",
            "mica_resolution",
        ):
            self.assertIn(field, spec["optional_fields"])


class TestModelBuildingPreflight(unittest.TestCase):
    def test_modelangelo_missing_map_unavailable(self) -> None:
        r = preflight_model_building(
            "modelangelo",
            {"sequence_path": "/s.fasta"},
        )
        self.assertFalse(r["available"])
        self.assertEqual(r["execution_mode"], "unavailable")
        self.assertIn("map_path", r["missing_inputs"])
        self.assertTrue(str(r.get("blocked_reason") or "").startswith("missing_required_inputs"))

    def test_emodelx_missing_map_unavailable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "EModelX"
            (repo / "models").mkdir(parents=True)
            (repo / "models" / "w.bin").write_text("x", encoding="utf-8")
            (repo / "run.py").write_text("print('ok')", encoding="utf-8")
            (root / "s.fasta").write_text(">a\nA", encoding="utf-8")
            r = preflight_model_building(
                "emodelx",
                {
                    "sequence_path": str(root / "s.fasta"),
                    "output_dir": str(root / "out"),
                    "emodelx_repo_dir": str(repo),
                },
            )
            self.assertFalse(r["available"])
            self.assertIn("map_path", r["missing_inputs"])

    def test_emodelx_missing_repo_unavailable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "m.mrc").write_text("x", encoding="utf-8")
            (root / "s.fasta").write_text(">a\nA", encoding="utf-8")
            r = preflight_model_building(
                "emodelx",
                {
                    "map_path": str(root / "m.mrc"),
                    "sequence_path": str(root / "s.fasta"),
                    "output_dir": str(root / "out"),
                    "emodelx_repo_dir": str(root / "missing_repo"),
                },
            )
            self.assertFalse(r["available"])
            self.assertIn("emodelx_repo_dir", r["missing_inputs"])

    def test_emodelx_missing_weights_unavailable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "EModelX"
            repo.mkdir(parents=True)
            (repo / "run.py").write_text("print('ok')", encoding="utf-8")
            (root / "m.mrc").write_text("x", encoding="utf-8")
            (root / "s.fasta").write_text(">a\nA", encoding="utf-8")
            r = preflight_model_building(
                "emodelx",
                {
                    "map_path": str(root / "m.mrc"),
                    "sequence_path": str(root / "s.fasta"),
                    "output_dir": str(root / "out"),
                    "emodelx_repo_dir": str(repo),
                },
            )
            self.assertFalse(r["available"])
            self.assertIn("emodelx_weights", r["missing_dependencies"])
            self.assertIn("missing_dependency: emodelx_weights", str(r.get("blocked_reason") or ""))

    def test_emodelx_inputs_ready_real_mode(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "EModelX"
            (repo / "models").mkdir(parents=True)
            (repo / "models" / "w.bin").write_text("x", encoding="utf-8")
            (repo / "run.py").write_text("print('ok')", encoding="utf-8")
            (root / "m.mrc").write_text("x", encoding="utf-8")
            (root / "s.fasta").write_text(">a\nA", encoding="utf-8")
            r = preflight_model_building(
                "emodelx",
                {
                    "map_path": str(root / "m.mrc"),
                    "sequence_path": str(root / "s.fasta"),
                    "output_dir": str(root / "out"),
                    "emodelx_repo_dir": str(repo),
                },
            )
            self.assertTrue(r["available"])
            self.assertEqual(r["execution_mode"], "real")
            self.assertFalse(r["placeholder"])
            self.assertTrue(bool(r.get("repo_path")))
            self.assertEqual(r.get("protocol"), "temp_free")
            self.assertTrue(str(r.get("command") or "").startswith(str(r.get("python_path"))))
            self.assertEqual(r.get("output_path"), str(root / "out"))

    def test_emodelx_run_pulchra_missing_path_unavailable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "EModelX"
            (repo / "models").mkdir(parents=True)
            (repo / "models" / "w.bin").write_text("x", encoding="utf-8")
            (repo / "run.py").write_text("print('ok')", encoding="utf-8")
            (root / "m.mrc").write_text("x", encoding="utf-8")
            (root / "s.fasta").write_text(">a\nA", encoding="utf-8")
            r = preflight_model_building(
                "emodelx",
                {
                    "map_path": str(root / "m.mrc"),
                    "sequence_path": str(root / "s.fasta"),
                    "output_dir": str(root / "out"),
                    "emodelx_repo_dir": str(repo),
                    "run_pulchra": True,
                },
            )
            self.assertFalse(r["available"])
            self.assertIn("pulchra_path", r["missing_dependencies"])

    def test_mica_missing_af3_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "MICA"
            (repo / "trained_models").mkdir(parents=True)
            (repo / "trained_models" / "MICA_best_model.pth").write_text("x", encoding="utf-8")
            (repo / "run.py").write_text("print('ok')", encoding="utf-8")
            (root / "m.mrc").write_text("x", encoding="utf-8")
            (root / "s.fasta").write_text(">a\nA", encoding="utf-8")
            r = preflight_model_building(
                "mica",
                {
                    "map_path": str(root / "m.mrc"),
                    "sequence_path": str(root / "s.fasta"),
                    "output_dir": str(root / "out"),
                    "mica_repo_dir": str(repo),
                },
            )
            self.assertFalse(r["available"])
            self.assertIn("af3_dir", r["missing_inputs"])
            self.assertIn("missing_required_inputs: af3_dir", str(r.get("blocked_reason") or ""))

    def test_mica_missing_repo_unavailable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            af3 = root / "af3"
            af3.mkdir(parents=True)
            (af3 / "x.txt").write_text("x", encoding="utf-8")
            (root / "m.mrc").write_text("x", encoding="utf-8")
            (root / "s.fasta").write_text(">a\nA", encoding="utf-8")
            r = preflight_model_building(
                "mica",
                {
                    "map_path": str(root / "m.mrc"),
                    "sequence_path": str(root / "s.fasta"),
                    "output_dir": str(root / "out"),
                    "mica_repo_dir": str(root / "missing_repo"),
                    "af3_dir": str(af3),
                },
            )
            self.assertFalse(r["available"])
            self.assertIn("mica_repo_dir", r["missing_inputs"])

    def test_mica_missing_weights_unavailable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "MICA"
            repo.mkdir(parents=True)
            (repo / "run.py").write_text("print('ok')", encoding="utf-8")
            af3 = root / "af3"
            af3.mkdir(parents=True)
            (af3 / "x.txt").write_text("x", encoding="utf-8")
            (root / "m.mrc").write_text("x", encoding="utf-8")
            (root / "s.fasta").write_text(">a\nA", encoding="utf-8")
            r = preflight_model_building(
                "mica",
                {
                    "map_path": str(root / "m.mrc"),
                    "sequence_path": str(root / "s.fasta"),
                    "output_dir": str(root / "out"),
                    "mica_repo_dir": str(repo),
                    "af3_dir": str(af3),
                },
            )
            self.assertFalse(r["available"])
            self.assertIn("mica_weights", r["missing_dependencies"])
            self.assertIn("missing_dependency: mica_weights", str(r.get("blocked_reason") or ""))

    def test_mica_accepts_af3_result_dir_alias_and_real_mode(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "MICA"
            (repo / "trained_models").mkdir(parents=True)
            (repo / "trained_models" / "MICA_best_model.pth").write_text("x", encoding="utf-8")
            (repo / "run.py").write_text("print('ok')", encoding="utf-8")
            af3 = root / "af3"
            af3.mkdir(parents=True)
            (af3 / "x.txt").write_text("x", encoding="utf-8")
            (root / "m.mrc").write_text("x", encoding="utf-8")
            (root / "s.fasta").write_text(">a\nA", encoding="utf-8")
            r = preflight_model_building(
                "mica",
                {
                    "map_path": str(root / "m.mrc"),
                    "sequence_path": str(root / "s.fasta"),
                    "output_dir": str(root / "out"),
                    "mica_repo_dir": str(repo),
                    "af3_result_dir": str(af3),
                },
            )
            self.assertTrue(r["available"])
            self.assertEqual(r["execution_mode"], "real")
            self.assertEqual(r.get("af3_dir"), str(af3))

    def test_mica_run_pulchra_missing_path_unavailable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "MICA"
            (repo / "trained_models").mkdir(parents=True)
            (repo / "trained_models" / "MICA_best_model.pth").write_text("x", encoding="utf-8")
            (repo / "run.py").write_text("print('ok')", encoding="utf-8")
            af3 = root / "af3"
            af3.mkdir(parents=True)
            (af3 / "x.txt").write_text("x", encoding="utf-8")
            (root / "m.mrc").write_text("x", encoding="utf-8")
            (root / "s.fasta").write_text(">a\nA", encoding="utf-8")
            r = preflight_model_building(
                "mica",
                {
                    "map_path": str(root / "m.mrc"),
                    "sequence_path": str(root / "s.fasta"),
                    "output_dir": str(root / "out"),
                    "mica_repo_dir": str(repo),
                    "af3_dir": str(af3),
                    "run_pulchra": True,
                },
            )
            self.assertFalse(r["available"])
            self.assertIn("pulchra_path", r["missing_dependencies"])

    def test_modelangelo_real_mode_when_inputs_ok(self) -> None:
        r = preflight_model_building(
            "modelangelo",
            {"map_path": "/m.mrc", "sequence_path": "/s.fasta"},
        )
        self.assertTrue(r["available"])
        self.assertEqual(r["execution_mode"], "real")
        self.assertFalse(r["placeholder"])

    def test_modelangelo_stale_configured_executable_unavailable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "model_builders.yaml"
            cfg.write_text(
                "model_angelo:\n"
                "  executable: /root/anaconda3/envs/model_angelo/bin/model_angelo\n"
                "  torch_home: /root/share/szs/CryoAgent/model-angelo/public/model_angelo_weights\n",
                encoding="utf-8",
            )
            r = preflight_model_building(
                "modelangelo",
                {"map_path": "/m.mrc", "sequence_path": "/s.fasta"},
                modelangelo_config_path=cfg,
            )
            self.assertFalse(r["available"])
            self.assertEqual(r["execution_mode"], "unavailable")
            self.assertIn("modelangelo_executable", r["missing_dependencies"])
            self.assertTrue(any("stale" in w for w in r["warnings"]))


if __name__ == "__main__":
    unittest.main()
