"""Stage-30 tests: T2 ModelAngelo real success stop-A.

Covers:
- Protocol + case registry can resolve and materialize T2.
- fixed_workflow_modelangelo pins modelangelo under fixed routing.
- ModelAngelo runner harvests primary structure + summary-class files.
- Success audit evidence_strength bumps to medium when artifacts are on disk.
- T1 seq_free emodelx routing is not regressed by this addition.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.baseline_runner import _select_backend, run_experiment_case
from agent.benchmark_harness import BenchmarkHarness
from agent.case_registry import (
    ExperimentCase,
    default_cases,
    load_case,
    materialize_case,
    resolve_case,
)
from agent.experiment_protocol import resolve_protocol
from agent.success_audit import classify_success
from app.scheduler import Scheduler
from relion.model_angelo_runner import _harvest_model_angelo_artifacts


class TestT2RegistryResolution(unittest.TestCase):
    def test_t2_registered_in_default_cases(self) -> None:
        cases = default_cases()
        self.assertIn("T2_modelangelo_real_success_stopA", cases)
        c = cases["T2_modelangelo_real_success_stopA"]
        self.assertEqual(c.target_stage, "model_building")
        self.assertEqual(c.case_type, "map_level_case")
        self.assertTrue(c.input_state.get("has_map"))
        self.assertTrue(c.input_state.get("has_sequence"))

    def test_t2_yaml_materializes_cleanly(self) -> None:
        yaml_path = (
            Path(__file__).resolve().parent.parent
            / "configs"
            / "cases"
            / "T2_modelangelo_real_success_stopA.yaml"
        )
        self.assertTrue(yaml_path.is_file())
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
        self.assertEqual(
            mat.available_inputs["sequence_path"],
            "/d/Tutorial5.0/Ecoli.fasta",
        )
        self.assertEqual(
            mat.available_inputs["output_dir"],
            "/r/T2_modelangelo_real_success_stopA/output",
        )


class TestFixedWorkflowModelAngeloRouting(unittest.TestCase):
    def test_protocol_yaml_resolves(self) -> None:
        proto = resolve_protocol(
            "fixed_workflow_modelangelo",
            search_dirs=[
                Path(__file__).resolve().parent.parent
                / "configs"
                / "experiment_protocols"
            ],
        )
        self.assertEqual(proto.protocol_name, "fixed_workflow_modelangelo")
        self.assertEqual(proto.routing_mode, "fixed")
        self.assertEqual(proto.allowed_backends, ["modelangelo"])
        self.assertFalse(proto.llm_supervision_enabled)

    def test_fixed_routing_picks_modelangelo_for_t2(self) -> None:
        proto = resolve_protocol(
            "fixed_workflow_modelangelo",
            search_dirs=[
                Path(__file__).resolve().parent.parent
                / "configs"
                / "experiment_protocols"
            ],
        )
        case = default_cases()["T2_modelangelo_real_success_stopA"]
        sel = _select_backend(proto, case)
        self.assertEqual(sel["selected_backend"], "modelangelo")
        self.assertEqual(sel["selection_reason"], "protocol.routing_mode=fixed")
        self.assertFalse(sel["fallback_applied"])

    def test_seq_free_heuristic_does_not_misroute_t2(self) -> None:
        # T2 has has_sequence=True; seq_free preference must not fire.
        proto = resolve_protocol("rule_only_agent")
        case = default_cases()["T2_modelangelo_real_success_stopA"]
        sel = _select_backend(proto, case)
        # rule_only_agent's default allowed list omits modelangelo; falls back
        # to registry_default via fallback_policy.  What matters here is that
        # it's not emodelx (that's T1's path).
        self.assertNotEqual(sel["selected_backend"], "emodelx")


class TestHarvestModelAngeloArtifacts(unittest.TestCase):
    def test_collects_structure_and_summary_excludes_logs(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            out.mkdir()
            (out / "model.cif").write_text("cif", encoding="utf-8")
            (out / "backup.pdb").write_text("pdb", encoding="utf-8")
            (out / "summary.json").write_text("{}", encoding="utf-8")
            (out / "ranking.csv").write_text("a,b\n", encoding="utf-8")
            (out / "score_table.tsv").write_text("n\t1\n", encoding="utf-8")
            (out / "model_angelo_run.log").write_text("log", encoding="utf-8")
            (out / "noise.txt").write_text("noise", encoding="utf-8")
            harvested = _harvest_model_angelo_artifacts(out)
            names = [Path(p).name for p in harvested]
            # Primary structure files present.
            self.assertIn("model.cif", names)
            self.assertIn("backup.pdb", names)
            # Summary-class files present.
            self.assertIn("summary.json", names)
            self.assertIn("ranking.csv", names)
            self.assertIn("score_table.tsv", names)
            # Log-style files excluded.
            self.assertNotIn("model_angelo_run.log", names)
            self.assertNotIn("noise.txt", names)
            # Structure files are listed before summary files.
            self.assertLess(names.index("model.cif"), names.index("summary.json"))

    def test_structure_files_ordered_newest_first(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            out.mkdir()
            older = out / "old.cif"
            newer = out / "new.cif"
            older.write_text("old", encoding="utf-8")
            # Bump mtime on newer to be strictly later.
            import os, time

            time.sleep(0.01)
            newer.write_text("new", encoding="utf-8")
            os.utime(older, (older.stat().st_atime, older.stat().st_mtime - 10))
            harvested = _harvest_model_angelo_artifacts(out)
            names = [Path(p).name for p in harvested]
            self.assertLess(names.index("new.cif"), names.index("old.cif"))

    def test_missing_output_dir_returns_empty(self) -> None:
        with TemporaryDirectory() as tmp:
            self.assertEqual(
                _harvest_model_angelo_artifacts(Path(tmp) / "nope"), []
            )


class TestSuccessAuditEvidenceForModelAngelo(unittest.TestCase):
    def test_artifacts_present_yields_medium(self) -> None:
        cr = {
            "benchmark_id": "T2",
            "backend": "modelangelo",
            "status": "success",
            "return_code": 0,
            "execution_mode": "real",
            "artifacts": ["/out/model.cif", "/out/summary.json"],
            "generated_paths": ["/out/model.cif", "/out/summary.json"],
            "extra": {},
        }
        cls = classify_success(cr)
        self.assertEqual(cls["evidence_strength"], "medium")
        self.assertNotEqual(cls["evidence_strength"], "low")


class TestT2EndToEndFixedWorkflow(unittest.TestCase):
    """Drive run_experiment_case under fixed_workflow_modelangelo with a fake
    model_angelo binary that emits a realistic output directory."""

    def test_t2_runs_and_populates_artifacts_and_success_audit(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Fixture data.
            data_root = root / "Tutorial5.0"
            post = data_root / "PostProcess" / "job035"
            post.mkdir(parents=True, exist_ok=True)
            map_path = post / "postprocess_masked.mrc"
            map_path.write_text("mrc", encoding="utf-8")
            fasta_path = data_root / "Ecoli.fasta"
            fasta_path.write_text(">A\nMRE\n", encoding="utf-8")
            out_dir = root / "T2_run" / "output"

            # Fake ``model_angelo`` shim that parses ``-v``, ``-pf``, ``-o``
            # and writes a realistic product set.
            import stat

            fake_bin = root / "fake_model_angelo.sh"
            fake_bin.write_text(
                "#!/bin/sh\n"
                "# fake model_angelo: parses -v/-pf/-o and writes minimal products\n"
                "while [ $# -gt 0 ]; do\n"
                '  case "$1" in\n'
                "    -v) MAP=$2; shift 2 ;;\n"
                "    -pf) FASTA=$2; shift 2 ;;\n"
                "    -o) OUT=$2; shift 2 ;;\n"
                "    *) shift ;;\n"
                "  esac\n"
                "done\n"
                'mkdir -p "$OUT"\n'
                'echo "cif-model" > "$OUT/model.cif"\n'
                'echo "{}" > "$OUT/summary.json"\n'
                'echo "run ok" > "$OUT/model_angelo_run.log"\n'
                "exit 0\n",
                encoding="utf-8",
            )
            fake_bin.chmod(fake_bin.stat().st_mode | stat.S_IXUSR | stat.S_IRUSR | stat.S_IWUSR)

            # model_builders.yaml pointing at the fake binary.
            cfg_dir = root / "configs"
            cfg_dir.mkdir()
            (cfg_dir / "model_builders.yaml").write_text(
                "model_angelo:\n"
                f"  executable: {fake_bin}\n"
                "  subcommand_with_seq: [build]\n"
                "  flags: {map: -v, protein_fasta: -pf, output: -o}\n",
                encoding="utf-8",
            )

            case = ExperimentCase(
                case_id="T2_modelangelo_real_success_stopA",
                case_type="map_level_case",
                dataset_name="tutorial5",
                dataset_root=str(data_root),
                input_state={"has_map": True, "has_sequence": True},
                available_inputs={
                    "map_path": str(map_path),
                    "sequence_path": str(fasta_path),
                    "output_dir": str(out_dir),
                },
                target_stage="model_building",
                evaluation_mode="success_only",
            )
            proto = resolve_protocol(
                "fixed_workflow_modelangelo",
                search_dirs=[
                    Path(__file__).resolve().parent.parent
                    / "configs"
                    / "experiment_protocols"
                ],
            )

            # Scheduler needs the model_builders_yaml pointed at the fixture.
            scheduler = Scheduler(project_root=root)
            scheduler.action_executor.model_builders_yaml = cfg_dir / "model_builders.yaml"
            harness = BenchmarkHarness(
                executor=scheduler.action_executor, project_root=root
            )

            result = run_experiment_case(
                protocol=proto,
                case=case,
                harness=harness,
                project_root=root,
                output_dir=root / "out_exp",
                dry_run=False,
                persist=False,
                write_audit=True,
            )
            payload = result.to_dict()

            self.assertEqual(payload["selected_backend"], "modelangelo")
            self.assertEqual(payload["status"], "success")
            self.assertEqual(payload["stage_reached"], "model_building")

            cr = payload["case_result"]
            self.assertEqual(cr.get("return_code"), 0)
            names = [Path(p).name for p in (cr.get("artifacts") or [])]
            gen_names = [Path(p).name for p in (cr.get("generated_paths") or [])]
            self.assertIn("model.cif", names)
            self.assertIn("summary.json", names)
            self.assertIn("model.cif", gen_names)
            self.assertNotIn("model_angelo_run.log", names)

            audit_json = payload["audit_paths"].get("success_audit_json")
            self.assertIsNotNone(audit_json)
            audit = json.loads(Path(audit_json).read_text(encoding="utf-8"))
            strength = audit["success_classification"]["evidence_strength"]
            self.assertIn(strength, ("medium", "high"))
            self.assertNotEqual(strength, "low")


class TestT1EmodelxRoutingNotRegressed(unittest.TestCase):
    def test_t1_still_routes_to_emodelx_under_rule_only(self) -> None:
        proto = resolve_protocol("rule_only_agent")
        case = default_cases()["T1_tutorial5_map_emodelx_stopA"]
        sel = _select_backend(proto, case)
        self.assertEqual(sel["selected_backend"], "emodelx")
        self.assertIn("seq_free", sel["selection_reason"])


if __name__ == "__main__":
    unittest.main()
