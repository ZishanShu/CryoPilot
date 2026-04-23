"""Stage-29 tests: seq_free routing + EModelX artifact harvest + success evidence."""

from __future__ import annotations

import json
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from agent.baseline_runner import _select_backend, run_experiment_case
from agent.benchmark_harness import BenchmarkHarness
from agent.case_registry import default_cases
from agent.experiment_protocol import resolve_protocol
from agent.success_audit import classify_success
from app.scheduler import Scheduler
from relion.emodelx_runner import _harvest_emodelx_artifacts, run_emodelx_pipeline


class TestSeqFreeRouting(unittest.TestCase):
    def setUp(self) -> None:
        self.case = default_cases()["T1_tutorial5_map_emodelx_stopA"]

    def test_rule_only_agent_picks_emodelx_not_modelangelo(self) -> None:
        # Default rule_only_agent allows topaz / cryodrgn / emodelx / mica.
        # Before this patch the fallback_policy=registry_default branch would
        # silently return modelangelo (even though it is not in allowed); the
        # fix must short-circuit to emodelx for seq_free cases.
        protocol = resolve_protocol("rule_only_agent")
        sel = _select_backend(protocol, self.case)
        self.assertEqual(sel["selected_backend"], "emodelx")
        self.assertFalse(sel["fallback_applied"])
        self.assertEqual(sel["category"], "model_building")
        self.assertIn("seq_free", sel["selection_reason"])

    def test_llm_supervised_agent_picks_emodelx_not_modelangelo(self) -> None:
        protocol = resolve_protocol("llm_supervised_agent")
        sel = _select_backend(protocol, self.case)
        self.assertEqual(sel["selected_backend"], "emodelx")
        self.assertIn("seq_free", sel["selection_reason"])

    def test_explicit_modelangelo_allowed_gets_rejection_reason(self) -> None:
        # When a custom protocol explicitly allows modelangelo alongside
        # emodelx, the rejection_reasons map must cite sequence_path.
        from agent.experiment_protocol import parse_protocol

        proto = parse_protocol(
            {
                "protocol_name": "rule_with_modelangelo",
                "protocol_kind": "rule_only_agent",
                "allowed_backends": ["emodelx", "modelangelo"],
                "routing_mode": "rule",
                "fallback_policy": "registry_default",
            }
        )
        sel = _select_backend(proto, self.case)
        self.assertEqual(sel["selected_backend"], "emodelx")
        self.assertIn("modelangelo", sel["rejected_backends"])
        self.assertIn(
            "sequence_path",
            sel["rejection_reasons"].get("modelangelo", ""),
        )

    def test_sequence_guided_case_still_routes_to_modelangelo(self) -> None:
        # Regression guard: the existing mid3d_map_only case has
        # has_sequence=true and must still pick modelangelo under rule mode.
        case = default_cases()["mid3d_map_only_model_building"]
        protocol = resolve_protocol("rule_only_agent")
        sel = _select_backend(protocol, case)
        self.assertEqual(sel["selected_backend"], "modelangelo")

    def test_mica_rejection_reason_not_invented(self) -> None:
        # We only assert modelangelo gets an explicit sequence_path reason;
        # the mica rejection reason is intentionally left to existing logic
        # (empty here — task spec explicitly says "保留现有依赖/输入逻辑").
        protocol = resolve_protocol("rule_only_agent")
        sel = _select_backend(protocol, self.case)
        self.assertNotIn("mica", sel["rejection_reasons"])

    def test_fixed_workflow_emodelx_unchanged(self) -> None:
        # When the protocol pins emodelx, selection_reason is not the
        # seq_free heuristic — fixed mode takes priority.
        # Build a single-allowed protocol using parse_protocol.
        from agent.experiment_protocol import parse_protocol

        proto = parse_protocol(
            {
                "protocol_name": "fixed_workflow_emodelx",
                "protocol_kind": "fixed_workflow",
                "allowed_backends": ["emodelx"],
                "routing_mode": "fixed",
                "llm_supervision_enabled": False,
                "fallback_policy": "none",
            }
        )
        sel = _select_backend(proto, self.case)
        self.assertEqual(sel["selected_backend"], "emodelx")
        self.assertEqual(sel["selection_reason"], "protocol.routing_mode=fixed")


class TestEmodelxArtifactHarvesting(unittest.TestCase):
    def test_harvest_includes_pdb_and_time_cost_csv(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "output"
            out.mkdir()
            (out / "_EModelX(noseq)_ca_model.pdb").write_text("pdb", encoding="utf-8")
            (out / "time_cost__EModelX(noseq).csv").write_text("t,s\n0,1\n", encoding="utf-8")
            (out / "unrelated.log").write_text("noise", encoding="utf-8")
            harvested = _harvest_emodelx_artifacts(out, created=[])
            names = [Path(p).name for p in harvested]
            self.assertIn("_EModelX(noseq)_ca_model.pdb", names)
            self.assertIn("time_cost__EModelX(noseq).csv", names)
            self.assertNotIn("unrelated.log", names)

    def test_harvest_unions_created_diff_with_patterns(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "out"
            out.mkdir()
            (out / "model.pdb").write_text("pdb", encoding="utf-8")
            harvested = _harvest_emodelx_artifacts(
                out,
                created=[str(out / "model.pdb"), str(out / "stray.pkl")],
            )
            # Both the diff-observed stray.pkl AND the pattern-matched pdb
            # survive; duplicates are deduplicated.
            self.assertEqual(len([p for p in harvested if p.endswith("model.pdb")]), 1)
            self.assertIn(str(out / "stray.pkl"), harvested)

    def test_harvest_gracefully_handles_missing_output_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            ghost = Path(tmp) / "does_not_exist"
            self.assertEqual(_harvest_emodelx_artifacts(ghost, created=[]), [])


class TestEmodelxRunnerReturnsArtifactsAndGeneratedPaths(unittest.TestCase):
    def _stub_preflight(self, repo: Path) -> dict:
        return {
            "backend_name": "emodelx",
            "available": True,
            "execution_mode": "real",
            "protocol": "seq_free",
            "repo_path": str(repo),
            "run_py_path": str(repo / "run.py"),
            "python_path": "/usr/bin/python3",
        }

    def test_success_payload_includes_both_artifact_and_generated_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "EModelX"
            (repo / "models").mkdir(parents=True)
            (repo / "models" / "w.bin").write_text("x", encoding="utf-8")
            (repo / "run.py").write_text("print('stub')", encoding="utf-8")
            out_dir = root / "out"

            fake_proc = MagicMock()
            fake_proc.returncode = 0
            fake_proc.stdout = "ok"
            fake_proc.stderr = ""

            def _fake_run(cmd, **_kwargs):  # noqa: ANN001
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "_EModelX(noseq)_ca_model.pdb").write_text("pdb", encoding="utf-8")
                (out_dir / "time_cost__EModelX(noseq).csv").write_text("t\n", encoding="utf-8")
                return fake_proc

            params = {
                "emodelx_protocol": "seq_free",
                "emodelx_repo_dir": str(repo),
                "map_path": str(root / "map.mrc"),
                "output_dir": str(out_dir),
            }
            with patch("relion.emodelx_runner.subprocess.run", side_effect=_fake_run):
                res = run_emodelx_pipeline(
                    params=params,
                    preflight=self._stub_preflight(repo),
                    dry_run=False,
                )
            self.assertEqual(res["status"], "completed")
            self.assertEqual(res["return_code"], 0)
            names = [Path(p).name for p in res["artifact_paths"]]
            self.assertIn("_EModelX(noseq)_ca_model.pdb", names)
            self.assertIn("time_cost__EModelX(noseq).csv", names)
            # generated_paths must mirror artifact_paths for consumers that
            # read either key.
            self.assertEqual(
                sorted(res["generated_paths"]),
                sorted(res["artifact_paths"]),
            )


class TestSuccessAuditEvidenceStrength(unittest.TestCase):
    def test_artifacts_present_without_checks_yields_medium(self) -> None:
        cr = {
            "benchmark_id": "T1",
            "backend": "emodelx",
            "status": "completed",
            "return_code": 0,
            "execution_mode": "real",
            "artifacts": [
                "/out/_EModelX(noseq)_ca_model.pdb",
                "/out/time_cost__EModelX(noseq).csv",
            ],
            "extra": {},  # no smoke_run_checks
        }
        classification = classify_success(cr)
        self.assertEqual(classification["evidence_strength"], "medium")
        self.assertEqual(classification["success_kind"], "execution_completed")

    def test_no_artifacts_and_no_checks_stays_low(self) -> None:
        cr = {
            "benchmark_id": "x",
            "backend": "emodelx",
            "status": "completed",
            "return_code": 0,
            "execution_mode": "real",
            "artifacts": [],
            "extra": {},
        }
        self.assertEqual(classify_success(cr)["evidence_strength"], "low")

    def test_full_checks_and_evaluation_stays_high(self) -> None:
        cr = {
            "status": "completed",
            "extra": {
                "smoke_run_checks": {
                    "preflight_ok": True,
                    "execution_ok": True,
                    "non_empty_output": True,
                    "smoke_run_success": True,
                },
                "picking_evaluation": {"evaluation_available": True, "precision": 0.9},
            },
        }
        self.assertEqual(classify_success(cr)["evidence_strength"], "high")

    def test_generated_paths_also_count_as_artifacts(self) -> None:
        cr = {
            "status": "completed",
            "generated_paths": ["/out/x.pdb"],
            "extra": {},
        }
        self.assertEqual(classify_success(cr)["evidence_strength"], "medium")


class TestEndToEndT1SuccessAudit(unittest.TestCase):
    """Run T1 through the baseline_runner with a fake EModelX binary."""

    def test_t1_rule_only_agent_collects_artifacts_and_writes_success_audit(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Repo layout expected by EModelX preflight.
            repo = root / "EModelX"
            (repo / "models").mkdir(parents=True)
            (repo / "models" / "w.bin").write_text("x", encoding="utf-8")
            run_py = repo / "run.py"
            run_py.write_text(
                "import os, sys\n"
                "args = sys.argv\n"
                "out = None\n"
                "for i, a in enumerate(args):\n"
                "    if a == '--output_dir':\n"
                "        out = args[i + 1]\n"
                "os.makedirs(out, exist_ok=True)\n"
                "open(os.path.join(out, '_EModelX(noseq)_ca_model.pdb'), 'w').write('pdb')\n"
                "open(os.path.join(out, 'time_cost__EModelX(noseq).csv'), 'w').write('t\\n')\n",
                encoding="utf-8",
            )
            map_path = root / "postprocess_masked.mrc"
            map_path.write_text("mrc", encoding="utf-8")
            out_dir_override = root / "runs" / "T1_rule_only"

            from agent.case_registry import ExperimentCase
            from agent.experiment_protocol import parse_protocol

            # Build a case pinned to these fixture paths (seq_free, no fasta).
            case = ExperimentCase(
                case_id="T1_tutorial5_map_emodelx_stopA",
                case_type="map_level_case",
                dataset_name="tutorial5",
                dataset_root=str(root),
                input_state={
                    "has_map": True,
                    "has_sequence": False,
                    "emodelx_protocol": "seq_free",
                },
                available_inputs={
                    "map_path": str(map_path),
                    "emodelx_repo_dir": str(repo),
                    "emodelx_python": "/usr/bin/python3",
                    "output_dir": str(out_dir_override / "output"),
                    "emodelx_protocol": "seq_free",
                },
                target_stage="model_building",
                evaluation_mode="success_only",
            )

            # Rule-mode protocol with all four backends allowed — this is
            # precisely the case where the old code would have picked
            # modelangelo and blocked at preflight.
            protocol = parse_protocol(
                {
                    "protocol_name": "rule_only_agent",
                    "protocol_kind": "rule_only_agent",
                    "allowed_backends": ["topaz", "cryodrgn", "emodelx", "mica"],
                    "routing_mode": "rule",
                    "llm_supervision_enabled": False,
                    "fallback_policy": "registry_default",
                }
            )

            scheduler = Scheduler(project_root=root)
            harness = BenchmarkHarness(
                executor=scheduler.action_executor,
                project_root=root,
            )
            result = run_experiment_case(
                protocol=protocol,
                case=case,
                harness=harness,
                project_root=root,
                output_dir=root / "out_exp",
                dry_run=False,
                persist=False,
                write_audit=True,
            )
            payload = result.to_dict()

            # Routing: emodelx selected (not the modelangelo fallback).
            self.assertEqual(payload["selected_backend"], "emodelx")
            sel = payload["case_result"]["extra"]["selection_decision"]
            self.assertIn("seq_free", sel["selection_reason"])
            self.assertFalse(sel["fallback_applied"])

            # Artifacts harvested.
            cr = payload["case_result"]
            artifact_names = [Path(p).name for p in (cr.get("artifacts") or [])]
            generated_names = [Path(p).name for p in (cr.get("generated_paths") or [])]
            self.assertIn("_EModelX(noseq)_ca_model.pdb", artifact_names)
            self.assertIn("time_cost__EModelX(noseq).csv", artifact_names)
            self.assertIn("_EModelX(noseq)_ca_model.pdb", generated_names)
            self.assertIn("time_cost__EModelX(noseq).csv", generated_names)

            # success_audit written with evidence_strength >= medium.
            audit_json = payload["audit_paths"].get("success_audit_json")
            self.assertIsNotNone(audit_json)
            audit = json.loads(Path(audit_json).read_text(encoding="utf-8"))
            strength = audit["success_classification"]["evidence_strength"]
            self.assertIn(strength, ("medium", "high"))
            self.assertNotEqual(strength, "low")


if __name__ == "__main__":
    unittest.main()
