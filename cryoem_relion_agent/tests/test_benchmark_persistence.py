"""Tests for benchmark persistence to .cryoem_agent/benchmarks."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from agent.backend_registry import get_backend_capability_matrix
from agent.benchmark_persistence import (
    build_persistence_payload,
    collect_environment_snapshot,
    diff_benchmark_payloads,
    load_latest_benchmark,
    make_run_id,
    persist_benchmark_run,
    render_benchmark_diff_markdown,
    summarize_case_diff,
)
from app.scheduler import Scheduler


class TestBenchmarkPersistenceBasics(unittest.TestCase):
    def test_persist_creates_latest_and_history_files(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sched = Scheduler(project_root=root)
            summary = {
                "case_count": 1,
                "completed": 0,
                "blocked": 1,
                "failed": 0,
                "results": [
                    {
                        "benchmark_id": "case1",
                        "category": "model_building",
                        "backend": "emodelx",
                        "status": "preflight_blocked",
                        "execution_mode": "unavailable",
                        "runtime_seconds": 0.1,
                        "blocked_reason": "missing_inputs",
                        "command": None,
                        "artifacts": [],
                        "preflight": {},
                        "selection_decision": None,
                        "notes": "test",
                    }
                ],
                "markdown": "# Backend Benchmark Summary\n\n- Cases: 1\n",
            }
            cap = get_backend_capability_matrix()
            info = persist_benchmark_run(
                project_root=root,
                summary=summary,
                capability_matrix=cap,
                dry_run=True,
            )
            latest_json = Path(info["latest_json"])
            latest_md = Path(info["latest_md"])
            hist_json = Path(info["history_json"])
            hist_md = Path(info["history_md"])
            self.assertTrue(latest_json.is_file())
            self.assertTrue(latest_md.is_file())
            self.assertTrue(hist_json.is_file())
            self.assertTrue(hist_md.is_file())
            # Second persist should append a new history file but overwrite latest.
            info2 = persist_benchmark_run(
                project_root=root,
                summary=summary,
                capability_matrix=cap,
                dry_run=False,
            )
            # history paths must differ even when two calls are close in time.
            self.assertNotEqual(info["history_json"], info2["history_json"])
            self.assertTrue(Path(info2["history_json"]).is_file())
            # latest should always point to the second run.
            latest = Path(info2["latest_json"])
            self.assertTrue(latest.is_file())

    def test_latest_json_payload_has_required_top_level_keys(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cap = get_backend_capability_matrix()
            summary = {"case_count": 0, "completed": 0, "blocked": 0, "failed": 0, "results": [], "markdown": ""}
            persist_benchmark_run(
                project_root=root,
                summary=summary,
                capability_matrix=cap,
                dry_run=True,
            )
            latest = load_latest_benchmark(root)
            self.assertIsInstance(latest, dict)
            for key in (
                "run_metadata",
                "benchmark_summary",
                "benchmark_results",
                "capability_snapshot",
                "selection_snapshot",
                "environment_snapshot",
                "previous_run_metadata",
                "benchmark_diff",
                "regression_summary",
            ):
                self.assertIn(key, latest)
            env = latest["environment_snapshot"]
            self.assertIn("git_commit", env)
            self.assertIn("python_version", env)
            self.assertIn("platform", env)

    def test_extra_smoke_fields_are_persisted(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = {
                "case_count": 1,
                "completed": 1,
                "blocked": 0,
                "failed": 0,
                "results": [
                    {
                        "benchmark_id": "cryoppp_topaz_smoke_ds1",
                        "category": "picking",
                        "backend": "topaz",
                        "status": "completed",
                        "execution_mode": "real",
                        "runtime_seconds": 1.23,
                        "blocked_reason": None,
                        "command": "topaz predict ...",
                        "artifacts": ["/tmp/pred.txt"],
                        "generated_paths": ["/tmp/pred.txt"],
                        "topaz_output_path": "/tmp/pred.txt",
                        "return_code": 0,
                        "preflight": {"available": True},
                        "selection_decision": {"selected_backend": "topaz"},
                        "notes": "smoke",
                        "extra": {"smoke_run_checks": {"smoke_run_success": True}},
                    }
                ],
                "markdown": "",
            }
            persist_benchmark_run(
                project_root=root,
                summary=summary,
                capability_matrix=get_backend_capability_matrix(),
                dry_run=False,
            )
            latest = load_latest_benchmark(root) or {}
            result = (latest.get("benchmark_results") or [])[0]
            self.assertIn("extra", result)
            self.assertIn("topaz_output_path", result)
            self.assertIn("generated_paths", result)


class TestGitMetadataTolerance(unittest.TestCase):
    def test_collect_environment_snapshot_tolerates_missing_git(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap = collect_environment_snapshot(root)
            self.assertIn("git_commit", snap)
            # Unknown git should not raise, just mark unknown.
            self.assertIsInstance(snap.get("git_commit"), str)

    def test_make_run_id_stable_format(self) -> None:
        rid = make_run_id()
        self.assertTrue(rid.startswith("bench_"))
        self.assertIn("Z_", rid)


class TestHistoryUniquenessWithSameRunId(unittest.TestCase):
    def test_same_run_id_generates_unique_history_files(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cap = get_backend_capability_matrix()
            summary = {"case_count": 1, "completed": 1, "blocked": 0, "failed": 0, "results": [], "markdown": ""}
            rid = "bench_fixed_run"
            info1 = persist_benchmark_run(
                project_root=root,
                summary=summary,
                capability_matrix=cap,
                dry_run=False,
                run_id=rid,
            )
            info2 = persist_benchmark_run(
                project_root=root,
                summary=summary,
                capability_matrix=cap,
                dry_run=False,
                run_id=rid,
            )
            self.assertNotEqual(info1["history_json"], info2["history_json"])
            self.assertTrue(Path(info1["history_json"]).is_file())
            self.assertTrue(Path(info2["history_json"]).is_file())


class TestBenchmarkDiffs(unittest.TestCase):
    def test_first_run_diff_is_supported(self) -> None:
        current = {
            "run_metadata": {"run_id": "r1"},
            "benchmark_summary": {"total_cases": 1, "completed": 0, "blocked": 1, "failed": 0},
            "benchmark_results": [],
            "capability_snapshot": {},
            "selection_snapshot": {},
        }
        out = diff_benchmark_payloads(current, None)
        self.assertTrue(out["regression_summary"]["first_run"])
        self.assertEqual(out["benchmark_diff"]["overall_status_change"], "first_run")
        md = render_benchmark_diff_markdown({**current, **out})
        self.assertIn("No previous benchmark run to compare.", md)

    def test_run_to_run_status_and_selection_diff(self) -> None:
        previous = {
            "run_metadata": {"run_id": "prev", "timestamp": "t0"},
            "environment_snapshot": {"git_commit": "c0", "git_branch": "b0"},
            "benchmark_summary": {
                "total_cases": 2,
                "completed": 1,
                "blocked": 1,
                "failed": 0,
                "backends_covered": ["mica"],
                "categories_covered": ["model_building"],
            },
            "benchmark_results": [
                {
                    "benchmark_id": "mica_smoke_case",
                    "backend": "mica",
                    "category": "model_building",
                    "status": "completed",
                    "execution_mode": "real",
                    "runtime_seconds": 1.0,
                    "blocked_reason": None,
                    "command": "cmd1",
                    "artifacts": ["a"],
                    "selection_decision": {"selected_backend": "mica", "selection_reason": "af3", "fallback_applied": False},
                },
                {
                    "benchmark_id": "emodelx_smoke_case",
                    "backend": "emodelx",
                    "category": "model_building",
                    "status": "preflight_blocked",
                    "execution_mode": "unavailable",
                    "runtime_seconds": 0.5,
                    "blocked_reason": "missing",
                    "command": None,
                    "artifacts": [],
                    "selection_decision": {"selected_backend": "modelangelo", "selection_reason": "fallback", "fallback_applied": True},
                },
            ],
            "capability_snapshot": {
                "mica": {"maturity_level": "real", "supports_af3_prior": True, "required_inputs": ["map_path"], "required_dependencies": ["mica_repo_dir"], "execution_mode_supported": ["real"], "supports_map_input": True, "supports_sequence_input": True, "supports_particle_input": False, "supports_consensus_pose": False, "fallback_candidates": ["emodelx"]},
            },
            "selection_snapshot": {
                "mica_smoke_case": {"selected_backend": "mica", "selection_reason": "af3", "fallback_applied": False},
                "emodelx_smoke_case": {"selected_backend": "modelangelo", "selection_reason": "fallback", "fallback_applied": True},
            },
        }
        current = {
            "run_metadata": {"run_id": "curr"},
            "benchmark_summary": {
                "total_cases": 2,
                "completed": 1,
                "blocked": 1,
                "failed": 0,
                "backends_covered": ["emodelx"],
                "categories_covered": ["model_building"],
            },
            "benchmark_results": [
                {
                    "benchmark_id": "mica_smoke_case",
                    "backend": "mica",
                    "category": "model_building",
                    "status": "preflight_blocked",
                    "execution_mode": "unavailable",
                    "runtime_seconds": 2.0,
                    "blocked_reason": "missing",
                    "command": None,
                    "artifacts": [],
                    "selection_decision": {"selected_backend": "emodelx", "selection_reason": "fallback", "fallback_applied": True},
                },
                {
                    "benchmark_id": "emodelx_smoke_case",
                    "backend": "emodelx",
                    "category": "model_building",
                    "status": "completed",
                    "execution_mode": "real",
                    "runtime_seconds": 0.8,
                    "blocked_reason": None,
                    "command": "cmd2",
                    "artifacts": ["b"],
                    "selection_decision": {"selected_backend": "emodelx", "selection_reason": "map_sequence_present_and_emodelx_available", "fallback_applied": False},
                },
            ],
            "capability_snapshot": {
                "mica": {"maturity_level": "real", "supports_af3_prior": False, "required_inputs": ["map_path"], "required_dependencies": ["mica_repo_dir"], "execution_mode_supported": ["real"], "supports_map_input": True, "supports_sequence_input": True, "supports_particle_input": False, "supports_consensus_pose": False, "fallback_candidates": ["emodelx"]},
                "emodelx": {"maturity_level": "real", "supports_af3_prior": False, "required_inputs": ["map_path"], "required_dependencies": ["emodelx_repo_dir"], "execution_mode_supported": ["real"], "supports_map_input": True, "supports_sequence_input": True, "supports_particle_input": False, "supports_consensus_pose": False, "fallback_candidates": ["modelangelo"]},
            },
            "selection_snapshot": {
                "mica_smoke_case": {"selected_backend": "emodelx", "selection_reason": "fallback", "fallback_applied": True},
                "emodelx_smoke_case": {"selected_backend": "emodelx", "selection_reason": "map_sequence_present_and_emodelx_available", "fallback_applied": False},
            },
        }
        out = diff_benchmark_payloads(current, previous)
        self.assertIn("mica_smoke_case", out["regression_summary"]["regressed_cases"])
        self.assertIn("emodelx_smoke_case", out["regression_summary"]["improved_cases"])
        self.assertIn("mica", out["benchmark_diff"]["summary_diffs"]["backend_set_removed"])
        self.assertIn("emodelx", out["benchmark_diff"]["summary_diffs"]["backend_set_added"])
        self.assertIn("mica", out["benchmark_diff"]["capability_diffs"]["changed_backends"])
        self.assertIn("supports_af3_prior", out["benchmark_diff"]["capability_diffs"]["changed_fields_by_backend"]["mica"])
        self.assertIn("mica_smoke_case", out["benchmark_diff"]["selection_diffs"]["cases_with_selection_change"])

    def test_summarize_case_diff_new_removed_and_runtime(self) -> None:
        new_case = summarize_case_diff({"benchmark_id": "x", "status": "completed", "runtime_seconds": 2.0}, None)
        removed_case = summarize_case_diff(None, {"benchmark_id": "y", "status": "completed", "runtime_seconds": 1.0})
        changed = summarize_case_diff(
            {"benchmark_id": "z", "status": "completed", "runtime_seconds": 2.2, "selection_decision": {"selected_backend": "a"}},
            {"benchmark_id": "z", "status": "preflight_blocked", "runtime_seconds": 1.1, "selection_decision": {"selected_backend": "b"}},
        )
        self.assertEqual(new_case["diff_class"], "new_case")
        self.assertEqual(removed_case["diff_class"], "removed_case")
        self.assertEqual(changed["diff_class"], "improved")
        self.assertTrue(changed["selection_changed"])
        self.assertAlmostEqual(changed["runtime_seconds_delta"], 1.1, places=4)


class TestReportIntegrationWithLatest(unittest.TestCase):
    def test_report_reads_latest_when_in_memory_missing(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Write a minimal latest.json manually.
            bench_root = Path(root) / ".cryoem_agent" / "benchmarks"
            bench_root.mkdir(parents=True, exist_ok=True)
            payload = {
                "run_metadata": {"run_id": "r1", "timestamp": "2026-01-01T00:00:00+00:00"},
                "benchmark_summary": {
                    "total_cases": 3,
                    "completed": 1,
                    "blocked": 2,
                    "failed": 0,
                    "backends_covered": ["emodelx"],
                    "categories_covered": ["model_building"],
                    "summary_markdown": "# Backend Benchmark Summary\n",
                },
                "benchmark_results": [],
                "capability_snapshot": {},
                "selection_snapshot": {},
                "environment_snapshot": {},
                "previous_run_metadata": {"previous_run_id": "r0"},
                "benchmark_diff": {},
                "regression_summary": {
                    "first_run": False,
                    "improved_cases": ["a"],
                    "regressed_cases": ["b"],
                    "unchanged_cases": ["c"],
                    "new_cases": [],
                    "removed_cases": [],
                },
            }
            (bench_root / "latest.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            sched = Scheduler(project_root=root)
            # Clear any in-memory benchmark summary.
            sched._last_benchmark_summary = None
            report = sched.generate_report()
            md = report.to_markdown()
            self.assertIn("## Benchmark", md)
            self.assertIn("Cases: 3", md)
            self.assertIn("Compared to: r0", md)
            self.assertIn("Regressed: 1", md)

    def test_report_first_run_message(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_root = Path(root) / ".cryoem_agent" / "benchmarks"
            bench_root.mkdir(parents=True, exist_ok=True)
            payload = {
                "run_metadata": {"run_id": "r1", "timestamp": "2026-01-01T00:00:00+00:00"},
                "benchmark_summary": {"total_cases": 1, "completed": 0, "blocked": 1, "failed": 0, "backends_covered": [], "categories_covered": []},
                "benchmark_results": [],
                "capability_snapshot": {},
                "selection_snapshot": {},
                "environment_snapshot": {},
                "previous_run_metadata": {"first_run": True},
                "benchmark_diff": {"overall_status_change": "first_run"},
                "regression_summary": {"first_run": True, "improved_cases": [], "regressed_cases": [], "unchanged_cases": [], "new_cases": [], "removed_cases": []},
            }
            (bench_root / "latest.json").write_text(json.dumps(payload), encoding="utf-8")
            sched = Scheduler(project_root=root)
            sched._last_benchmark_summary = None
            report = sched.generate_report()
            self.assertIn("No previous benchmark run to compare.", report.to_markdown())


class TestSchedulerRunBenchmarksWritesFiles(unittest.TestCase):
    def test_scheduler_run_benchmarks_persists_files(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sched = Scheduler(project_root=root)
            summary = sched.run_benchmarks(dry_run=True)
            self.assertIn("case_count", summary)
            bench_root = Path(root) / ".cryoem_agent" / "benchmarks"
            self.assertTrue((bench_root / "latest.json").is_file())
            self.assertTrue((bench_root / "latest.md").is_file())
            latest = json.loads((bench_root / "latest.json").read_text(encoding="utf-8"))
            self.assertIn("previous_run_metadata", latest)
            self.assertIn("benchmark_diff", latest)
            self.assertIn("regression_summary", latest)


if __name__ == "__main__":
    unittest.main()

