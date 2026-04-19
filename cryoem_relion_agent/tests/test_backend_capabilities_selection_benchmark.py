"""Capability matrix, selection policy, and benchmark harness tests."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.backend_registry import get_backend_capability_matrix
from agent.benchmark_harness import BenchmarkHarness, default_smoke_cases
from agent.policy_engine import PolicyEngine
from app.scheduler import Scheduler
from schemas.sample_state import SampleState


class TestBackendCapabilityMatrix(unittest.TestCase):
    def test_required_backends_exist(self) -> None:
        m = get_backend_capability_matrix()
        for bid in ("relion_log", "topaz", "cryolo", "human_review", "cryodrgn", "modelangelo", "emodelx", "mica"):
            self.assertIn(bid, m)
        self.assertTrue(m["mica"]["supports_af3_prior"])
        self.assertTrue(m["emodelx"]["supports_map_input"])
        self.assertTrue(m["emodelx"]["supports_sequence_input"])


class TestSelectionPolicy(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = PolicyEngine(rules_path=None)
        self.state = SampleState(project_id="p1", project_path="/tmp/proj", stage="PostProcess", status="running")

    def test_model_building_prefers_mica_with_af3(self) -> None:
        dec = self.engine.select_backend_for_category(
            category="model_building",
            state=self.state.model_copy(update={"map_path": "/m.mrc", "sequence_path": "/s.fasta"}),
            context={"af3_dir": "/af3"},
            availability_snapshot={"mica": {"available": True}, "emodelx": {"available": True}},
        )
        self.assertEqual(dec["selected_backend"], "mica")
        self.assertIn("mica", dec["candidate_backends"])

    def test_model_building_prefers_emodelx_without_af3(self) -> None:
        dec = self.engine.select_backend_for_category(
            category="model_building",
            state=self.state.model_copy(update={"map_path": "/m.mrc", "sequence_path": "/s.fasta"}),
            context={},
            availability_snapshot={"mica": {"available": False}, "emodelx": {"available": True}},
        )
        self.assertEqual(dec["selected_backend"], "emodelx")

    def test_heterogeneity_fallback_when_cryodrgn_unavailable(self) -> None:
        dec = self.engine.select_backend_for_category(
            category="heterogeneity",
            state=self.state,
            context={"particles_star_path": "/p.star"},
            availability_snapshot={"cryodrgn": {"available": False}, "human_review": {"available": True}},
        )
        self.assertEqual(dec["selected_backend"], "human_review")
        self.assertIn("cryodrgn", dec["rejected_backends"])
        self.assertEqual(dec["rejection_reasons"].get("cryodrgn"), "unavailable")

    def test_picking_default_relion_when_topaz_unavailable(self) -> None:
        dec = self.engine.select_backend_for_category(
            category="picking",
            state=self.state,
            context={"micrographs_star_path": "/m.star"},
            availability_snapshot={"topaz": {"available": False}, "relion_log": {"available": True}},
        )
        self.assertEqual(dec["selected_backend"], "relion_log")
        self.assertIn("topaz", dec["rejected_backends"])

    def test_preferred_backend_honored_when_available(self) -> None:
        dec = self.engine.select_backend_for_category(
            category="model_building",
            state=self.state.model_copy(update={"map_path": "/m.mrc", "sequence_path": "/s.fasta"}),
            context={"af3_dir": "/af3"},
            preferred_backend="emodelx",
            availability_snapshot={"emodelx": {"available": True}, "mica": {"available": True}},
        )
        self.assertEqual(dec["selected_backend"], "emodelx")
        self.assertEqual(dec["selection_reason"], "preferred_backend_selected")


class TestBenchmarkHarness(unittest.TestCase):
    def test_smoke_benchmark_runs_and_summarizes(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sched = Scheduler(project_root=root)
            summary = sched.run_benchmarks(dry_run=False)
            self.assertGreaterEqual(summary.get("case_count", 0), 4)
            self.assertIn("results", summary)
            self.assertIn("markdown", summary)
            self.assertTrue(summary["markdown"].startswith("# Backend Benchmark Summary"))

    def test_harness_result_fields(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            h = BenchmarkHarness(executor=Scheduler(project_root=root).action_executor, project_root=root)
            case = default_smoke_cases()[0]
            r = h.run_case(case, dry_run=False)
            for k in ("status", "execution_mode", "runtime_seconds", "benchmark_id", "backend", "category"):
                self.assertIn(k, r)


if __name__ == "__main__":
    unittest.main()

