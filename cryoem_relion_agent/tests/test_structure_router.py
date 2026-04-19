"""Unit tests for structure router rules."""

from __future__ import annotations

import unittest

from agent.reporter import Reporter
from agent.structure_router import ModelingContext, StructureRouter
from schemas.action import ActionName
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState


class TestStructureRouter(unittest.TestCase):
    """Validate route choices under representative contexts."""

    def setUp(self) -> None:
        self.router = StructureRouter(rules_path=None)
        self.state = SampleState(
            project_id="p1",
            project_path="/tmp/proj",
            stage="PostProcess",
            status="running",
            map_path="/tmp/proj/PostProcess/map.mrc",
            sequence_path="/tmp/proj/seq.fasta",
        )
        self.qc = QCMetrics(current_resolution=3.0)

    def test_unknown_sequence_routes_human(self) -> None:
        ctx = ModelingContext(known_sequence=False, af3_available=False)
        action = self.router.route(self.state, self.qc, ctx)
        self.assertEqual(action.name, ActionName.REQUEST_HUMAN_REVIEW)

    def test_mica_priority(self) -> None:
        ctx = ModelingContext(known_sequence=True, af3_available=True, chain_count=1, af3_dir="/tmp/af3")
        action = self.router.route(self.state, self.qc, ctx)
        self.assertEqual(action.name, ActionName.RUN_MICA)
        self.assertEqual(action.parameters.get("af3_dir"), "/tmp/af3")

    def test_emodelx_for_multichain(self) -> None:
        qc = QCMetrics(current_resolution=4.5)
        ctx = ModelingContext(known_sequence=True, af3_available=False, chain_count=3)
        action = self.router.route(self.state, qc, ctx)
        self.assertEqual(action.name, ActionName.RUN_EMODELX)
        self.assertEqual(action.parameters.get("sequence_path"), self.state.sequence_path)

    def test_default_modelangelo(self) -> None:
        qc = QCMetrics(current_resolution=4.2)
        ctx = ModelingContext(known_sequence=True, af3_available=False, chain_count=1)
        action = self.router.route(self.state, qc, ctx)
        self.assertEqual(action.name, ActionName.RUN_MODELANGELO)

    def test_can_route_structure_true_for_38a_with_sequence(self) -> None:
        qc = QCMetrics(refine3d_completed=True, current_resolution=3.8)
        ctx = ModelingContext(known_sequence=True, af3_available=False, chain_count=1)
        can_route, reason = self.router.can_route_structure(self.state, qc, ctx)
        self.assertTrue(can_route)
        self.assertEqual(reason, "map_and_sequence_ready")

    def test_report_contains_routing_reason_when_not_triggered(self) -> None:
        reporter = Reporter()
        state = self.state.model_copy()
        state.sequence_path = None
        qc = QCMetrics(refine3d_completed=True, current_resolution=3.8)
        report = reporter.build_report(state=state, qc=qc, recent_action=None, next_action=None)
        self.assertIsNotNone(report.structure_routing)
        assert report.structure_routing is not None
        self.assertEqual(report.structure_routing.rationale, "missing_sequence")


if __name__ == "__main__":
    unittest.main()
