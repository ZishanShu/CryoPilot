"""Tests for sequence CLI injection and scheduler + structure routing integration."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.structure_router import ModelingContext
from app.scheduler import Scheduler
from schemas.action import Action, ActionName
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState


class TestSchedulerSequenceInjection(unittest.TestCase):
    """CLI ``--sequence`` maps to persisted SampleState.sequence_path."""

    def test_sequence_override_persists(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Class2D/job001").mkdir(parents=True)
            (root / "Class2D/job001/classes.star").write_text("data_\nloop_\n1\n", encoding="utf-8")
            fasta = root / "seq.fasta"
            fasta.write_text(">A\nM\n", encoding="utf-8")

            sched = Scheduler(project_root=root)
            sched.scan_project(sequence_override=fasta)
            loaded = sched.state_manager.load_state()
            assert loaded is not None
            self.assertEqual(loaded.sequence_path, str(fasta.resolve()))

    def test_sequence_override_replaces_previous(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Class2D/job001").mkdir(parents=True)
            (root / "Class2D/job001/classes.star").write_text("data_\nloop_\n1\n", encoding="utf-8")
            a = root / "a.fasta"
            b = root / "b.fasta"
            a.write_text(">A\nA\n", encoding="utf-8")
            b.write_text(">B\nB\n", encoding="utf-8")

            sched = Scheduler(project_root=root)
            sched.scan_project(sequence_override=a)
            sched.scan_project(sequence_override=b)
            loaded = sched.state_manager.load_state()
            assert loaded is not None
            self.assertEqual(loaded.sequence_path, str(b.resolve()))


class TestSchedulerStructureRouting(unittest.TestCase):
    """``_resolve_next_action`` returns router tool actions when policy delegates."""

    def setUp(self) -> None:
        self.base_state = SampleState(
            project_id="p1",
            project_path="/tmp/proj",
            stage="PostProcess",
            status="running",
            map_path="/tmp/map.mrc",
            sequence_path="/tmp/seq.fasta",
        )
        self.base_qc = QCMetrics(
            motioncorr_completed=True,
            ctf_completed=True,
            picking_completed=True,
            class2d_completed=True,
            init3d_completed=True,
            refine3d_completed=True,
            current_resolution=3.8,
        )

    def test_routes_modelangelo_default(self) -> None:
        sched = Scheduler(Path("/tmp"))
        action = sched._resolve_next_action(
            self.base_state,
            self.base_qc,
            ModelingContext(known_sequence=True, af3_available=False, chain_count=1),
        )
        self.assertEqual(action.name, ActionName.RUN_MODELANGELO)

    def test_routes_emodelx_multichain(self) -> None:
        sched = Scheduler(Path("/tmp"))
        action = sched._resolve_next_action(
            self.base_state,
            self.base_qc,
            ModelingContext(known_sequence=True, af3_available=False, chain_count=3),
        )
        self.assertEqual(action.name, ActionName.RUN_EMODELX)

    def test_routes_mica_af3_resolution(self) -> None:
        sched = Scheduler(Path("/tmp"))
        qc = self.base_qc.model_copy(update={"current_resolution": 3.4})
        action = sched._resolve_next_action(
            self.base_state,
            qc,
            ModelingContext(
                known_sequence=True,
                af3_available=True,
                chain_count=1,
                af3_dir="/tmp/af3_out",
            ),
        )
        self.assertEqual(action.name, ActionName.RUN_MICA)

    def test_explicit_preferred_emodelx_overrides_router_modelangelo(self) -> None:
        sched = Scheduler(Path("/tmp"))
        action = sched._resolve_next_action(
            self.base_state,
            self.base_qc,
            ModelingContext(
                known_sequence=True,
                af3_available=False,
                chain_count=1,
                preferred_model_building_backend="emodelx",
            ),
        )
        self.assertEqual(action.name, ActionName.RUN_EMODELX)
        self.assertEqual(action.parameters.get("model_building_backend"), "emodelx")
        self.assertEqual(action.parameters.get("model_building_availability"), "unavailable")
        self.assertEqual(action.parameters.get("model_building_execution_mode"), "unavailable")
        self.assertIsInstance(action.parameters.get("model_building_selection_decision"), dict)
        pf = action.parameters.get("model_building_preflight")
        self.assertIsInstance(pf, dict)
        self.assertIsInstance(action.parameters.get("emodelx_preflight"), dict)
        assert isinstance(pf, dict)
        self.assertFalse(pf.get("available"))

    def test_mica_missing_af3_preflight_unavailable(self) -> None:
        sched = Scheduler(Path("/tmp"))
        qc = self.base_qc.model_copy(update={"current_resolution": 3.4})
        action = sched._resolve_next_action(
            self.base_state,
            qc,
            ModelingContext(known_sequence=True, af3_available=True, chain_count=1),
        )
        self.assertEqual(action.name, ActionName.RUN_MICA)
        self.assertEqual(action.parameters.get("model_building_backend"), "mica")
        self.assertEqual(action.parameters.get("model_building_availability"), "unavailable")
        self.assertIn("af3_dir", action.parameters.get("model_building_missing_inputs") or [])

    def test_restricted_policy_maps_mica_route_to_modelangelo(self) -> None:
        import textwrap
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            rules = Path(tmp) / "policy_rules.yaml"
            rules.write_text(
                textwrap.dedent(
                    """
                    model_building:
                      default_backend: modelangelo
                      allowed_backends:
                        - modelangelo
                    """
                ).strip(),
                encoding="utf-8",
            )
            sched = Scheduler(Path("/tmp"))
            sched.policy_engine.rules = sched.policy_engine._load_rules(rules)
            sched.policy_engine._refresh_backend_registry()
            qc = self.base_qc.model_copy(update={"current_resolution": 3.4})
            action = sched._resolve_next_action(
                self.base_state,
                qc,
                ModelingContext(known_sequence=True, af3_available=True, chain_count=1),
            )
            self.assertEqual(action.name, ActionName.RUN_MODELANGELO)
            self.assertEqual(action.parameters.get("model_building_backend"), "modelangelo")
            self.assertEqual(action.parameters.get("model_building_resolution_note"), "fallback_invalid_backend")

    def test_validation_when_no_sequence(self) -> None:
        sched = Scheduler(Path("/tmp"))
        state = self.base_state.model_copy(update={"sequence_path": None})
        action = sched._resolve_next_action(state, self.base_qc, None)
        self.assertEqual(action.name, ActionName.RUN_VALIDATION)
        self.assertIn("missing_sequence", str(action.parameters.get("routing_reason")))

    def test_emodelx_resolved_available_when_inputs_ready(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_dir = root / "agent_root"
            cfg_dir = base_dir / "configs"
            cfg_dir.mkdir(parents=True)
            for name in ("policy_rules.yaml", "router_rules.yaml", "schemes.yaml"):
                src = Path(__file__).resolve().parents[1] / "configs" / name
                (cfg_dir / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            (cfg_dir / "emodelx.yaml").write_text(
                "emodelx_repo_dir: null\nemodelx_python: null\nemodelx_default_protocol: temp_free\n",
                encoding="utf-8",
            )
            repo = root / "EModelX"
            (repo / "models").mkdir(parents=True)
            (repo / "models" / "w.bin").write_text("x", encoding="utf-8")
            (repo / "run.py").write_text("print('ok')", encoding="utf-8")
            state = self.base_state.model_copy(
                update={
                    "project_path": str(root / "proj"),
                    "map_path": str(root / "m.mrc"),
                    "sequence_path": str(root / "s.fasta"),
                }
            )
            Path(state.map_path).write_text("map", encoding="utf-8")
            Path(state.sequence_path).write_text(">a\nA", encoding="utf-8")
            sched = Scheduler(project_root=root / "proj", base_dir=base_dir)
            action = sched._resolve_next_action(
                state,
                self.base_qc,
                ModelingContext(
                    known_sequence=True,
                    af3_available=False,
                    chain_count=1,
                    preferred_model_building_backend="emodelx",
                ),
            )
            self.assertEqual(action.name, ActionName.RUN_EMODELX)
            self.assertEqual(action.parameters.get("model_building_backend"), "emodelx")
            self.assertEqual(action.parameters.get("model_building_availability"), "available")
            self.assertEqual(action.parameters.get("model_building_execution_mode"), "real")
            self.assertTrue(bool(action.parameters.get("emodelx_repo_dir")))
            self.assertIsInstance(action.parameters.get("emodelx_preflight"), dict)

    def test_mica_resolved_available_when_inputs_ready(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_dir = root / "agent_root"
            cfg_dir = base_dir / "configs"
            cfg_dir.mkdir(parents=True)
            for name in ("policy_rules.yaml", "router_rules.yaml", "schemes.yaml"):
                src = Path(__file__).resolve().parents[1] / "configs" / name
                (cfg_dir / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            (cfg_dir / "mica.yaml").write_text(
                "mica_repo_dir: null\nmica_python: null\nmica_default_device: cpu\n",
                encoding="utf-8",
            )
            repo = root / "MICA"
            (repo / "trained_models").mkdir(parents=True)
            (repo / "trained_models" / "MICA_best_model.pth").write_text("x", encoding="utf-8")
            (repo / "run.py").write_text("print('ok')", encoding="utf-8")
            af3 = root / "af3_results"
            af3.mkdir(parents=True)
            (af3 / "m.cif").write_text("x", encoding="utf-8")
            state = self.base_state.model_copy(
                update={
                    "project_path": str(root / "proj"),
                    "map_path": str(root / "m.mrc"),
                    "sequence_path": str(root / "s.fasta"),
                }
            )
            Path(state.map_path).write_text("map", encoding="utf-8")
            Path(state.sequence_path).write_text(">a\nA", encoding="utf-8")
            sched = Scheduler(project_root=root / "proj", base_dir=base_dir)
            action = sched._resolve_next_action(
                state,
                self.base_qc.model_copy(update={"current_resolution": 3.4}),
                ModelingContext(
                    known_sequence=True,
                    af3_available=True,
                    chain_count=1,
                    af3_dir=str(af3),
                    preferred_model_building_backend="mica",
                ),
            )
            self.assertEqual(action.name, ActionName.RUN_MICA)
            self.assertEqual(action.parameters.get("model_building_backend"), "mica")
            self.assertEqual(action.parameters.get("model_building_availability"), "available")
            self.assertEqual(action.parameters.get("model_building_execution_mode"), "real")
            self.assertTrue(bool(action.parameters.get("mica_repo_dir")))
            self.assertIsInstance(action.parameters.get("mica_preflight"), dict)
            self.assertIsInstance(action.parameters.get("model_building_selection_decision"), dict)


class TestReporterRoutingFields(unittest.TestCase):
    """Report structure routing block includes trigger + sequence."""

    def test_report_shows_triggered_routing(self) -> None:
        from agent.reporter import Reporter

        state = SampleState(
            project_id="p1",
            project_path="/tmp/p",
            stage="PostProcess",
            status="running",
            map_path="/m.mrc",
            sequence_path="/s.fasta",
        )
        qc = QCMetrics(refine3d_completed=True, current_resolution=3.8)
        next_action = Action(
            name=ActionName.RUN_MODELANGELO,
            reason="Known sequence in general case; ModelAngelo default route.",
            parameters={"routing_triggered": True, "routing_reason": "router_default"},
            priority=70,
        )
        report = Reporter().build_report(state=state, qc=qc, recent_action=None, next_action=next_action)
        assert report.structure_routing is not None
        self.assertTrue(report.structure_routing.triggered)
        self.assertEqual(report.structure_routing.sequence_path, "/s.fasta")
        self.assertEqual(report.structure_routing.selected_tool, "run_modelangelo")
        md = report.to_markdown()
        self.assertIn("Sequence path:", md)
        self.assertIn("Chosen action:", md)


if __name__ == "__main__":
    unittest.main()
