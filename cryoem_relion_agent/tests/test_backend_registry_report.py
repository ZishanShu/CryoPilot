"""Reporter shows backend summaries consistently across routing domains."""

from __future__ import annotations

import unittest

from agent.reporter import Reporter
from schemas.action import Action, ActionName
from schemas.action_history import ActionHistoryEntry
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState


class TestBackendRegistryReport(unittest.TestCase):
    def test_report_shows_three_backend_lines_when_data_present(self) -> None:
        state = SampleState(project_id="p1", project_path="/tmp/p", stage="PostProcess", status="running")
        qc = QCMetrics(motioncorr_completed=True, ctf_completed=True)
        history = [
            ActionHistoryEntry(
                run_id="r1",
                stage="PostProcess",
                action_name="run_modelangelo",
                branch_name="modelangelo",
                output_summary="(dry_run model_building_backend=modelangelo)",
            ),
            ActionHistoryEntry(
                run_id="r1",
                stage="Refine3D",
                action_name="run_cryodrgn",
                branch_name="heterogeneity",
                output_summary="(heterogeneity_backend=cryodrgn)",
            ),
        ]
        next_action = Action(
            name=ActionName.RUN_AUTOPICK,
            reason="test",
            parameters={"picking_backend": "cryolo", "branch_name": "main"},
            priority=78,
        )
        report = Reporter().build_report(
            state=state,
            qc=qc,
            recent_action=None,
            next_action=next_action,
            action_history=history,
            current_run_id="r1",
        )
        md = report.to_markdown()
        self.assertIn("Picking backend: cryolo", md)
        self.assertIn("Heterogeneity backend: cryodrgn", md)
        self.assertIn("Model-building backend: modelangelo", md)

    def test_report_compact_when_no_backend_metadata(self) -> None:
        state = SampleState(project_id="p1", project_path="/tmp/p", stage="Class2D", status="running")
        qc = QCMetrics(class2d_completed=True)
        history = [
            ActionHistoryEntry(run_id="r2", stage="Class2D", action_name="run_class2d", branch_name="main"),
        ]
        report = Reporter().build_report(
            state=state,
            qc=qc,
            recent_action=None,
            next_action=None,
            action_history=history,
            current_run_id="r2",
        )
        md = report.to_markdown()
        self.assertIn("## Actions", md)
        self.assertNotIn("Picking backend:", md)
        self.assertNotIn("Heterogeneity backend:", md)
        self.assertNotIn("Model-building backend:", md)

    def test_model_building_backend_from_next_action(self) -> None:
        state = SampleState(project_id="p1", project_path="/tmp/p", stage="PostProcess", status="running")
        qc = QCMetrics(refine3d_completed=True)
        next_action = Action(
            name=ActionName.RUN_MODELANGELO,
            reason="build",
            parameters={"model_building_backend": "modelangelo", "branch_name": "modelangelo"},
            priority=58,
        )
        report = Reporter().build_report(
            state=state,
            qc=qc,
            recent_action=None,
            next_action=next_action,
            action_history=[],
            current_run_id=None,
        )
        self.assertIn("Model-building backend: modelangelo", report.to_markdown())

    def test_model_building_backend_emodelx_from_next_action(self) -> None:
        state = SampleState(project_id="p1", project_path="/tmp/p", stage="PostProcess", status="running")
        qc = QCMetrics(refine3d_completed=True)
        next_action = Action(
            name=ActionName.RUN_EMODELX,
            reason="multi-chain",
            parameters={
                "model_building_backend": "emodelx",
                "branch_name": "emodelx",
                "map_path": "/m.mrc",
                "sequence_path": "/s.fasta",
                "model_building_availability": "available",
                "model_building_execution_mode": "real",
                "model_building_selection_decision": {"selected_backend": "emodelx", "selection_reason": "preferred_backend_selected"},
            },
            priority=80,
        )
        report = Reporter().build_report(
            state=state,
            qc=qc,
            recent_action=None,
            next_action=next_action,
            action_history=[],
            current_run_id=None,
        )
        md = report.to_markdown()
        self.assertIn("Model-building backend: emodelx", md)
        self.assertIn("Availability: available", md)
        self.assertIn("Execution mode: real", md)
        self.assertIn("Model-building selection reason:", md)
        self.assertIn("Output path: /tmp/out", Reporter().build_report(
            state=state,
            qc=qc,
            recent_action=None,
            next_action=Action(
                name=ActionName.RUN_EMODELX,
                reason="multi-chain",
                parameters={
                    "model_building_backend": "emodelx",
                    "branch_name": "emodelx",
                    "emodelx_preflight": {
                        "backend_name": "emodelx",
                        "available": True,
                        "execution_mode": "real",
                        "repo_path": "/opt/EModelX",
                        "python_path": "/usr/bin/python3",
                        "command": "/usr/bin/python3 /opt/EModelX/run.py --protocol temp_free --EM_map /m.mrc --fasta /s.fasta --output_dir /tmp/out",
                        "output_path": "/tmp/out",
                    },
                },
                priority=80,
            ),
            action_history=[],
            current_run_id=None,
        ).to_markdown())

    def test_model_building_backend_mica_from_next_action(self) -> None:
        state = SampleState(project_id="p1", project_path="/tmp/p", stage="PostProcess", status="running")
        qc = QCMetrics(refine3d_completed=True)
        next_action = Action(
            name=ActionName.RUN_MICA,
            reason="high-res",
            parameters={
                "model_building_backend": "mica",
                "branch_name": "mica",
                "map_path": "/m.mrc",
                "sequence_path": "/s.fasta",
                "model_building_availability": "unavailable",
                "model_building_execution_mode": "unavailable",
                "model_building_blocked_reason": "missing_required_inputs: af3_dir",
                "model_building_missing_inputs": ["af3_dir"],
                "mica_preflight": {
                    "backend_name": "mica",
                    "available": False,
                    "execution_mode": "unavailable",
                    "blocked_reason": "missing_required_inputs: af3_dir",
                    "repo_path": "/opt/MICA",
                    "python_path": "/usr/bin/python3",
                    "af3_dir": None,
                },
            },
            priority=90,
        )
        report = Reporter().build_report(
            state=state,
            qc=qc,
            recent_action=None,
            next_action=next_action,
            action_history=[],
            current_run_id=None,
        )
        md = report.to_markdown()
        self.assertIn("Model-building backend: mica", md)
        self.assertIn("Availability: unavailable", md)
        self.assertIn("Blocked reason:", md)
        self.assertIn("af3_dir", md)
        self.assertIn("MICA repo: /opt/MICA", md)

    def test_model_building_backend_mica_available_report_fields(self) -> None:
        state = SampleState(project_id="p1", project_path="/tmp/p", stage="PostProcess", status="running")
        qc = QCMetrics(refine3d_completed=True)
        next_action = Action(
            name=ActionName.RUN_MICA,
            reason="af3 route",
            parameters={
                "model_building_backend": "mica",
                "branch_name": "mica",
                "model_building_availability": "available",
                "model_building_execution_mode": "real",
                "mica_preflight": {
                    "backend_name": "mica",
                    "available": True,
                    "execution_mode": "real",
                    "repo_path": "/opt/MICA",
                    "python_path": "/usr/bin/python3",
                    "run_py_path": "/opt/MICA/run.py",
                    "af3_dir": "/data/af3_results",
                    "command": "/usr/bin/python3 /opt/MICA/run.py -m /m.mrc -f /s.fasta -a /data/af3_results -o /out",
                    "output_path": "/out",
                },
            },
            priority=90,
        )
        report = Reporter().build_report(
            state=state,
            qc=qc,
            recent_action=None,
            next_action=next_action,
            action_history=[],
            current_run_id=None,
        )
        md = report.to_markdown()
        self.assertIn("Model-building backend: mica", md)
        self.assertIn("Availability: available", md)
        self.assertIn("Execution mode: real", md)
        self.assertIn("MICA repo: /opt/MICA", md)
        self.assertIn("AF3 dir: /data/af3_results", md)

    def test_model_building_backend_from_history_summary(self) -> None:
        state = SampleState(project_id="p1", project_path="/tmp/p", stage="PostProcess", status="running")
        qc = QCMetrics()
        history = [
            ActionHistoryEntry(
                run_id="r1",
                stage="PostProcess",
                action_name="run_emodelx",
                branch_name="emodelx",
                output_summary="(dry_run model_building_backend=emodelx)",
            ),
        ]
        report = Reporter().build_report(
            state=state,
            qc=qc,
            recent_action=None,
            next_action=None,
            action_history=history,
            current_run_id="r1",
        )
        self.assertIn("Model-building backend: emodelx", report.to_markdown())

    def test_preflight_parsed_from_history_output_summary_tags(self) -> None:
        state = SampleState(project_id="p1", project_path="/tmp/p", stage="PostProcess", status="running")
        qc = QCMetrics()
        history = [
            ActionHistoryEntry(
                run_id="r1",
                stage="PostProcess",
                action_name="run_emodelx",
                branch_name="emodelx",
                output_summary=(
                    "(status dry_run model_building_backend=emodelx, "
                    "model_building_availability=available, "
                    "model_building_execution_mode=real, "
                    "emodelx_repo=/opt/EModelX, emodelx_python=/usr/bin/python3)"
                ),
            ),
        ]
        report = Reporter().build_report(
            state=state,
            qc=qc,
            recent_action=None,
            next_action=None,
            action_history=history,
            current_run_id="r1",
        )
        md = report.to_markdown()
        self.assertIn("Availability: available", md)
        self.assertIn("Execution mode: real", md)
        self.assertIn("EModelX repo: /opt/EModelX", md)
        self.assertIn("Output path: /tmp/out", Reporter().build_report(
            state=state,
            qc=qc,
            recent_action=None,
            next_action=Action(
                name=ActionName.RUN_EMODELX,
                reason="multi-chain",
                parameters={
                    "model_building_backend": "emodelx",
                    "branch_name": "emodelx",
                    "emodelx_preflight": {
                        "backend_name": "emodelx",
                        "available": True,
                        "execution_mode": "real",
                        "repo_path": "/opt/EModelX",
                        "python_path": "/usr/bin/python3",
                        "command": "/usr/bin/python3 /opt/EModelX/run.py --protocol temp_free --EM_map /m.mrc --fasta /s.fasta --output_dir /tmp/out",
                        "output_path": "/tmp/out",
                    },
                },
                priority=80,
            ),
            action_history=[],
            current_run_id=None,
        ).to_markdown())


if __name__ == "__main__":
    unittest.main()
