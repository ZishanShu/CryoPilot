"""Scheduler trace-link tests for structure routing + model branch."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.scheduler import Scheduler
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState


class TestStructureRouterScheduler(unittest.TestCase):
    def test_scheduler_records_structure_and_model_branches_with_links(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sched = Scheduler(project_root=root)
            state = SampleState(
                project_id="p1",
                project_path=str(root),
                stage="PostProcess",
                status="running",
                map_path=str(root / "map.mrc"),
                sequence_path=str(root / "seq.fasta"),
            )
            qc = QCMetrics(
                motioncorr_completed=True,
                ctf_completed=True,
                picking_completed=True,
                class2d_completed=True,
                init3d_completed=True,
                refine3d_completed=True,
                current_resolution=3.8,
            )
            sched.scan_project = lambda sequence_override=None, persist_state=True: (state, qc)  # type: ignore[method-assign]
            sched.action_executor.execute = lambda **_: {  # type: ignore[method-assign]
                "action_name": "run_modelangelo",
                "status": "dry_run",
                "output_dir": str(root / "out"),
            }
            _action, _result = sched.run_next(dry_run=True)
            items = sched.history_manager.retrieve_action_history()
            self.assertGreaterEqual(len(items), 3)
            run_ids = {x.run_id for x in items}
            self.assertEqual(len(run_ids), 1)

            router = [x for x in items if x.action_name == "route_structure"][-1]
            model = [x for x in items if x.action_name == "run_modelangelo"][-1]
            self.assertEqual(router.branch_name, "structure_router")
            self.assertEqual(model.branch_name, "modelangelo")
            self.assertEqual(model.parent_decision_id, router.entry_id)
            self.assertTrue(bool(model.previous_entry_id))


if __name__ == "__main__":
    unittest.main()

