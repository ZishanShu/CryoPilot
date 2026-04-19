"""Tests that generate-report does not persist sample_state.json by default."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.scheduler import Scheduler
from schemas.sample_state import SampleState
from tests.helpers.fixture_projects import build_fixture_project


class TestReportReadonlyState(unittest.TestCase):
    def test_generate_report_default_does_not_modify_sample_state_file(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            proj = build_fixture_project(root, "postprocess_ready")
            agent_dir = proj / ".cryoem_agent"
            agent_dir.mkdir(parents=True, exist_ok=True)
            state_path = agent_dir / "sample_state.json"
            sentinel = SampleState(
                project_id="fixture_pp",
                project_path=str(proj.resolve()),
                stage="Import",
                status="running",
            )
            raw = json.dumps(sentinel.model_dump(mode="json"), ensure_ascii=False, indent=2)
            state_path.write_text(raw, encoding="utf-8")
            before = state_path.read_bytes()

            sched = Scheduler(project_root=proj)
            report = sched.generate_report(refresh_state=False)
            after = state_path.read_bytes()

            self.assertEqual(before, after, msg="sample_state.json must not change on default generate_report")
            self.assertIsNotNone(report.next_action)
            self.assertTrue(bool(report.state.stage))

    def test_generate_report_refresh_state_persists_scanned_stage(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            proj = build_fixture_project(root, "postprocess_ready")
            agent_dir = proj / ".cryoem_agent"
            agent_dir.mkdir(parents=True, exist_ok=True)
            state_path = agent_dir / "sample_state.json"
            sentinel = SampleState(
                project_id="fixture_pp",
                project_path=str(proj.resolve()),
                stage="Import",
                status="running",
            )
            state_path.write_text(
                json.dumps(sentinel.model_dump(mode="json"), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            sched = Scheduler(project_root=proj)
            sched.generate_report(refresh_state=True)
            loaded = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded.get("stage"), "PostProcess")


if __name__ == "__main__":
    unittest.main()
