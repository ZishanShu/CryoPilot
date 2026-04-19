"""Unit tests for state manager read/write behavior."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.state_manager import StateManager
from schemas.sample_state import SampleState


class TestStateManager(unittest.TestCase):
    """Ensure state persistence and recovery work as expected."""

    def test_save_then_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            manager = StateManager(state_file)

            state = SampleState(
                project_id="p1",
                project_path="/tmp/proj",
                stage="Class2D",
                status="running",
                n_micrographs=1200,
                n_particles_picked=220000,
                n_particles_after_2d=110000,
                n_particles_after_3d=0,
            )
            ok = manager.save_state(state)
            self.assertTrue(ok)

            loaded = manager.load_state()
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.project_id, "p1")
            self.assertEqual(loaded.stage, "Class2D")

    def test_load_missing_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(Path(tmpdir) / "missing.json")
            loaded = manager.load_state()
            self.assertIsNone(loaded)


if __name__ == "__main__":
    unittest.main()
