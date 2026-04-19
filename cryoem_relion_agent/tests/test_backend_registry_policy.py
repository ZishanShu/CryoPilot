"""Policy engine uses centralized backend registry for routing domains."""

from __future__ import annotations

import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.policy_engine import PolicyEngine
from schemas.sample_state import SampleState


class TestBackendRegistryPolicy(unittest.TestCase):
    def setUp(self) -> None:
        self.state = SampleState(
            project_id="p1",
            project_path="/tmp/proj",
            stage="CtfFind",
            status="running",
        )

    def test_defaults_preserved_via_policy_selectors(self) -> None:
        engine = PolicyEngine(rules_path=None)
        self.assertEqual(engine._select_picking_backend(self.state, preferred_backend=None), "relion_log")
        self.assertEqual(engine._select_heterogeneity_backend(self.state, preferred_backend=None), "human_review")
        mb, note = engine.backend_registry.resolve("model_building", None, self.state.project_id)
        self.assertEqual(mb, "modelangelo")
        self.assertIsNone(note)

    def test_explicit_allowed_picking_and_heterogeneity(self) -> None:
        engine = PolicyEngine(rules_path=None)
        self.assertEqual(engine._select_picking_backend(self.state, preferred_backend="topaz"), "topaz")
        self.assertEqual(engine._select_heterogeneity_backend(self.state, preferred_backend="cryodrgn"), "cryodrgn")

    def test_restricted_allowed_list_falls_back(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rules.yaml"
            p.write_text(
                textwrap.dedent(
                    """
                    picking:
                      default_backend: relion_log
                      allowed_backends:
                        - relion_log
                    """
                ).strip(),
                encoding="utf-8",
            )
            engine = PolicyEngine(rules_path=p)
        self.assertEqual(engine._select_picking_backend(self.state, preferred_backend="topaz"), "relion_log")

    def test_select_model_building_explicit_emodelx(self) -> None:
        engine = PolicyEngine(rules_path=None)
        self.assertEqual(engine._select_model_building_backend(self.state, preferred_backend="emodelx"), "emodelx")

    def test_select_model_building_restricted_mica_falls_back(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rules.yaml"
            p.write_text(
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
            engine = PolicyEngine(rules_path=p)
        self.assertEqual(engine._select_model_building_backend(self.state, preferred_backend="mica"), "modelangelo")


if __name__ == "__main__":
    unittest.main()
