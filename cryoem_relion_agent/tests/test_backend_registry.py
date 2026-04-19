"""Tests for centralized backend registry resolution."""

from __future__ import annotations

import unittest
from copy import deepcopy

from agent.backend_registry import (
    DEFAULT_DOMAINS,
    MODEL_BUILDING_BACKEND_METADATA,
    BackendRegistry,
    iter_model_building_backend_ids,
    model_action_to_backend,
    model_building_backend_to_action_name,
)


class TestBackendRegistry(unittest.TestCase):
    def test_default_resolution_picking_heterogeneity_model_building(self) -> None:
        reg = BackendRegistry(deepcopy(DEFAULT_DOMAINS))
        self.assertEqual(reg.resolve("picking", None)[0], "relion_log")
        self.assertEqual(reg.resolve("heterogeneity", None)[0], "human_review")
        self.assertEqual(reg.resolve("model_building", None)[0], "modelangelo")

    def test_explicit_valid_backends(self) -> None:
        reg = BackendRegistry(deepcopy(DEFAULT_DOMAINS))
        self.assertEqual(reg.resolve("picking", "topaz")[0], "topaz")
        self.assertEqual(reg.resolve("picking", "cryolo")[0], "cryolo")
        self.assertEqual(reg.resolve("heterogeneity", "cryodrgn")[0], "cryodrgn")
        self.assertEqual(reg.resolve("model_building", "emodelx")[0], "emodelx")

    def test_invalid_backend_falls_back_with_note(self) -> None:
        domains = deepcopy(DEFAULT_DOMAINS)
        domains["heterogeneity"]["allowed_backends"] = ["human_review"]
        domains["heterogeneity"]["default_backend"] = "human_review"
        reg = BackendRegistry(domains)
        backend, note = reg.resolve("heterogeneity", "cryodrgn")
        self.assertEqual(backend, "human_review")
        self.assertEqual(note, "fallback_invalid_backend")

    def test_model_action_to_backend(self) -> None:
        self.assertEqual(model_action_to_backend("run_modelangelo"), "modelangelo")
        self.assertEqual(model_action_to_backend("run_emodelx"), "emodelx")
        self.assertEqual(model_action_to_backend("run_mica"), "mica")

    def test_model_building_registry_metadata_lists_three_backends(self) -> None:
        for bid in iter_model_building_backend_ids():
            self.assertIn(bid, MODEL_BUILDING_BACKEND_METADATA)
            meta = MODEL_BUILDING_BACKEND_METADATA[bid]
            self.assertEqual(meta.get("backend_name"), bid)
            self.assertEqual(meta.get("category"), "model_building")

    def test_model_building_backend_to_action_name(self) -> None:
        self.assertEqual(model_building_backend_to_action_name("modelangelo"), "run_modelangelo")
        self.assertEqual(model_building_backend_to_action_name("emodelx"), "run_emodelx")
        self.assertEqual(model_building_backend_to_action_name("mica"), "run_mica")

    def test_model_building_resolve_mica_when_allowed(self) -> None:
        reg = BackendRegistry(deepcopy(DEFAULT_DOMAINS))
        self.assertEqual(reg.resolve("model_building", "mica")[0], "mica")

    def test_model_building_restricted_allowed_explicit_mica_falls_back(self) -> None:
        domains = deepcopy(DEFAULT_DOMAINS)
        domains["model_building"]["allowed_backends"] = ["modelangelo"]
        domains["model_building"]["default_backend"] = "modelangelo"
        reg = BackendRegistry(domains)
        backend, note = reg.resolve("model_building", "mica")
        self.assertEqual(backend, "modelangelo")
        self.assertEqual(note, "fallback_invalid_backend")


if __name__ == "__main__":
    unittest.main()
