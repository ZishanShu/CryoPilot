"""Tests for agent.case_input_mapping (case -> backend contract bridge)."""

from __future__ import annotations

import unittest

from agent.case_input_mapping import (
    SEMANTIC_TO_CONTRACT,
    map_case_inputs_to_backend_params,
    supported_backends,
)


class TestMapping(unittest.TestCase):
    def test_topaz_semantic_micrographs_star_is_translated(self) -> None:
        mapped = map_case_inputs_to_backend_params(
            available_inputs={
                "micrographs_star": "/d/F1/micrographs.star",
                "particle_diameter_angstrom": 200,
                "output_dir": "/d/F1/picks",
            },
            backend="topaz",
        )
        self.assertEqual(mapped["micrographs_star_path"], "/d/F1/micrographs.star")
        self.assertEqual(mapped["particle_diameter_angstrom"], 200)
        self.assertEqual(mapped["output_dir"], "/d/F1/picks")

    def test_same_name_contract_field_wins_over_alias(self) -> None:
        mapped = map_case_inputs_to_backend_params(
            available_inputs={
                "micrographs_star": "/d/alias.star",
                "micrographs_star_path": "/d/explicit.star",
            },
            backend="topaz",
        )
        self.assertEqual(mapped["micrographs_star_path"], "/d/explicit.star")

    def test_bool_flags_are_dropped(self) -> None:
        # ``True``/``False`` in available_inputs are registry presence flags, not
        # real paths, so they must never leak into backend params.
        mapped = map_case_inputs_to_backend_params(
            available_inputs={
                "micrographs_star": True,
                "micrographs_star_path": False,
            },
            backend="topaz",
        )
        self.assertEqual(mapped, {})

    def test_unknown_keys_are_ignored(self) -> None:
        mapped = map_case_inputs_to_backend_params(
            available_inputs={"not_a_real_field": "x", "micrographs_star": "/d/m.star"},
            backend="topaz",
        )
        self.assertNotIn("not_a_real_field", mapped)
        self.assertEqual(mapped["micrographs_star_path"], "/d/m.star")

    def test_mica_map_sequence_and_af3_dir(self) -> None:
        mapped = map_case_inputs_to_backend_params(
            available_inputs={
                "map_path": "/m.mrc",
                "sequence_path": "/s.fasta",
                "af3_dir": False,
                "mica_repo_dir": "/opt/mica",
            },
            backend="mica",
        )
        self.assertEqual(mapped["map_path"], "/m.mrc")
        self.assertEqual(mapped["sequence_path"], "/s.fasta")
        self.assertEqual(mapped["mica_repo_dir"], "/opt/mica")
        # af3_dir: False is a flag-of-absence, must be dropped so preflight
        # reports it as missing_required_inputs rather than "False".
        self.assertNotIn("af3_dir", mapped)

    def test_empty_string_is_dropped(self) -> None:
        mapped = map_case_inputs_to_backend_params(
            available_inputs={"micrographs_star": "   ", "micrographs_star_path": ""},
            backend="topaz",
        )
        self.assertEqual(mapped, {})

    def test_supported_backends_covers_core_set(self) -> None:
        backends = set(supported_backends())
        for expected in ("topaz", "cryodrgn", "modelangelo", "emodelx", "mica"):
            self.assertIn(expected, backends)

    def test_mapping_is_pure_not_mutating_input(self) -> None:
        src = {"micrographs_star": "/d/m.star"}
        _ = map_case_inputs_to_backend_params(available_inputs=src, backend="topaz")
        self.assertEqual(src, {"micrographs_star": "/d/m.star"})

    def test_semantic_map_does_not_define_contract_only_collisions(self) -> None:
        # Sanity: no alias rule maps A -> A (would be a no-op mistake).
        for backend, aliases in SEMANTIC_TO_CONTRACT.items():
            for semantic, contract in aliases.items():
                self.assertNotEqual(
                    semantic, contract, f"{backend}: {semantic} aliased to itself"
                )


if __name__ == "__main__":
    unittest.main()
