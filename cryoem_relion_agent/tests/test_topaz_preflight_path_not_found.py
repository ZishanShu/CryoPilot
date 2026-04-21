"""Tests for the missing_required_inputs vs input_path_not_found distinction."""

from __future__ import annotations

import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.picking_preflight import preflight_topaz_pick


class TestTopazPreflightPathNotFound(unittest.TestCase):
    def test_missing_field_reports_missing_required_inputs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            r = preflight_topaz_pick(
                {
                    "particle_diameter_angstrom": 180,
                    "output_dir": str(root / "out"),
                },
                project_root=root,
                topaz_config_path=None,
            )
            self.assertFalse(r["available"])
            self.assertIn("micrographs_star_path", r["missing_inputs"])
            self.assertEqual(r["missing_input_paths"], [])
            self.assertIn("missing_required_inputs", str(r["blocked_reason"]))
            self.assertNotIn("input_path_not_found", str(r["blocked_reason"]))

    def test_present_field_with_nonexistent_path_reports_input_path_not_found(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            r = preflight_topaz_pick(
                {
                    "micrographs_star_path": str(root / "missing" / "mic.star"),
                    "particle_diameter_angstrom": 180,
                    "output_dir": str(root / "out"),
                },
                project_root=root,
                topaz_config_path=None,
            )
            self.assertFalse(r["available"])
            self.assertIn("micrographs_star_path", r["missing_input_paths"])
            self.assertNotIn("micrographs_star_path", r["missing_inputs"])
            blocked = str(r["blocked_reason"])
            self.assertIn("input_path_not_found: micrographs_star_path", blocked)
            self.assertNotIn("missing_required_inputs: micrographs_star_path", blocked)

    def test_available_result_exposes_empty_missing_input_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            star = root / "mic.star"
            star.write_text("data_\n", encoding="utf-8")
            fake_exe = root / "fake_topaz"
            fake_exe.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
            fake_exe.chmod(fake_exe.stat().st_mode | stat.S_IXUSR)
            r = preflight_topaz_pick(
                {
                    "micrographs_star_path": str(star),
                    "particle_diameter_angstrom": 180,
                    "output_dir": str(root / "out"),
                    "topaz_executable": str(fake_exe),
                },
                project_root=root,
                topaz_config_path=None,
            )
            self.assertTrue(r["available"])
            self.assertEqual(r["missing_inputs"], [])
            self.assertEqual(r["missing_input_paths"], [])


if __name__ == "__main__":
    unittest.main()
