from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from backends.reconstruction.cryodrgn_ai_backend import preflight


class TestCryodrgnAiPreflight(unittest.TestCase):
    def test_missing_particles_blocks_backend(self) -> None:
        with patch("backends.reconstruction.cryodrgn_ai_backend._resolve_executable", return_value="/bin/cryodrgn"):
            with patch("backends.reconstruction.cryodrgn_ai_backend._run_help", return_value=True):
                pf = preflight({"execution_mode": "dry_run"})
        self.assertFalse(pf["available"])
        self.assertIn("particles_star_or_mrcs", pf["missing_inputs"])

    def test_missing_ctf_blocks_unless_star_can_be_converted(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            mrcs = root / "particles.mrcs"
            mrcs.write_bytes(b"mrcs")
            with patch("backends.reconstruction.cryodrgn_ai_backend._resolve_executable", return_value="/bin/cryodrgn"):
                with patch("backends.reconstruction.cryodrgn_ai_backend._run_help", return_value=True):
                    pf = preflight({"particles_star_or_mrcs": str(mrcs), "execution_mode": "dry_run"})
            self.assertFalse(pf["available"])
            self.assertIn("ctf_metadata", pf["missing_inputs"])

            star = root / "particles.star"
            star.write_text("data_particles\n", encoding="utf-8")
            with patch("backends.reconstruction.cryodrgn_ai_backend._resolve_executable", return_value="/bin/cryodrgn"):
                with patch("backends.reconstruction.cryodrgn_ai_backend._run_help", return_value=True):
                    pf2 = preflight(
                        {
                            "particles_star_or_mrcs": str(star),
                            "datadir": str(root),
                            "apix": 1.1,
                            "box_size": 256,
                            "execution_mode": "dry_run",
                        }
                    )
            self.assertTrue(pf2["available"])
            self.assertEqual(pf2["input_mode"], "relion_star")

    def test_real_mode_warns_when_gpu_unavailable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            mrcs = root / "particles.mrcs"
            ctf = root / "ctf.pkl"
            mrcs.write_bytes(b"x")
            ctf.write_bytes(b"x")
            with patch("backends.reconstruction.cryodrgn_ai_backend._resolve_executable", return_value="/bin/cryodrgn"):
                with patch("backends.reconstruction.cryodrgn_ai_backend._run_help", return_value=True):
                    with patch("backends.reconstruction.cryodrgn_ai_backend._gpu_available", return_value=False):
                        pf = preflight(
                            {
                                "particles_star_or_mrcs": str(mrcs),
                                "ctf_metadata": str(ctf),
                                "execution_mode": "real",
                            }
                        )
            self.assertTrue(pf["available"])
            self.assertIn("gpu_unavailable", pf["warnings"])


if __name__ == "__main__":
    unittest.main()
