"""Integration tests for QCExtractor merge logic."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from relion.qc_extractor import QCExtractor


class TestQCExtractorIntegration(unittest.TestCase):
    def test_resolution_none_and_extractor_does_not_crash(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Class2D/job001").mkdir(parents=True)
            (root / "Class2D/job001/classes.star").write_text(
                "data_\nloop_\n_rlnClassNumber #1\n1\n",
                encoding="utf-8",
            )
            ext = QCExtractor()
            state, qc = ext.extract(root)
            self.assertIsNone(qc.current_resolution)
            self.assertIsNotNone(state)

    def test_postprocess_star_yields_plausible_resolution(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pp = root / "PostProcess/job035/postprocess.star"
            pp.parent.mkdir(parents=True)
            pp.write_text(
                "\n".join(
                    [
                        "data_general",
                        "_rlnFinalResolution                               3.387915",
                        "_rlnPostprocessedMapMasked PostProcess/job035/postprocess_masked.mrc",
                    ]
                ),
                encoding="utf-8",
            )
            ext = QCExtractor()
            state, qc = ext.extract(root)
            self.assertIsNotNone(qc.current_resolution)
            self.assertAlmostEqual(float(qc.current_resolution), 3.387915, places=3)


class TestTutorial50Project(unittest.TestCase):
    """Optional smoke test on real tutorial tree when present."""

    _TUTORIAL = Path("/root/share/szs/CryoAgent/Tutorial5.0")

    def test_tutorial_resolution_not_1_0_and_micrographs_sane(self) -> None:
        if not self._TUTORIAL.is_dir():
            self.skipTest("Tutorial5.0 not present")
        ext = QCExtractor()
        state, qc = ext.extract(self._TUTORIAL)
        if qc.current_resolution is not None:
            self.assertGreater(abs(float(qc.current_resolution) - 1.0), 0.05)
            self.assertGreater(float(qc.current_resolution), 1.5)
        self.assertLess(state.n_micrographs, 500_000)
        self.assertGreater(state.n_micrographs, 0)


if __name__ == "__main__":
    unittest.main()
