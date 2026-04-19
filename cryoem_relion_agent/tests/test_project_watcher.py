"""Tests for ProjectWatcher micrograph counting and map preference."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from relion.project_watcher import ProjectWatcher


_MICROGRAPH_STAR = """
# version

data_micrographs

loop_
_rlnMicrographName #1
micrographs/a.mrc
micrographs/b.mrc
"""


class TestProjectWatcherMicrographs(unittest.TestCase):
    def test_counts_rows_from_motioncorr_micrographs_star(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            star = root / "MotionCorr/job002/corrected_micrographs.star"
            star.parent.mkdir(parents=True)
            star.write_text(_MICROGRAPH_STAR, encoding="utf-8")
            watcher = ProjectWatcher(root)
            state = watcher.scan()
            self.assertEqual(state.n_micrographs, 2)

    def test_postprocess_masked_map_preferred(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pp = root / "PostProcess/job001"
            pp.mkdir(parents=True)
            (pp / "postprocess_masked.mrc").write_text("m", encoding="utf-8")
            (pp / "refined.mrc").write_text("m", encoding="utf-8")
            watcher = ProjectWatcher(root)
            state = watcher.scan()
            self.assertIsNotNone(state.map_path)
            assert state.map_path is not None
            self.assertIn("postprocess_masked.mrc", state.map_path)


if __name__ == "__main__":
    unittest.main()
