"""Scan RELION project directory and derive a conservative SampleState."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from relion.star_parser import (
    count_micrograph_rows_data_micrographs,
    count_unique_micrographs_from_star_text,
)
from schemas.sample_state import SampleState

logger = logging.getLogger(__name__)

# Reject absurd micrograph counts from mis-parsed particle stacks or corrupted scans.
_MAX_SANE_MICROGRAPHS = 1_000_000

STAGE_PRIORITY_HIGH_TO_LOW: List[str] = [
    "PostProcess",
    "Polish",
    "CtfRefine",
    "Refine3D",
    "Class3D",
    "InitialModel",
    "Class2D",
    "Extract",
    "AutoPick",
    "CtfFind",
    "MotionCorr",
    "Import",
]

ORIENTATION_KEYWORDS = ["preferred orientation", "orientation bias", "anisotropy"]
HETEROGENEITY_KEYWORDS = ["heterogeneity", "multiple conformations", "compositional heterogeneity"]


class ProjectWatcher:
    """Watch and scan RELION project structure."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root

    def scan(self) -> SampleState:
        """Return a best-effort SampleState from filesystem scan."""
        project_id = self.project_root.name
        state = SampleState(
            project_id=project_id,
            project_path=str(self.project_root),
            stage="unknown",
            status="running",
        )
        try:
            if not self.project_root.exists():
                state.status = "error"
                state.notes.append(f"Project path not found: {self.project_root}")
                return state

            state.stage = self.detect_stage()
            state.n_micrographs, micrographs_note = self._count_micrographs()

            if micrographs_note:
                state.notes.append(micrographs_note)
            state.n_particles_picked = self._estimate_particles_from_stage("Extract")
            state.n_particles_after_2d = self._estimate_particles_from_stage("Class2D")
            state.n_particles_after_3d = self._estimate_particles_from_stage("Class3D")

            map_path = self._find_best_map_file()
            if map_path:
                state.map_path = str(map_path)

            orientation_warn, heterogeneity_warn = self._scan_warning_keywords()
            state.orientation_warning = orientation_warn
            state.heterogeneity_warning = heterogeneity_warn
            state.notes.append("Project scan completed with conservative heuristics.")
            return state
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected failure while scanning project: %s", exc)
            state.status = "error"
            state.notes.append(f"Scan failed with exception: {exc}")
            return state

    def detect_stage(self) -> str:
        """Detect highest valid completed stage by priority."""
        for stage in STAGE_PRIORITY_HIGH_TO_LOW:
            if self.is_stage_completed(stage):
                return stage
        return "unknown"

    def is_stage_completed(self, stage: str) -> bool:
        """
        Check whether a stage is completed using conservative file evidence.

        Rules:
        - Stage directory must exist.
        - Prefer tangible outputs (star/map/log files) over directory-only checks.
        - For late stages, require stronger evidence to avoid false positives.
        """
        stage_dir = self.project_root / stage
        if not stage_dir.exists() or not stage_dir.is_dir():
            return False

        star_count = self._count_files(stage_dir, "*.star")
        log_count = self._count_files(stage_dir, "*.log") + self._count_files(stage_dir, "*.txt")
        map_count = self._count_files(stage_dir, "*.mrc") + self._count_files(stage_dir, "*.map")

        if stage == "Import":
            return star_count > 0 or log_count > 0 or self._count_files(stage_dir, "*") > 0
        if stage == "MotionCorr":
            return self._count_files(stage_dir, "*micrograph*.mrc") > 0 or star_count > 0
        if stage == "CtfFind":
            return star_count > 0 or log_count > 0
        if stage == "AutoPick":
            return star_count > 0 or log_count > 0
        if stage == "Extract":
            return star_count > 0
        if stage == "Class2D":
            return star_count > 0 or log_count > 0
        if stage == "InitialModel":
            return map_count > 0 or star_count > 0 or self._logs_show_completion(stage_dir)
        if stage == "Class3D":
            return map_count > 0 or star_count > 0 or self._logs_show_completion(stage_dir)
        if stage == "Refine3D":
            return map_count > 0 or self._logs_show_completion(stage_dir)
        if stage == "CtfRefine":
            return star_count > 0 or self._logs_show_completion(stage_dir)
        if stage == "Polish":
            return star_count > 0 or self._logs_show_completion(stage_dir)
        if stage == "PostProcess":
            return map_count > 0 or self._logs_show_completion(stage_dir)
        return False

    def _count_files(self, target_dir: Path, pattern: str) -> int:
        try:
            return sum(1 for _ in target_dir.rglob(pattern))
        except OSError:
            return 0

    def _logs_show_completion(self, stage_dir: Path) -> bool:
        for log_file in list(stage_dir.rglob("*.log")) + list(stage_dir.rglob("*.txt")):
            try:
                txt = log_file.read_text(encoding="utf-8", errors="ignore").lower()
                if "job finished" in txt or "finished successfully" in txt or "completed" in txt:
                    return True
            except OSError:
                continue
        return False

    def _count_micrographs(self) -> tuple[int, Optional[str]]:
        """
        Count micrographs from motion-correction / CTF lists — never from whole-project ``.mrc`` string hits.

        Returns ``(count, optional_note)`` when the estimate is unreliable or conservative.
        """
        note: Optional[str] = None

        # 1) MotionCorr: corrected_micrographs.star or *micrographs*.star (authoritative list).
        for rel in ("MotionCorr",):
            base = self.project_root / rel
            if not base.exists():
                continue
            for name in ("corrected_micrographs.star",):
                for star in base.rglob(name):
                    try:
                        txt = star.read_text(encoding="utf-8", errors="ignore")
                    except OSError:
                        continue
                    n_rows = count_micrograph_rows_data_micrographs(txt)
                    if n_rows is not None and n_rows > 0:
                        return min(n_rows, _MAX_SANE_MICROGRAPHS), note
                    n_unique = count_unique_micrographs_from_star_text(txt)
                    if n_unique is not None and n_unique > 0:
                        return min(n_unique, _MAX_SANE_MICROGRAPHS), note
            for star in base.rglob("*micrographs*.star"):
                if "Movies" in str(star):
                    continue
                try:
                    txt = star.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                n_unique = count_unique_micrographs_from_star_text(txt)
                if n_unique is not None and n_unique > 0:
                    return min(n_unique, _MAX_SANE_MICROGRAPHS), note
                n_rows = count_micrograph_rows_data_micrographs(txt)
                if n_rows is not None and n_rows > 0:
                    return min(n_rows, _MAX_SANE_MICROGRAPHS), note

        # 2) CtfFind: micrographs star
        ctf_base = self.project_root / "CtfFind"
        if ctf_base.exists():
            for star in sorted(ctf_base.rglob("*.star"), key=lambda p: p.stat().st_mtime, reverse=True):
                try:
                    txt = star.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                n_unique = count_unique_micrographs_from_star_text(txt)
                if n_unique is not None and n_unique > 0:
                    return min(n_unique, _MAX_SANE_MICROGRAPHS), note
                n_rows = count_micrograph_rows_data_micrographs(txt)
                if n_rows is not None and n_rows > 0:
                    return min(n_rows, _MAX_SANE_MICROGRAPHS), note

        # 3) Count physical micrograph movies under MotionCorr (filename heuristic).
        mc = self.project_root / "MotionCorr"
        if mc.exists():
            movie_mrc = [p for p in mc.rglob("*micrograph*.mrc") if "Movies" not in str(p)]
            if movie_mrc:
                return len(movie_mrc), note

        # 4) Fallback: unique micrograph names from any particle STAR (expensive but bounded).
        try:
            for star in sorted(self.project_root.rglob("*.star"), key=lambda p: p.stat().st_size if p.exists() else 0):
                if "PostProcess" in str(star) or "Refine3D" in str(star) or "Class3D" in str(star):
                    continue
                try:
                    txt = star.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if "_rlnMicrographName" not in txt:
                    continue
                n_unique = count_unique_micrographs_from_star_text(txt)
                if n_unique is not None and 0 < n_unique < _MAX_SANE_MICROGRAPHS:
                    return n_unique, note
        except OSError:
            pass

        note = "n_micrographs unavailable_or_unreliable"
        return 0, note

    def _estimate_particles_from_stage(self, stage_name: str) -> int:
        stage_dir = self.project_root / stage_name
        if not stage_dir.exists():
            return 0
        star_files = sorted(stage_dir.rglob("*.star"))
        if not star_files:
            return 0
        target = max(star_files, key=lambda p: p.stat().st_size if p.exists() else 0)
        try:
            text = target.read_text(encoding="utf-8", errors="ignore")
            rows = 0
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("_"):
                    continue
                if line.startswith("data_") or line.startswith("loop_"):
                    continue
                rows += 1
            return rows
        except OSError:
            return 0

    def _find_best_map_file(self) -> Optional[Path]:
        """Prefer PostProcess masked map, then other postprocessed mrc, then refine/class maps."""
        pp = self.project_root / "PostProcess"
        if pp.exists():
            masked = list(pp.rglob("postprocess_masked.mrc"))
            if masked:
                return max(masked, key=lambda p: p.stat().st_mtime if p.exists() else 0.0)
            post_any = list(pp.rglob("postprocess.mrc"))
            if post_any:
                return max(post_any, key=lambda p: p.stat().st_mtime if p.exists() else 0.0)

        preferred_dirs = ["PostProcess", "Refine3D", "Class3D", "InitialModel"]
        map_candidates: List[Path] = []
        for name in preferred_dirs:
            d = self.project_root / name
            if d.exists():
                map_candidates.extend(d.rglob("*.mrc"))
                map_candidates.extend(d.rglob("*.map"))
        if not map_candidates:
            return None
        return max(map_candidates, key=lambda p: p.stat().st_mtime if p.exists() else 0.0)

    def _scan_warning_keywords(self) -> tuple[bool, bool]:
        orientation_warn = False
        heterogeneity_warn = False
        log_files = list(self.project_root.rglob("*.log")) + list(self.project_root.rglob("*.txt"))
        for log in log_files:
            try:
                content = log.read_text(encoding="utf-8", errors="ignore").lower()
                if any(k in content for k in ORIENTATION_KEYWORDS):
                    orientation_warn = True
                if any(k in content for k in HETEROGENEITY_KEYWORDS):
                    heterogeneity_warn = True
            except OSError:
                continue
        return orientation_warn, heterogeneity_warn
