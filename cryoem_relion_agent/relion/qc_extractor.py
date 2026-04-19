"""Aggregate project scan + parsed STAR/LOG into unified QCMetrics."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

from relion.log_parser import LogParser
from relion.project_watcher import ProjectWatcher
from relion.star_parser import (
    PLAUSIBLE_RESOLUTION_MAX_ANGSTROM,
    StarParser,
    extract_resolution_from_star_text,
)
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState

logger = logging.getLogger(__name__)


class QCExtractor:
    """Extract QC metrics from RELION project with graceful degradation."""

    def __init__(self) -> None:
        self.star_parser = StarParser()
        self.log_parser = LogParser()

    def extract(self, project_root: Path) -> tuple[SampleState, QCMetrics]:
        watcher = ProjectWatcher(project_root)
        state = watcher.scan()
        stage_done = {
            "motioncorr_completed": watcher.is_stage_completed("MotionCorr"),
            "ctf_completed": watcher.is_stage_completed("CtfFind"),
            "picking_completed": watcher.is_stage_completed("AutoPick") and watcher.is_stage_completed("Extract"),
            "class2d_completed": watcher.is_stage_completed("Class2D"),
            "init3d_completed": watcher.is_stage_completed("InitialModel") or watcher.is_stage_completed("Class3D"),
            "refine3d_completed": watcher.is_stage_completed("Refine3D")
            or watcher.is_stage_completed("CtfRefine")
            or watcher.is_stage_completed("Polish")
            or watcher.is_stage_completed("PostProcess"),
            "ctf_refine_completed": watcher.is_stage_completed("CtfRefine")
            or watcher.is_stage_completed("Polish")
            or watcher.is_stage_completed("PostProcess"),
            "polish_completed": watcher.is_stage_completed("Polish")
            or watcher.is_stage_completed("PostProcess"),
            "postprocess_completed": watcher.is_stage_completed("PostProcess"),
        }
        parsed = self._parse_project_files(project_root)
        # Single prioritized resolution — never ``min()`` across all STAR files (FSC tables break this).
        current_resolution = self._normalize_resolution(self._collect_resolution_from_project(project_root, parsed))
        if state.map_path and current_resolution is not None and state.stage in {"unknown", "Import", "MotionCorr"}:
            # Align stage with stronger late-stage evidence from parsed map/resolution.
            state.stage = "Refine3D"

        particle_count_warning = state.n_particles_after_2d < 50000 and stage_done["class2d_completed"]
        retain_ratio_2d = self._safe_ratio(state.n_particles_after_2d, state.n_particles_picked)
        retain_ratio_3d = self._safe_ratio(state.n_particles_after_3d, state.n_particles_after_2d)
        orientation_warning = bool(state.orientation_warning or parsed["orientation_warning"])
        heterogeneity_warning = bool(state.heterogeneity_warning or parsed["heterogeneity_warning"])

        qc = QCMetrics(
            motioncorr_completed=stage_done["motioncorr_completed"],
            ctf_completed=stage_done["ctf_completed"],
            picking_completed=stage_done["picking_completed"],
            class2d_completed=stage_done["class2d_completed"],
            init3d_completed=stage_done["init3d_completed"],
            refine3d_completed=stage_done["refine3d_completed"],
            ctf_refine_completed=stage_done["ctf_refine_completed"],
            polish_completed=stage_done["polish_completed"],
            postprocess_completed=stage_done["postprocess_completed"],
            retain_ratio_2d=retain_ratio_2d,
            retain_ratio_3d=retain_ratio_3d,
            current_resolution=current_resolution,
            particle_count_warning=particle_count_warning,
            orientation_warning=orientation_warning,
            heterogeneity_warning=heterogeneity_warning,
            confidence_score=None,
        )
        if current_resolution is not None:
            state.estimated_resolution = current_resolution
        if parsed["map_path"] and not state.map_path:
            state.map_path = str(parsed["map_path"])
        return state, qc

    def _collect_resolution_from_project(
        self,
        project_root: Path,
        parsed: Dict[str, Optional[float | bool | Path | List[float]]],
    ) -> Optional[float]:
        """
        Pick one global resolution estimate using source priority — not ``min()`` over all files.

        Priority: PostProcess ``postprocess.star`` → Refine3D logs → other logs → trusted STAR fields.
        """
        # 1) PostProcess job outputs (most recent job first)
        post_candidates = sorted(
            [p for p in project_root.rglob("PostProcess/**/postprocess.star") if p.is_file()],
            key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
            reverse=True,
        )
        for path in post_candidates:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            res = extract_resolution_from_star_text(text)
            if res is not None:
                logger.debug("Resolution from PostProcess STAR %s: %s", path, res)
                return res

        # 2) Explicit candidate from parsed Refine3D / postprocess logs (already keyword-filtered in log_parser)
        cand = parsed.get("best_resolution_from_logs")
        if isinstance(cand, float):
            return cand

        # 3) Refine3D logs (newest first)
        for log in sorted(
            [p for p in project_root.rglob("Refine3D/**/*.log") if p.is_file()],
            key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
            reverse=True,
        ):
            try:
                parsed_log = self.log_parser.parse(log)
            except OSError:
                continue
            res = parsed_log.get("resolution")
            if isinstance(res, float):
                return res

        # 4) Any job log under project (keyword-filtered)
        for log in sorted(
            [p for p in project_root.rglob("**/*.log") if p.is_file()],
            key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
            reverse=True,
        ):
            try:
                parsed_log = self.log_parser.parse(log)
            except OSError:
                continue
            res = parsed_log.get("resolution")
            if isinstance(res, float):
                return res

        # 5) STAR files: structured extract only (skip FSC loops)
        for star in sorted(
            [p for p in project_root.rglob("**/*.star") if p.is_file()],
            key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
            reverse=True,
        ):
            try:
                text = star.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            res = extract_resolution_from_star_text(text)
            if res is not None:
                return res

        return None

    def _parse_project_files(self, project_root: Path) -> Dict[str, Optional[float | bool | Path | List[float]]]:
        """Collect warnings and map paths; log best resolution hint for merge (see ``_collect_resolution_from_project``)."""
        map_path: Optional[Path] = None
        orientation_warning = False
        heterogeneity_warning = False
        best_log_resolution: Optional[float] = None

        for path in list(project_root.rglob("*.star")):
            try:
                parsed = self.star_parser.parse(path)
            except OSError:
                continue
            if parsed.get("map_path") and map_path is None:
                candidate = Path(str(parsed["map_path"]))
                map_path = candidate if candidate.is_absolute() else (path.parent / candidate)
            orientation_warning = orientation_warning or bool(parsed.get("orientation_warning"))
            heterogeneity_warning = heterogeneity_warning or bool(parsed.get("heterogeneity_warning"))

        for path in list(project_root.rglob("*.log")) + list(project_root.rglob("*.txt")):
            try:
                parsed = self.log_parser.parse(path)
            except OSError:
                continue
            if parsed.get("map_path") and map_path is None:
                candidate = Path(str(parsed["map_path"]))
                map_path = candidate if candidate.is_absolute() else (path.parent / candidate)
            orientation_warning = orientation_warning or bool(parsed.get("orientation_warning"))
            heterogeneity_warning = heterogeneity_warning or bool(parsed.get("heterogeneity_warning"))
            res = parsed.get("resolution")
            if isinstance(res, float):
                if best_log_resolution is None or res < best_log_resolution:
                    best_log_resolution = res

        return {
            "map_path": map_path,
            "orientation_warning": orientation_warning,
            "heterogeneity_warning": heterogeneity_warning,
            "best_resolution_from_logs": best_log_resolution,
        }

    def _pick_best_resolution(self, parsed: Dict[str, Optional[float | bool | Path | List[float]]]) -> Optional[float]:
        """Deprecated path: kept for tests that monkeypatch; prefer ``_collect_resolution_from_project``."""
        return parsed.get("best_resolution_from_logs") if isinstance(parsed.get("best_resolution_from_logs"), float) else None

    def _safe_ratio(self, numerator: int, denominator: int) -> Optional[float]:
        if denominator <= 0:
            return None
        try:
            value = numerator / denominator
            return max(0.0, min(1.0, round(value, 4)))
        except ZeroDivisionError:
            return None

    def _normalize_resolution(self, value: object) -> Optional[float]:
        """Normalize invalid resolution values to None."""
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"", "none", "nan"}:
                return None
            try:
                parsed = float(text)
            except ValueError:
                return None
        else:
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                return None
        if math.isnan(parsed) or parsed <= 0.0:
            return None
        if parsed > PLAUSIBLE_RESOLUTION_MAX_ANGSTROM:
            return None
        return parsed
