"""Unified QC metrics schema for cryo-EM RELION agent."""

from __future__ import annotations

import math
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Keep in sync with ``relion.star_parser`` plausible-resolution band (Angstrom).
_PLAUSIBLE_RESOLUTION_MIN_ANGSTROM = 0.5
_PLAUSIBLE_RESOLUTION_MAX_ANGSTROM = 50.0


class QCMetrics(BaseModel):
    """Quality control metrics derived from RELION jobs and outputs."""

    motioncorr_completed: bool = False
    ctf_completed: bool = False
    picking_completed: bool = False
    class2d_completed: bool = False
    init3d_completed: bool = False
    refine3d_completed: bool = False
    ctf_refine_completed: bool = False
    polish_completed: bool = False
    postprocess_completed: bool = False

    retain_ratio_2d: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    retain_ratio_3d: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    current_resolution: Optional[float] = Field(default=None, gt=0.0)

    particle_count_warning: bool = False
    orientation_warning: bool = False
    heterogeneity_warning: bool = False
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("current_resolution", mode="before")
    @classmethod
    def _normalize_current_resolution(cls, value: object) -> Optional[float]:
        """
        Normalize invalid resolution-like inputs to None.

        Invalid values include: None, empty string, nan, non-numeric, <= 0.
        """
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
        # Match cryo-EM single-particle global resolution sanity band (Angstrom).
        if not (_PLAUSIBLE_RESOLUTION_MIN_ANGSTROM < parsed < _PLAUSIBLE_RESOLUTION_MAX_ANGSTROM):
            return None
        return parsed

    @model_validator(mode="after")
    def _auto_confidence(self) -> "QCMetrics":
        """Auto-fill confidence when not provided, using simple conservative heuristic."""
        if self.confidence_score is not None:
            return self

        completed_flags = [
            self.motioncorr_completed,
            self.ctf_completed,
            self.picking_completed,
            self.class2d_completed,
            self.init3d_completed,
            self.refine3d_completed,
            self.ctf_refine_completed,
            self.polish_completed,
            self.postprocess_completed,
        ]
        base = sum(1 for x in completed_flags if x) / len(completed_flags)

        penalty = 0.0
        if self.particle_count_warning:
            penalty += 0.15
        if self.orientation_warning:
            penalty += 0.2
        if self.heterogeneity_warning:
            penalty += 0.2

        score = max(0.0, min(1.0, base - penalty))
        self.confidence_score = round(score, 3)
        return self
