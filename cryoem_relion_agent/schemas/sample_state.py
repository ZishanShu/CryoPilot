"""Unified sample state schema for cryo-EM RELION agent."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class SampleState(BaseModel):
    """State snapshot for one RELION project/sample."""

    project_id: str = Field(..., description="Unique project identifier.")
    project_path: str = Field(..., description="Absolute or relative project path.")
    stage: str = Field(default="unknown", description="Current pipeline stage.")
    status: str = Field(default="running", description="Current pipeline status.")
    last_action: Optional[str] = Field(default=None, description="Last action name.")

    n_micrographs: int = Field(default=0, ge=0)
    n_particles_picked: int = Field(default=0, ge=0)
    n_particles_after_2d: int = Field(default=0, ge=0)
    n_particles_after_3d: int = Field(default=0, ge=0)

    estimated_resolution: Optional[float] = Field(
        default=None, description="Estimated global resolution in Angstrom."
    )
    orientation_warning: bool = Field(default=False)
    heterogeneity_warning: bool = Field(default=False)

    map_path: Optional[str] = Field(default=None, description="Best current map path.")
    sequence_path: Optional[str] = Field(default=None, description="Sequence file path.")
    candidate_maps: List[Any] = Field(
        default_factory=list,
        description="Candidate maps emitted by upstream reconstruction for downstream routing.",
    )
    last_execution: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Last run-next execution payload (e.g. ModelAngelo result dict).",
    )
    notes: List[str] = Field(default_factory=list)

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of state update.",
    )

    @field_validator("project_id", "project_path", "stage", "status")
    @classmethod
    def _must_not_be_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must not be empty")
        return value.strip()

    @field_validator("estimated_resolution", mode="before")
    @classmethod
    def _normalize_estimated_resolution(cls, value: object) -> Optional[float]:
        """Align with QC resolution band; invalid values become None (same rule as ``QCMetrics``)."""
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(parsed) or parsed <= 0.0:
            return None
        if not (0.5 < parsed < 50.0):
            return None
        return parsed

    def touch(self) -> None:
        """Update timestamp to current UTC time."""
        self.updated_at = datetime.now(timezone.utc)
