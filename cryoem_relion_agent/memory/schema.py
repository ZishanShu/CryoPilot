"""Pydantic schemas for sqlite case memory records."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class CaseRecord(BaseModel):
    """One historical cryo-EM case record for retrieval and reporting."""

    case_id: str = Field(..., description="Unique case ID.")
    project_id: str = Field(..., description="RELION project ID.")
    sample_type: str = Field(default="unknown")
    n_micrographs: int = Field(default=0, ge=0)
    n_particles_after_2d: int = Field(default=0, ge=0)
    estimated_resolution: Optional[float] = Field(default=None, gt=0.0)
    orientation_warning: bool = False
    heterogeneity_warning: bool = False
    chosen_model_builder: Optional[str] = None
    final_status: str = Field(default="unknown")
    notes: str = Field(default="")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
