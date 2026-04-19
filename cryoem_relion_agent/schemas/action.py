"""Action schema definitions for policy and execution."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict

from pydantic import BaseModel, Field, field_validator


class ActionName(str, Enum):
    """Supported action names in cryo-EM RELION agent."""

    RUN_AUTOPICK = "run_autopick"
    RUN_EXTRACT = "run_extract"
    RUN_CLASS2D = "run_class2d"
    RUN_IMPORT_PREPROCESS = "run_import_preprocess"
    RUN_PICKING_EXTRACT = "run_picking_extract"
    RUN_CLASS2D_CLEANUP = "run_class2d_cleanup"
    RUN_INIT3D_OR_CLASS3D = "run_init3d_or_class3d"
    RUN_INITMODEL_3DCLASS = "run_initmodel_3dclass"
    RUN_REFINE3D = "run_refine3d"
    RUN_CTF_REFINE = "run_ctf_refine"
    RUN_POLISH = "run_polish"
    RUN_REFINE_CTF_POLISH = "run_refine_ctf_polish"
    RUN_POSTPROCESS = "run_postprocess"
    RUN_MODELANGELO = "run_modelangelo"
    RUN_CRYODRGN = "run_cryodrgn"
    RUN_EMODELX = "run_emodelx"
    RUN_MICA = "run_mica"
    RUN_VALIDATION = "run_validation"
    REQUEST_HUMAN_REVIEW = "request_human_review"
    STOP_PIPELINE = "stop_pipeline"


class Action(BaseModel):
    """Standardized action object produced by policy/router and consumed by runners."""

    name: ActionName = Field(..., description="Action enum value.")
    reason: str = Field(..., min_length=1, description="Human-readable reason.")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action params.")
    priority: int = Field(default=50, ge=0, le=100, description="Higher means more urgent.")

    @field_validator("reason")
    @classmethod
    def _strip_reason(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("reason must not be empty")
        return value
