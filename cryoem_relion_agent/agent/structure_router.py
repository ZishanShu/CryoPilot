"""Structure modeling tool router based on map quality and context."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field

from schemas.action import Action, ActionName
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState

logger = logging.getLogger(__name__)

DEFAULT_ROUTER_RULES: Dict[str, Any] = {
    "thresholds": {
        "mica_resolution_max": 3.5,
    },
    "priorities": {
        "run_mica": 90,
        "run_emodelx": 80,
        "run_modelangelo": 70,
        "request_human_review": 95,
    },
}


class ModelingContext(BaseModel):
    """Input context for structure model builder routing."""

    known_sequence: bool = Field(default=False)
    chain_count: Optional[int] = Field(default=None, ge=1)
    af3_available: bool = Field(default=False)
    complex_type: Optional[str] = Field(default=None)
    preferred_model_building_backend: Optional[str] = Field(
        default=None,
        description="Optional override; resolved via BackendRegistry with allowed_backends.",
    )
    af3_dir: Optional[str] = Field(
        default=None,
        description="AF3 output or inputs directory for MICA preflight (optional until MICA is chosen).",
    )


class StructureRouter:
    """Route to ModelAngelo/EModelX/MICA or human review."""

    def __init__(self, rules_path: Path | None = None) -> None:
        self.rules = self._load_rules(rules_path)

    def _load_rules(self, rules_path: Path | None) -> Dict[str, Any]:
        if not rules_path:
            logger.warning("No router rules file provided, using defaults.")
            return DEFAULT_ROUTER_RULES

        try:
            if not rules_path.exists():
                logger.warning("Router rules not found: %s, using defaults.", rules_path)
                return DEFAULT_ROUTER_RULES
            loaded = yaml.safe_load(rules_path.read_text(encoding="utf-8")) or {}
            merged = DEFAULT_ROUTER_RULES.copy()
            for key in ("thresholds", "priorities"):
                merged[key] = {**DEFAULT_ROUTER_RULES.get(key, {}), **loaded.get(key, {})}
            logger.info("Router rules loaded from %s", rules_path)
            return merged
        except (OSError, yaml.YAMLError) as exc:
            logger.error("Failed to load router rules, fallback defaults: %s", exc)
            return DEFAULT_ROUTER_RULES

    def route(
        self,
        state: SampleState,
        qc: QCMetrics,
        context: ModelingContext,
    ) -> Action:
        """Select structure modeling action from context and QC."""
        priorities = self.rules["priorities"]
        mica_resolution_max = self.rules["thresholds"]["mica_resolution_max"]
        resolution = qc.current_resolution or state.estimated_resolution

        if not context.known_sequence:
            return Action(
                name=ActionName.REQUEST_HUMAN_REVIEW,
                reason="Sequence unknown; manual expert modeling strategy required.",
                parameters={"project_id": state.project_id},
                priority=priorities["request_human_review"],
            )

        if (
            context.known_sequence
            and context.af3_available
            and resolution is not None
            and resolution <= mica_resolution_max
        ):
            mica_params: Dict[str, Any] = {"map_path": state.map_path, "sequence_path": state.sequence_path}
            if context.af3_dir and str(context.af3_dir).strip():
                mica_params["af3_dir"] = str(context.af3_dir).strip()
            return Action(
                name=ActionName.RUN_MICA,
                reason=f"Known sequence + AF3 available + high-resolution map ({resolution:.2f}A).",
                parameters=mica_params,
                priority=priorities["run_mica"],
            )

        if context.chain_count is not None and context.chain_count > 1:
            return Action(
                name=ActionName.RUN_EMODELX,
                reason="Known sequence and multi-chain complex; EModelX preferred.",
                parameters={
                    "chain_count": context.chain_count,
                    "map_path": state.map_path,
                    "sequence_path": state.sequence_path,
                },
                priority=priorities["run_emodelx"],
            )

        return Action(
            name=ActionName.RUN_MODELANGELO,
            reason="Known sequence in general case; ModelAngelo default route.",
            parameters={"map_path": state.map_path, "sequence_path": state.sequence_path},
            priority=priorities["run_modelangelo"],
        )

    def can_route_structure(
        self,
        state: SampleState,
        qc: QCMetrics,
        context: ModelingContext,
    ) -> tuple[bool, str]:
        """
        Decide whether structure routing should be triggered.

        This is the trigger-layer check. Tool selection stays in `route()`.
        """
        if not state.map_path:
            return False, "missing_map_path"

        has_sequence = context.known_sequence or bool(state.sequence_path)
        if not has_sequence:
            return False, "missing_sequence"

        map_quality_ready = qc.refine3d_completed or qc.current_resolution is not None
        if not map_quality_ready:
            return False, "insufficient_qc"

        return True, "map_and_sequence_ready"
