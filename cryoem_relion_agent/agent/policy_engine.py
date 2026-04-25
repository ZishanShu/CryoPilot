"""Rule-based policy engine for deciding next RELION pipeline action."""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import yaml

from agent.backend_registry import BackendRegistry, get_backend_capability_matrix
from agent.history_manager import ActionHistoryManager
from schemas.action import Action, ActionName
from schemas.action_history import ActionHistoryEntry
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState

logger = logging.getLogger(__name__)

DEFAULT_POLICY_RULES: Dict[str, Any] = {
    "thresholds": {
        "min_particles_for_3d": 50000,
        "min_retain_ratio_2d": 0.2,
        "target_resolution_for_modeling": 4.0,
    },
    "priorities": {
        "request_human_review": 95,
        "run_cryodrgn": 94,
        "run_autopick": 78,
        "run_extract": 74,
        "run_class2d": 71,
        "run_import_preprocess": 80,
        "run_picking_extract": 75,
        "run_class2d_cleanup": 70,
        "run_init3d_or_class3d": 66,
        "run_initmodel_3dclass": 65,
        "run_refine3d": 62,
        "run_ctf_refine": 61,
        "run_polish": 60,
        "run_postprocess": 59,
        "run_refine_ctf_polish": 60,
        "run_modelangelo": 58,
        "run_validation": 55,
        "stop_pipeline": 10,
    },
    "picking": {
        "default_backend": "relion_log",
        "allowed_backends": ["relion_log", "topaz", "cryolo"],
        "project_overrides": {},
    },
    "heterogeneity": {
        "default_backend": "human_review",
        "allowed_backends": ["human_review", "cryodrgn"],
        "project_overrides": {},
    },
    "model_building": {
        "default_backend": "modelangelo",
        "allowed_backends": ["modelangelo", "emodelx", "mica"],
        "project_overrides": {},
    },
}


class PolicyEngine:
    """Evaluate SampleState + QCMetrics to produce next Action."""

    def __init__(self, rules_path: Path | None = None) -> None:
        self.rules = self._load_rules(rules_path)
        self._backend_registry_yaml_path = Path(__file__).resolve().parents[1] / "configs" / "backend_registry.yaml"
        self.backend_registry = BackendRegistry.from_policy_and_file(
            self.rules,
            self._backend_registry_yaml_path if self._backend_registry_yaml_path.exists() else None,
        )
        self.history_manager: Optional[ActionHistoryManager] = None
        self._trace_run_id: Optional[str] = None
        self._trace_previous_entry_id: Optional[str] = None
        self._trace_branch_name: str = "main"
        self._last_decision_entry_id: Optional[str] = None

    def set_history_manager(self, history_manager: Optional[ActionHistoryManager]) -> None:
        """Attach optional history manager for decision tracing."""
        self.history_manager = history_manager

    def set_trace_context(
        self,
        *,
        run_id: Optional[str] = None,
        previous_entry_id: Optional[str] = None,
        branch_name: str = "main",
    ) -> None:
        """Set trace context used for history linking during one orchestration run."""
        self._trace_run_id = run_id
        self._trace_previous_entry_id = previous_entry_id
        self._trace_branch_name = branch_name
        self._last_decision_entry_id = None

    def get_last_decision_entry_id(self) -> Optional[str]:
        """Return last decision entry id created by this policy cycle."""
        return self._last_decision_entry_id

    def _log_decision(
        self,
        state: SampleState,
        qc: QCMetrics,
        action_name: str,
        reason: str,
        branch_name: Optional[str] = None,
    ) -> None:
        if self.history_manager is None:
            return
        entry = ActionHistoryEntry(
            run_id=self._trace_run_id or uuid4().hex,
            parent_decision_id=None,
            previous_entry_id=self._trace_previous_entry_id,
            branch_name=branch_name or self._trace_branch_name,
            stage=state.stage,
            action_name=action_name,
            decision_reason=reason,
            qc_snapshot=qc.model_dump(mode="json"),
            execution_status="decided",
            output_summary=f"policy decided {action_name}",
        )
        entry_id = self.history_manager.add_action_to_history(entry)
        if entry_id:
            self._trace_previous_entry_id = entry_id
            self._last_decision_entry_id = entry_id

    @staticmethod
    def _infer_branch_from_action(action: Action) -> str:
        branch = str(action.parameters.get("branch_name", "")).strip()
        if branch:
            return branch
        if action.name == ActionName.REQUEST_HUMAN_REVIEW:
            return "heterogeneity"
        if action.name == ActionName.RUN_CRYODRGN:
            return "heterogeneity"
        return "main"

    def _return_action(
        self,
        state: SampleState,
        qc: QCMetrics,
        action: Action,
        *,
        log_history: bool = True,
    ) -> Action:
        if log_history:
            self._log_decision(
                state,
                qc,
                action.name.value,
                action.reason,
                branch_name=self._infer_branch_from_action(action),
            )
        return action

    def _load_rules(self, rules_path: Path | None) -> Dict[str, Any]:
        if not rules_path:
            logger.warning("No policy rules file provided, using defaults.")
            return deepcopy(DEFAULT_POLICY_RULES)

        try:
            if not rules_path.exists():
                logger.warning("Policy rules not found: %s, using defaults.", rules_path)
                return deepcopy(DEFAULT_POLICY_RULES)

            loaded = yaml.safe_load(rules_path.read_text(encoding="utf-8")) or {}
            merged = deepcopy(DEFAULT_POLICY_RULES)
            for key in ("thresholds", "priorities", "picking", "heterogeneity", "model_building"):
                base = DEFAULT_POLICY_RULES.get(key, {})
                extra = loaded.get(key, {})
                if key == "picking":
                    legacy = loaded.get("picker", {})
                    merged[key] = {**base, **legacy, **extra}
                else:
                    merged[key] = {**base, **extra}
            logger.info("Policy rules loaded from %s", rules_path)
            return merged
        except (OSError, yaml.YAMLError) as exc:
            logger.error("Failed to load policy rules, fallback to defaults: %s", exc)
            return deepcopy(DEFAULT_POLICY_RULES)

    @staticmethod
    def _stage_rank(stage: str) -> int:
        order = [
            "unknown",
            "Import",
            "MotionCorr",
            "CtfFind",
            "AutoPick",
            "Extract",
            "Class2D",
            "InitialModel",
            "Class3D",
            "Refine3D",
            "CtfRefine",
            "Polish",
            "PostProcess",
        ]
        try:
            return order.index(stage)
        except ValueError:
            return 0

    def _is_stage_at_least(self, state: SampleState, stage: str) -> bool:
        return self._stage_rank(state.stage) >= self._stage_rank(stage)

    def _refresh_backend_registry(self) -> None:
        """Rebuild registry from current ``self.rules`` so runtime rule changes apply."""
        self.backend_registry = BackendRegistry.from_policy_and_file(
            self.rules,
            self._backend_registry_yaml_path if self._backend_registry_yaml_path.exists() else None,
        )

    def _select_picking_backend(self, state: SampleState, preferred_backend: Optional[str] = None) -> str:
        b, _ = self.backend_registry.resolve("picking", preferred_backend, state.project_id)
        return b

    def _select_heterogeneity_backend(self, state: SampleState, preferred_backend: Optional[str] = None) -> str:
        b, _ = self.backend_registry.resolve("heterogeneity", preferred_backend, state.project_id)
        return b

    def _select_model_building_backend(self, state: SampleState, preferred_backend: Optional[str] = None) -> str:
        b, _ = self.backend_registry.resolve("model_building", preferred_backend, state.project_id)
        return b

    def _build_heterogeneity_action(
        self,
        state: SampleState,
        *,
        reason: str,
        priorities: Dict[str, int],
        preferred_backend: Optional[str] = None,
    ) -> Action:
        backend = self._select_heterogeneity_backend(state, preferred_backend=preferred_backend)
        if backend == "cryodrgn":
            return Action(
                name=ActionName.RUN_CRYODRGN,
                reason=f"{reason} (backend=cryodrgn).",
                parameters={"stage": state.stage, "branch_name": "heterogeneity", "heterogeneity_backend": "cryodrgn"},
                priority=priorities.get("run_cryodrgn", 94),
            )
        return Action(
            name=ActionName.REQUEST_HUMAN_REVIEW,
            reason=f"{reason} (backend=human_review).",
            parameters={"stage": state.stage, "branch_name": "heterogeneity", "heterogeneity_backend": "human_review"},
            priority=priorities["request_human_review"],
        )

    def decide_picking_stage(
        self,
        state: SampleState,
        qc: QCMetrics,
        *,
        preferred_backend: Optional[str] = None,
    ) -> Optional[Action]:
        priorities = self.rules["priorities"]
        if qc.motioncorr_completed and qc.ctf_completed and not self._is_stage_at_least(state, "AutoPick"):
            backend = self._select_picking_backend(state, preferred_backend=preferred_backend)
            return Action(
                name=ActionName.RUN_AUTOPICK,
                reason=f"CTF is ready but AutoPick stage is pending (backend={backend}).",
                parameters={
                    "project_path": state.project_path,
                    "branch_name": "main",
                    "picking_backend": backend,
                },
                priority=priorities.get("run_autopick", 78),
            )
        return None

    @staticmethod
    def _extract_latest_picking_qc(state: SampleState) -> Optional[Dict[str, Any]]:
        """Return picking_qc dict from last_execution, or None if absent."""
        le = getattr(state, "last_execution", None)
        if le is None:
            return None
        if not isinstance(le, dict):
            try:
                le = vars(le)
            except TypeError:
                return None
        qc = le.get("picking_qc")
        if isinstance(qc, dict):
            return qc
        md = le.get("metadata")
        if isinstance(md, dict):
            qc = md.get("picking_qc")
            if isinstance(qc, dict):
                return qc
        return None

    @staticmethod
    def _last_execution_dict(state: SampleState) -> Optional[Dict[str, Any]]:
        le = getattr(state, "last_execution", None)
        if isinstance(le, dict):
            return le
        if le is None:
            return None
        try:
            return vars(le)
        except TypeError:
            return None

    @staticmethod
    def _last_execution_picking_backend(le: Dict[str, Any]) -> str:
        for key in ("picking_backend",):
            v = le.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip().lower()
        md = le.get("metadata")
        if isinstance(md, dict):
            v = md.get("picking_backend")
            if isinstance(v, str) and v.strip():
                return v.strip().lower()
        params = le.get("parameters")
        if isinstance(params, dict):
            v = params.get("picking_backend")
            if isinstance(v, str) and v.strip():
                return v.strip().lower()
        return ""

    @staticmethod
    def _gate_params_from_le(le: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build gate action parameters from a persisted last_execution dict.

        Works for both the original Topaz execution result (via picking_qc) and
        the persisted gate execution result (via direct fields + metadata).
        Returns None if the evidence for failed_qc is absent.
        """
        md: Dict[str, Any] = le.get("metadata") if isinstance(le.get("metadata"), dict) else {}

        def _f(key: str):
            v = le.get(key)
            return v if v is not None else md.get(key)

        qc_status = str(_f("picking_qc_status") or "").strip()
        branch = str(_f("branch_name") or "").strip()
        if qc_status == "failed_qc" and branch == "picking_qc":
            # Persisted gate result — recover directly stored fields.
            flags = list(_f("picking_qc_flags") or [])
            rec = str(_f("recommended_next_action") or "manual_review")
            backend = str(_f("picking_backend") or "topaz")
            return {
                "branch_name": "picking_qc",
                "picking_backend": backend,
                "picking_qc_status": "failed_qc",
                "picking_qc_flags": flags,
                "recommended_next_action": rec,
                "n_picks": _f("n_picks"),
                "n_micrographs_total": _f("n_micrographs_total"),
                "zero_pick_micrograph_ratio": _f("zero_pick_micrograph_ratio"),
                "precision": _f("precision"),
                "recall": _f("recall"),
                "f1": _f("f1"),
            }

        # Original Topaz execution result — look for picking_qc dict.
        picking_qc: Optional[Dict[str, Any]] = le.get("picking_qc")
        if not isinstance(picking_qc, dict) and isinstance(md, dict):
            picking_qc = md.get("picking_qc")
        if not isinstance(picking_qc, dict):
            return None
        if picking_qc.get("qc_status") != "failed_qc":
            return None
        flags = list(picking_qc.get("qc_flags") or [])
        rec = str(picking_qc.get("recommended_next_action") or "manual_review")
        backend = str(_f("picking_backend") or "topaz")
        ge = picking_qc.get("ground_truth_evaluation") or {}
        return {
            "branch_name": "picking_qc",
            "picking_backend": backend,
            "picking_qc_status": "failed_qc",
            "picking_qc_flags": flags,
            "recommended_next_action": rec,
            "n_picks": picking_qc.get("n_picks"),
            "n_micrographs_total": picking_qc.get("n_micrographs_total"),
            "zero_pick_micrograph_ratio": picking_qc.get("zero_pick_micrograph_ratio"),
            "precision": ge.get("precision"),
            "recall": ge.get("recall"),
            "f1": ge.get("f1"),
        }

    def _build_picking_qc_gate_action(
        self,
        state: SampleState,
        priorities: Dict[str, int],
    ) -> Optional[Action]:
        """Return the gate action when Topaz picking_qc indicates failed_qc.

        The gate is idempotent: it fires from:
        - Case A: last_execution is run_autopick (Topaz) with picking_qc.qc_status == failed_qc
        - Case B: last_execution is already a picking_qc gate result (request_human_review
          with branch_name=picking_qc and picking_qc_status=failed_qc), which happens after
          a dry-run has overwritten the original Topaz last_execution.

        Returns ``None`` when evidence is absent or contradicts the gate condition.
        """
        le = self._last_execution_dict(state)
        if not le:
            return None
        action_name = str(le.get("action_name") or "").strip().lower()
        md: Dict[str, Any] = le.get("metadata") if isinstance(le.get("metadata"), dict) else {}

        # Case B — idempotent re-fire from a persisted gate execution result.
        if action_name == "request_human_review":
            branch = str(le.get("branch_name") or md.get("branch_name") or "").strip()
            qc_status = str(le.get("picking_qc_status") or md.get("picking_qc_status") or "").strip()
            if branch == "picking_qc" and qc_status == "failed_qc":
                gp = self._gate_params_from_le(le)
                if gp is not None:
                    flags = gp["picking_qc_flags"]
                    rec = gp["recommended_next_action"]
                    reason_flags = ", ".join(flags) if flags else "unknown"
                    return Action(
                        name=ActionName.REQUEST_HUMAN_REVIEW,
                        reason=(
                            f"Picking QC failed after Topaz: {reason_flags}. "
                            f"Recommended recovery: {rec}."
                        ),
                        parameters=gp,
                        priority=priorities.get("request_human_review", 95),
                    )
            # A different human_review action (e.g. heterogeneity) — do not trigger gate.
            return None

        # Case A — primary: last_execution is a Topaz run_autopick.
        if action_name and action_name != "run_autopick":
            return None
        backend = self._last_execution_picking_backend(le)
        if backend and backend != "topaz":
            return None
        gp = self._gate_params_from_le(le)
        if gp is None:
            return None
        flags = gp["picking_qc_flags"]
        rec = gp["recommended_next_action"]
        reason_flags = ", ".join(flags) if flags else "unknown"
        return Action(
            name=ActionName.REQUEST_HUMAN_REVIEW,
            reason=(
                f"Picking QC failed after Topaz: {reason_flags}. "
                f"Recommended recovery: {rec}."
            ),
            parameters=gp,
            priority=priorities.get("request_human_review", 95),
        )

    def decide_extract_stage(self, state: SampleState, qc: QCMetrics) -> Optional[Action]:
        priorities = self.rules["priorities"]
        if self._is_stage_at_least(state, "AutoPick") and not self._is_stage_at_least(state, "Extract"):
            return Action(
                name=ActionName.RUN_EXTRACT,
                reason="AutoPick appears done while extraction is pending.",
                parameters={"project_path": state.project_path, "branch_name": "main"},
                priority=priorities.get("run_extract", 74),
            )
        return None

    def decide_class2d_stage(self, state: SampleState, qc: QCMetrics) -> Optional[Action]:
        priorities = self.rules["priorities"]
        if (self._is_stage_at_least(state, "Extract") or state.n_particles_picked > 0) and not qc.class2d_completed:
            return Action(
                name=ActionName.RUN_CLASS2D,
                reason="Extraction is available and Class2D is pending.",
                parameters={"project_path": state.project_path, "branch_name": "main"},
                priority=priorities.get("run_class2d", 71),
            )
        return None

    def decide_3d_init_stage(self, state: SampleState, qc: QCMetrics) -> Optional[Action]:
        priorities = self.rules["priorities"]
        if qc.class2d_completed and not qc.init3d_completed and not self._is_stage_at_least(state, "InitialModel"):
            return Action(
                name=ActionName.RUN_INIT3D_OR_CLASS3D,
                reason="Class2D is done and Init3D/Class3D stage is pending.",
                parameters={"project_path": state.project_path, "branch_name": "main"},
                priority=priorities.get("run_init3d_or_class3d", priorities.get("run_initmodel_3dclass", 66)),
            )
        return None

    def decide_refine3d_stage(self, state: SampleState, qc: QCMetrics) -> Optional[Action]:
        priorities = self.rules["priorities"]
        if qc.init3d_completed and not qc.refine3d_completed and not self._is_stage_at_least(state, "Refine3D"):
            return Action(
                name=ActionName.RUN_REFINE3D,
                reason="Init3D/Class3D appears done while Refine3D is pending.",
                parameters={"project_path": state.project_path, "branch_name": "main"},
                priority=priorities.get("run_refine3d", priorities.get("run_refine_ctf_polish", 62)),
            )
        return None

    def decide_postprocess_stage(self, state: SampleState, qc: QCMetrics) -> Optional[Action]:
        priorities = self.rules["priorities"]
        if qc.polish_completed and not qc.postprocess_completed:
            return Action(
                name=ActionName.RUN_POSTPROCESS,
                reason="Polish appears done while PostProcess is pending.",
                parameters={"project_path": state.project_path, "branch_name": "main"},
                priority=priorities.get("run_postprocess", 59),
            )
        return None

    def decide_ctf_refine_stage(self, state: SampleState, qc: QCMetrics) -> Optional[Action]:
        priorities = self.rules["priorities"]
        if (
            qc.refine3d_completed
            and not qc.ctf_refine_completed
            and not qc.polish_completed
            and not qc.postprocess_completed
        ):
            return Action(
                name=ActionName.RUN_CTF_REFINE,
                reason="Refine3D appears done while CtfRefine is pending.",
                parameters={"project_path": state.project_path, "branch_name": "main"},
                priority=priorities.get("run_ctf_refine", 61),
            )
        return None

    def decide_polish_stage(self, state: SampleState, qc: QCMetrics) -> Optional[Action]:
        priorities = self.rules["priorities"]
        if qc.ctf_refine_completed and not qc.polish_completed and not qc.postprocess_completed:
            return Action(
                name=ActionName.RUN_POLISH,
                reason="CtfRefine appears done while Polish is pending.",
                parameters={"project_path": state.project_path, "branch_name": "main"},
                priority=priorities.get("run_polish", 60),
            )
        return None

    def decide_next_action(
        self,
        state: SampleState,
        qc: QCMetrics,
        *,
        log_history: bool = True,
        preferred_picking_backend: Optional[str] = None,
        preferred_heterogeneity_backend: Optional[str] = None,
    ) -> Optional[Action]:
        """Return next pipeline action, or ``None`` when structure routing should decide the tool."""
        self._refresh_backend_registry()
        self._last_decision_entry_id = None
        thresholds = self.rules["thresholds"]
        priorities = self.rules["priorities"]
        ret = lambda a: self._return_action(state, qc, a, log_history=log_history)

        if not (qc.motioncorr_completed and qc.ctf_completed):
            return ret(Action(
                name=ActionName.RUN_IMPORT_PREPROCESS,
                reason="Preprocess (Import/MotionCorr/CTF) not completed.",
                parameters={"project_path": state.project_path},
                priority=priorities["run_import_preprocess"],
            ))

        # Picking QC gate: runs before any picking/extract decision so that a
        # filesystem scan still showing CtfFind cannot push us back to run_autopick
        # after a Topaz run has already produced failed_qc.
        gate_action = self._build_picking_qc_gate_action(state, priorities)
        if gate_action is not None:
            return ret(gate_action)

        upstream_picking = self.decide_picking_stage(
            state,
            qc,
            preferred_backend=preferred_picking_backend,
        )
        if upstream_picking is not None:
            return ret(upstream_picking)

        upstream_extract = self.decide_extract_stage(state, qc)
        if upstream_extract is not None:
            return ret(upstream_extract)

        upstream_class2d = self.decide_class2d_stage(state, qc)
        if upstream_class2d is not None:
            return ret(upstream_class2d)

        # Keep quality guardrails ahead of 3D/post-refine progression.
        if qc.heterogeneity_warning:
            return ret(self._build_heterogeneity_action(
                state,
                reason="Detected heterogeneity warning, needs expert branch decision.",
                priorities=priorities,
                preferred_backend=preferred_heterogeneity_backend,
            ))

        if qc.retain_ratio_2d is not None and qc.retain_ratio_2d < thresholds["min_retain_ratio_2d"]:
            low_ratio_action = self._build_heterogeneity_action(
                state,
                reason=(
                    f"2D retain ratio too low ({qc.retain_ratio_2d:.3f} < "
                    f"{thresholds['min_retain_ratio_2d']:.3f})."
                ),
                priorities=priorities,
                preferred_backend=preferred_heterogeneity_backend,
            )
            merged = dict(low_ratio_action.parameters)
            merged["retain_ratio_2d"] = qc.retain_ratio_2d
            return ret(low_ratio_action.model_copy(update={"parameters": merged}))

        # On clearly postprocess-complete states, preserve existing structure routing precedence.
        postprocess_ready = qc.postprocess_completed or self._is_stage_at_least(state, "PostProcess")
        if postprocess_ready:
            if self.should_route_structure(state, qc):
                if log_history:
                    self._log_decision(
                        state,
                        qc,
                        "delegate_structure_router",
                        "policy delegates structure-tool routing",
                        branch_name="structure_router",
                    )
                return None

            if self.can_validate_only(state, qc):
                return ret(Action(
                    name=ActionName.RUN_VALIDATION,
                    reason="Map available but structure routing unavailable; fallback to validation.",
                    parameters={
                        "map_path": state.map_path,
                        "current_resolution": qc.current_resolution,
                        "routing_triggered": False,
                        "routing_reason": "missing_sequence",
                    },
                    priority=priorities["run_validation"],
                ))

        mid3d_init = self.decide_3d_init_stage(state, qc)
        if mid3d_init is not None:
            return ret(mid3d_init)

        mid3d_refine = self.decide_refine3d_stage(state, qc)
        if mid3d_refine is not None:
            return ret(mid3d_refine)

        postrefine_ctf = self.decide_ctf_refine_stage(state, qc)
        if postrefine_ctf is not None:
            return ret(postrefine_ctf)

        postrefine_polish = self.decide_polish_stage(state, qc)
        if postrefine_polish is not None:
            return ret(postrefine_polish)

        mid3d_post = self.decide_postprocess_stage(state, qc)
        if mid3d_post is not None:
            return ret(mid3d_post)

        if not qc.picking_completed:
            return ret(Action(
                name=ActionName.RUN_PICKING_EXTRACT,
                reason="Particle picking/extraction not completed.",
                parameters={"project_path": state.project_path},
                priority=priorities["run_picking_extract"],
            ))

        if not qc.class2d_completed:
            return ret(Action(
                name=ActionName.RUN_CLASS2D_CLEANUP,
                reason="2D classification not completed.",
                parameters={"project_path": state.project_path},
                priority=priorities["run_class2d_cleanup"],
            ))

        if not qc.init3d_completed and state.n_particles_after_2d >= thresholds["min_particles_for_3d"]:
            return ret(Action(
                name=ActionName.RUN_INITMODEL_3DCLASS,
                reason="2D done and particle count is sufficient for 3D initialization.",
                parameters={"n_particles_after_2d": state.n_particles_after_2d},
                priority=priorities["run_initmodel_3dclass"],
            ))

        if qc.init3d_completed and not qc.refine3d_completed:
            return ret(Action(
                name=ActionName.RUN_REFINE_CTF_POLISH,
                reason="Initial 3D available but refine/polish not completed.",
                parameters={"project_path": state.project_path},
                priority=priorities["run_refine_ctf_polish"],
            ))

        if self.should_route_structure(state, qc):
            # Delegate to ``StructureRouter`` in ``Scheduler`` (returns run_modelangelo / emodelx / mica).
            if log_history:
                self._log_decision(
                    state,
                    qc,
                    "delegate_structure_router",
                    "policy delegates structure-tool routing",
                    branch_name="structure_router",
                )
            return None

        if self.can_validate_only(state, qc):
            return ret(Action(
                name=ActionName.RUN_VALIDATION,
                reason="Map available but structure routing unavailable; fallback to validation.",
                parameters={
                    "map_path": state.map_path,
                    "current_resolution": qc.current_resolution,
                    "routing_triggered": False,
                    "routing_reason": "missing_sequence",
                },
                priority=priorities["run_validation"],
            ))

        return ret(Action(
            name=ActionName.STOP_PIPELINE,
            reason="No further automatic action matched; pipeline considered complete/stable.",
            parameters={},
            priority=priorities["stop_pipeline"],
        ))

    def is_map_ready(self, state: SampleState, qc: QCMetrics) -> bool:
        """Map is considered ready when map_path exists and late-stage evidence is present."""
        if not state.map_path:
            return False
        return qc.refine3d_completed or qc.current_resolution is not None

    def should_route_structure(self, state: SampleState, qc: QCMetrics) -> bool:
        """Trigger structure routing when map is ready and sequence is available."""
        if not self.is_map_ready(state, qc):
            return False
        return bool(state.sequence_path)

    def can_validate_only(self, state: SampleState, qc: QCMetrics) -> bool:
        """Validation-only fallback when map is ready but routing cannot start."""
        return self.is_map_ready(state, qc) and not bool(state.sequence_path)

    def select_backend_for_category(
        self,
        *,
        category: str,
        state: SampleState,
        context: Dict[str, Any],
        preferred_backend: Optional[str] = None,
        availability_snapshot: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Structured backend selection decision with explainable reasoning.

        Returns:
            {
              selected_backend, selection_reason, candidate_backends, rejected_backends,
              rejection_reasons, used_inputs, availability_snapshot, fallback_applied
            }
        """
        self._refresh_backend_registry()
        sec = self.backend_registry.section(category)
        allowed = [str(x).strip().lower() for x in sec.get("allowed_backends", []) if str(x).strip()]
        default_backend = str(sec.get("default_backend") or "").strip().lower()
        capability = get_backend_capability_matrix()
        avail = dict(availability_snapshot or {})
        used_inputs = {
            "map_path": bool(state.map_path),
            "sequence_path": bool(state.sequence_path),
            "af3_dir": bool(context.get("af3_dir")),
            "micrographs_star_path": bool(context.get("micrographs_star_path")),
            "particles_star_path": bool(context.get("particles_star_path")),
        }
        rejection_reasons: Dict[str, str] = {}
        rejected: list[str] = []

        def _is_available(bid: str) -> bool:
            p = avail.get(bid, {})
            if not isinstance(p, dict) or not p:
                return True
            return bool(p.get("available", True))

        # Preferred backend always evaluated first when valid and allowed.
        if preferred_backend is not None and str(preferred_backend).strip():
            pref = str(preferred_backend).strip().lower()
            if pref not in allowed:
                rejection_reasons[pref] = "not_in_allowlist"
                rejected.append(pref)
            elif not _is_available(pref):
                rejection_reasons[pref] = "unavailable"
                rejected.append(pref)
            else:
                return {
                    "category": category,
                    "selected_backend": pref,
                    "selection_reason": "preferred_backend_selected",
                    "candidate_backends": allowed,
                    "rejected_backends": rejected,
                    "rejection_reasons": rejection_reasons,
                    "used_inputs": used_inputs,
                    "availability_snapshot": avail,
                    "fallback_applied": False,
                }

        selected = default_backend if default_backend in allowed else (allowed[0] if allowed else "")
        reason = "default_backend"

        if category == "model_building":
            # Prefer MICA when AF3 prior available and backend available.
            if used_inputs["af3_dir"] and "mica" in allowed and _is_available("mica"):
                selected = "mica"
                reason = "af3_dir_present_and_mica_available"
            elif used_inputs["map_path"] and used_inputs["sequence_path"] and "emodelx" in allowed and _is_available("emodelx"):
                selected = "emodelx"
                reason = "map_sequence_present_and_emodelx_available"
            elif "modelangelo" in allowed and _is_available("modelangelo"):
                selected = "modelangelo"
                reason = "fallback_modelangelo"
        elif category == "heterogeneity":
            if used_inputs["particles_star_path"] and "cryodrgn" in allowed and _is_available("cryodrgn"):
                selected = "cryodrgn"
                reason = "particles_input_present_and_cryodrgn_available"
            elif "human_review" in allowed:
                selected = "human_review"
                reason = "fallback_human_review"
        elif category == "picking":
            if used_inputs["micrographs_star_path"] and "topaz" in allowed and _is_available("topaz"):
                selected = "topaz"
                reason = "micrographs_input_present_and_topaz_available"
            elif "relion_log" in allowed:
                selected = "relion_log"
                reason = "fallback_relion_log"

        for bid in allowed:
            if bid == selected:
                continue
            if not _is_available(bid):
                rejection_reasons[bid] = "unavailable"
            else:
                cap = capability.get(bid, {})
                # Light explainers by category.
                if category == "model_building" and bid == "mica" and not used_inputs["af3_dir"]:
                    rejection_reasons[bid] = "missing_af3_dir"
                elif category == "model_building" and bid == "emodelx" and not (used_inputs["map_path"] and used_inputs["sequence_path"]):
                    rejection_reasons[bid] = "missing_map_or_sequence"
                elif category == "heterogeneity" and bid == "cryodrgn" and not used_inputs["particles_star_path"]:
                    rejection_reasons[bid] = "missing_particles_input"
                elif category == "picking" and bid == "topaz" and not used_inputs["micrographs_star_path"]:
                    rejection_reasons[bid] = "missing_micrographs_input"
                elif cap:
                    rejection_reasons[bid] = "lower_priority_candidate"
                else:
                    rejection_reasons[bid] = "not_selected"
            rejected.append(bid)

        return {
            "category": category,
            "selected_backend": selected,
            "selection_reason": reason,
            "candidate_backends": allowed,
            "rejected_backends": rejected,
            "rejection_reasons": rejection_reasons,
            "used_inputs": used_inputs,
            "availability_snapshot": avail,
            "fallback_applied": bool(preferred_backend and selected != str(preferred_backend).strip().lower()),
        }
