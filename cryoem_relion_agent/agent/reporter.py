"""Reporter for JSON and Markdown outputs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from schemas.action import Action
from schemas.action_history import ActionHistoryEntry
from schemas.qc_metrics import QCMetrics
from schemas.report import ExecutionSummary, PipelineReport, RoutingDecision
from schemas.sample_state import SampleState

logger = logging.getLogger(__name__)


class Reporter:
    _ACTION_STAGE_LABELS: Dict[str, str] = {
        "run_autopick": "Picking",
        "run_extract": "Extract",
        "run_class2d": "Class2D",
        "run_class2d_cleanup": "Class2D",
        "run_init3d_or_class3d": "Init3D",
        "run_initmodel_3dclass": "Init3D",
        "run_refine3d": "Refine3D",
        "run_refine_ctf_polish": "Refine3D",
        "run_ctf_refine": "CtfRefine",
        "run_polish": "Polish",
        "run_postprocess": "Postprocess",
        "run_modelangelo": "ModelAngelo",
        "run_emodelx": "EModelX",
        "run_mica": "MICA",
        "run_cryodrgn": "cryoDRGN",
        "request_human_review": "HumanReview",
        "route_structure": "StructureRouter",
    }

    """Generate structured reports for current project status."""

    _MODEL_BUILDING_TAG_KEYS: tuple[str, ...] = (
        "model_building_backend",
        "model_building_availability",
        "model_building_execution_mode",
        "model_building_blocked_reason",
        "emodelx_repo",
        "emodelx_python",
        "mica_repo",
        "mica_python",
        "mica_af3_dir",
        "model_selection_reason",
    )

    _PICKING_TAG_KEYS: tuple[str, ...] = (
        "picking_backend",
        "picking_availability",
        "picking_execution_mode",
        "picking_blocked_reason",
        "topaz_executable",
        "picking_selection_reason",
    )

    _HETEROGENEITY_TAG_KEYS: tuple[str, ...] = (
        "heterogeneity_backend",
        "heterogeneity_availability",
        "heterogeneity_execution_mode",
        "heterogeneity_blocked_reason",
        "cryodrgn_executable",
        "heterogeneity_selection_reason",
    )

    _RECONSTRUCTION_TAG_KEYS: tuple[str, ...] = (
        "reconstruction_backend",
        "selected_backend",
        "reconstruction_selection_reason",
        "command_manifest",
        "execution_mode",
    )

    _ALL_SUMMARY_TAG_KEYS: tuple[str, ...] = (
        _PICKING_TAG_KEYS + _MODEL_BUILDING_TAG_KEYS + _HETEROGENEITY_TAG_KEYS + _RECONSTRUCTION_TAG_KEYS
    )

    @staticmethod
    def _parse_parenthesized_tag(text: str, key: str) -> Optional[str]:
        """Parse ``key=value`` inside scheduler ``(tag, tag, ...)`` summaries."""
        marker = f"{key}="
        pos = text.find(marker)
        if pos < 0:
            return None
        tail = text[pos + len(marker) :]
        stops: List[int] = []
        paren = tail.find(")")
        if paren >= 0:
            stops.append(paren)
        for other in Reporter._ALL_SUMMARY_TAG_KEYS:
            if other == key:
                continue
            seg = f", {other}="
            idx = tail.find(seg)
            if idx >= 0:
                stops.append(idx)
        end = min(stops) if stops else len(tail)
        return tail[:end].strip() or None

    @staticmethod
    def _extract_model_building_preflight(
        next_action: Optional[Action],
        history: Sequence[ActionHistoryEntry],
    ) -> Optional[Dict[str, Any]]:
        if next_action and isinstance(next_action.parameters, dict):
            raw = next_action.parameters.get("model_building_preflight")
            if not isinstance(raw, dict):
                raw = next_action.parameters.get("emodelx_preflight")
            if not isinstance(raw, dict):
                raw = next_action.parameters.get("mica_preflight")
            if isinstance(raw, dict):
                return raw
            p = next_action.parameters
            if p.get("model_building_availability") or p.get("model_building_execution_mode"):
                mi = p.get("model_building_missing_inputs")
                if not isinstance(mi, list):
                    mi = []
                br = p.get("model_building_blocked_reason")
                if br == "":
                    br = None
                return {
                    "backend_name": p.get("model_building_backend"),
                    "available": str(p.get("model_building_availability", "")).strip() == "available",
                    "execution_mode": str(p.get("model_building_execution_mode") or ""),
                    "blocked_reason": br,
                    "missing_inputs": mi,
                    "missing_dependencies": list(p.get("model_building_missing_dependencies") or []),
                    "warnings": [],
                    "repo_path": p.get("emodelx_repo_dir") or p.get("mica_repo_dir"),
                    "python_path": p.get("emodelx_python") or p.get("mica_python"),
                    "af3_dir": p.get("af3_dir"),
                    "output_path": p.get("output_dir"),
                }
        for item in reversed(list(history)):
            text = str(item.output_summary or "")
            if "model_building_availability=" not in text and "model_building_execution_mode=" not in text:
                continue
            avail_s = Reporter._parse_parenthesized_tag(text, "model_building_availability")
            mode_s = Reporter._parse_parenthesized_tag(text, "model_building_execution_mode")
            br_s = Reporter._parse_parenthesized_tag(text, "model_building_blocked_reason")
            if not avail_s and not mode_s:
                continue
            return {
                "available": avail_s == "available",
                "execution_mode": mode_s or "",
                "blocked_reason": br_s,
                "backend_name": Reporter._parse_parenthesized_tag(text, "model_building_backend"),
                "repo_path": Reporter._parse_parenthesized_tag(text, "mica_repo")
                or Reporter._parse_parenthesized_tag(text, "emodelx_repo"),
                "python_path": Reporter._parse_parenthesized_tag(text, "mica_python")
                or Reporter._parse_parenthesized_tag(text, "emodelx_python"),
                "af3_dir": Reporter._parse_parenthesized_tag(text, "mica_af3_dir"),
                "output_path": Reporter._parse_parenthesized_tag(text, "mica_output")
                or Reporter._parse_parenthesized_tag(text, "emodelx_output"),
            }
        return None

    @staticmethod
    def _extract_picking_preflight(
        next_action: Optional[Action],
        history: Sequence[ActionHistoryEntry],
    ) -> Optional[Dict[str, Any]]:
        if next_action and isinstance(next_action.parameters, dict):
            p = next_action.parameters
            if str(p.get("picking_backend", "")).lower() == "topaz":
                raw_pf = p.get("topaz_preflight")
                if isinstance(raw_pf, dict):
                    return raw_pf
                if p.get("picking_availability") or p.get("picking_execution_mode"):
                    mi = p.get("picking_missing_inputs")
                    if not isinstance(mi, list):
                        mi = []
                    br = p.get("picking_blocked_reason")
                    if br == "":
                        br = None
                    return {
                        "backend_name": "topaz",
                        "available": str(p.get("picking_availability", "")).strip() == "available",
                        "execution_mode": str(p.get("picking_execution_mode") or ""),
                        "blocked_reason": br,
                        "missing_inputs": mi,
                        "missing_dependencies": list(p.get("picking_missing_dependencies") or []),
                    }
        for item in reversed(list(history)):
            text = str(item.output_summary or "")
            if "picking_availability=" not in text and "picking_execution_mode=" not in text:
                continue
            avail_s = Reporter._parse_parenthesized_tag(text, "picking_availability")
            mode_s = Reporter._parse_parenthesized_tag(text, "picking_execution_mode")
            br_s = Reporter._parse_parenthesized_tag(text, "picking_blocked_reason")
            if not avail_s and not mode_s:
                continue
            return {
                "available": avail_s == "available",
                "execution_mode": mode_s or "",
                "blocked_reason": br_s,
                "backend_name": "topaz",
                "executable_path": Reporter._parse_parenthesized_tag(text, "topaz_executable"),
            }
        return None

    @staticmethod
    def _extract_picking_backend(next_action: Optional[Action], history: Sequence[ActionHistoryEntry]) -> Optional[str]:
        if next_action and next_action.name.value == "run_autopick":
            v = str(next_action.parameters.get("picking_backend", "")).strip()
            if v:
                return v
        for item in reversed(list(history)):
            if item.action_name != "run_autopick":
                continue
            text = str(item.output_summary or "")
            v = Reporter._parse_parenthesized_tag(text, "picking_backend")
            if v:
                return v
        return None

    @staticmethod
    def _extract_heterogeneity_backend(
        next_action: Optional[Action],
        history: Sequence[ActionHistoryEntry],
    ) -> Optional[str]:
        if next_action and next_action.name.value in {"request_human_review", "run_cryodrgn"}:
            v = str(next_action.parameters.get("heterogeneity_backend", "")).strip()
            if v:
                return v
        for item in reversed(list(history)):
            if item.action_name not in {"run_cryodrgn", "request_human_review"}:
                continue
            text = str(item.output_summary or "")
            v = Reporter._parse_parenthesized_tag(text, "heterogeneity_backend")
            if v:
                return v
        return None

    @staticmethod
    def _extract_heterogeneity_preflight(
        next_action: Optional[Action],
        history: Sequence[ActionHistoryEntry],
    ) -> Optional[Dict[str, Any]]:
        if next_action and isinstance(next_action.parameters, dict):
            p = next_action.parameters
            if str(p.get("heterogeneity_backend", "")).strip().lower() == "cryodrgn":
                raw_pf = p.get("cryodrgn_preflight")
                if isinstance(raw_pf, dict) and raw_pf:
                    return raw_pf
                if p.get("heterogeneity_availability") or p.get("heterogeneity_execution_mode"):
                    mi = p.get("heterogeneity_missing_inputs")
                    if not isinstance(mi, list):
                        mi = []
                    br = p.get("heterogeneity_blocked_reason")
                    if br == "":
                        br = None
                    return {
                        "backend_name": "cryodrgn",
                        "available": str(p.get("heterogeneity_availability", "")).strip() == "available",
                        "execution_mode": str(p.get("heterogeneity_execution_mode") or ""),
                        "blocked_reason": br,
                        "missing_inputs": mi,
                        "missing_dependencies": list(p.get("heterogeneity_missing_dependencies") or []),
                        "executable_path": None,
                        "warnings": [],
                    }
        for item in reversed(list(history)):
            text = str(item.output_summary or "")
            if "heterogeneity_availability=" not in text and "heterogeneity_execution_mode=" not in text:
                continue
            avail_s = Reporter._parse_parenthesized_tag(text, "heterogeneity_availability")
            mode_s = Reporter._parse_parenthesized_tag(text, "heterogeneity_execution_mode")
            br_s = Reporter._parse_parenthesized_tag(text, "heterogeneity_blocked_reason")
            ex_s = Reporter._parse_parenthesized_tag(text, "cryodrgn_executable")
            if not avail_s and not mode_s:
                continue
            return {
                "backend_name": "cryodrgn",
                "available": avail_s == "available",
                "execution_mode": mode_s or "",
                "blocked_reason": br_s,
                "executable_path": ex_s,
                "missing_inputs": [],
                "missing_dependencies": [],
                "warnings": [],
            }
        return None

    @staticmethod
    def _extract_reconstruction_backend(
        next_action: Optional[Action],
        history: Sequence[ActionHistoryEntry],
    ) -> Optional[str]:
        if next_action and next_action.name.value == "run_cryodrgn_ai":
            v = str(next_action.parameters.get("reconstruction_backend", "")).strip()
            if v:
                return v
        for item in reversed(list(history)):
            if item.action_name != "run_cryodrgn_ai":
                continue
            text = str(item.output_summary or "")
            v = Reporter._parse_parenthesized_tag(text, "reconstruction_backend") or Reporter._parse_parenthesized_tag(text, "selected_backend")
            if v:
                return v
        return None

    @staticmethod
    def _extract_cryodrgn_ai_preflight(
        next_action: Optional[Action],
        history: Sequence[ActionHistoryEntry],
    ) -> Optional[Dict[str, Any]]:
        if next_action and isinstance(next_action.parameters, dict) and next_action.name.value == "run_cryodrgn_ai":
            raw = next_action.parameters.get("cryodrgn_ai_preflight")
            if isinstance(raw, dict):
                return raw
        for item in reversed(list(history)):
            if item.action_name != "run_cryodrgn_ai":
                continue
            text = str(item.output_summary or "")
            selected = Reporter._parse_parenthesized_tag(text, "reconstruction_backend") or Reporter._parse_parenthesized_tag(text, "selected_backend")
            if selected:
                return {
                    "backend_name": selected,
                    "command_manifest": Reporter._parse_parenthesized_tag(text, "command_manifest"),
                    "execution_mode": Reporter._parse_parenthesized_tag(text, "execution_mode"),
                }
        return None

    @staticmethod
    def _extract_model_building_backend(
        next_action: Optional[Action],
        history: Sequence[ActionHistoryEntry],
    ) -> Optional[str]:
        if next_action and next_action.name.value in {"run_modelangelo", "run_emodelx", "run_mica"}:
            v = str(next_action.parameters.get("model_building_backend", "")).strip()
            if v:
                return v
        for item in reversed(list(history)):
            text = str(item.output_summary or "")
            v = Reporter._parse_parenthesized_tag(text, "model_building_backend")
            if v:
                return v
        return None

    def build_report(
        self,
        state: SampleState,
        qc: QCMetrics,
        recent_action: Optional[Action],
        next_action: Optional[Action],
        structure_tool: Optional[str] = None,
        structure_reason: Optional[str] = None,
        action_history: Optional[Sequence[ActionHistoryEntry]] = None,
        current_run_id: Optional[str] = None,
        capability_matrix: Optional[Dict[str, Dict[str, Any]]] = None,
        benchmark_summary: Optional[Dict[str, Any]] = None,
    ) -> PipelineReport:
        """Build a PipelineReport object."""
        human_review_required = False
        human_review_reasons: list[str] = []

        if next_action and next_action.name.value == "request_human_review":
            human_review_required = True
            human_review_reasons.append(next_action.reason)

        routing = self._build_routing_section(state, qc, next_action, structure_tool, structure_reason)

        execution: Optional[ExecutionSummary] = None
        if state.last_execution:
            le = state.last_execution
            execution = ExecutionSummary(
                action_name=le.get("action_name"),
                status=le.get("status"),
                command=le.get("command"),
                dry_run=le.get("dry_run"),
                return_code=le.get("return_code"),
                output_dir=le.get("output_dir"),
                generated_model_path=le.get("generated_model_path"),
                error_message=le.get("error_message"),
            )

        history = list(action_history or [])
        report_history = self._pick_run_history(history, current_run_id=current_run_id)
        selected_run_id = current_run_id or (report_history[-1].run_id if report_history else None)
        stage_transition_summary = self._summarize_stage_transitions(report_history)
        branch_summaries = self._summarize_branches(report_history)
        picking_backend = self._extract_picking_backend(next_action=next_action, history=report_history)
        heterogeneity_backend = self._extract_heterogeneity_backend(next_action=next_action, history=report_history)
        model_building_backend = self._extract_model_building_backend(next_action=next_action, history=report_history)
        reconstruction_backend = self._extract_reconstruction_backend(next_action=next_action, history=report_history)
        mb_preflight = self._extract_model_building_preflight(next_action=next_action, history=report_history)
        pk_preflight = self._extract_picking_preflight(next_action=next_action, history=report_history)
        het_preflight = self._extract_heterogeneity_preflight(next_action=next_action, history=report_history)
        cryodrgn_ai_preflight = self._extract_cryodrgn_ai_preflight(next_action=next_action, history=report_history)
        extra: Dict[str, str] = {}
        if picking_backend:
            extra["picking_backend"] = picking_backend
        if heterogeneity_backend:
            extra["heterogeneity_backend"] = heterogeneity_backend
        if model_building_backend:
            extra["model_building_backend"] = model_building_backend
        if reconstruction_backend:
            extra["reconstruction_backend"] = reconstruction_backend
        if mb_preflight:
            extra["model_building_availability"] = "available" if mb_preflight.get("available") else "unavailable"
            if mb_preflight.get("execution_mode"):
                extra["model_building_execution_mode"] = str(mb_preflight["execution_mode"])
            if mb_preflight.get("blocked_reason"):
                extra["model_building_blocked_reason"] = str(mb_preflight["blocked_reason"])
            mb_name = str(model_building_backend or mb_preflight.get("backend_name") or "").lower()
            if mb_preflight.get("repo_path"):
                if mb_name == "mica":
                    extra["mica_repo"] = str(mb_preflight["repo_path"])[:500]
                else:
                    extra["emodelx_repo"] = str(mb_preflight["repo_path"])[:500]
            if mb_preflight.get("python_path"):
                if mb_name == "mica":
                    extra["mica_python"] = str(mb_preflight["python_path"])[:500]
                else:
                    extra["emodelx_python"] = str(mb_preflight["python_path"])[:500]
            if mb_preflight.get("output_path"):
                if mb_name == "mica":
                    extra["mica_output"] = str(mb_preflight["output_path"])[:500]
                else:
                    extra["emodelx_output"] = str(mb_preflight["output_path"])[:500]
            if mb_preflight.get("af3_dir"):
                extra["mica_af3_dir"] = str(mb_preflight["af3_dir"])[:500]
        if pk_preflight:
            extra["picking_availability"] = "available" if pk_preflight.get("available") else "unavailable"
            if pk_preflight.get("execution_mode"):
                extra["picking_execution_mode"] = str(pk_preflight["execution_mode"])
            if pk_preflight.get("blocked_reason"):
                extra["picking_blocked_reason"] = str(pk_preflight["blocked_reason"])
            ex = pk_preflight.get("executable_path")
            if ex:
                extra["topaz_executable"] = str(ex)[:500]
        if next_action and isinstance(next_action.parameters, dict):
            ps = next_action.parameters.get("picking_selection_decision")
            hs = next_action.parameters.get("heterogeneity_selection_decision")
            ms = next_action.parameters.get("model_building_selection_decision")
            if isinstance(ps, dict) and ps.get("selection_reason"):
                extra["picking_selection_reason"] = str(ps.get("selection_reason"))
            if isinstance(hs, dict) and hs.get("selection_reason"):
                extra["heterogeneity_selection_reason"] = str(hs.get("selection_reason"))
            if isinstance(ms, dict) and ms.get("selection_reason"):
                extra["model_selection_reason"] = str(ms.get("selection_reason"))
            rs = next_action.parameters.get("reconstruction_selection_decision")
            if isinstance(rs, dict) and rs.get("selection_reason"):
                extra["reconstruction_selection_reason"] = str(rs.get("selection_reason"))
            if next_action.parameters.get("command_manifest"):
                extra["command_manifest"] = str(next_action.parameters.get("command_manifest"))
        if het_preflight:
            extra["heterogeneity_availability"] = "available" if het_preflight.get("available") else "unavailable"
            if het_preflight.get("execution_mode"):
                extra["heterogeneity_execution_mode"] = str(het_preflight["execution_mode"])
            if het_preflight.get("blocked_reason"):
                extra["heterogeneity_blocked_reason"] = str(het_preflight["blocked_reason"])
            hex_p = het_preflight.get("executable_path")
            if hex_p:
                extra["cryodrgn_executable"] = str(hex_p)[:500]
        if cryodrgn_ai_preflight:
            extra["cryodrgn_ai_preflight_result"] = "available" if cryodrgn_ai_preflight.get("available") else "unavailable"
            if cryodrgn_ai_preflight.get("execution_mode"):
                extra["reconstruction_execution_mode"] = str(cryodrgn_ai_preflight["execution_mode"])
            if cryodrgn_ai_preflight.get("blocked_reason"):
                extra["reconstruction_blocked_reason"] = str(cryodrgn_ai_preflight["blocked_reason"])
        if state.last_execution and isinstance(state.last_execution, dict):
            le = state.last_execution
            if le.get("action_name") == "run_cryodrgn_ai":
                for key in ("command_manifest", "stdout_log", "stderr_log", "recommended_downstream_action"):
                    if le.get(key):
                        extra[key] = str(le[key])[:1000]
                if le.get("output_artifacts"):
                    extra["output_artifacts"] = str(le.get("output_artifacts"))[:1000]

        return PipelineReport(
            state=state,
            qc=qc,
            recent_action=recent_action,
            next_action=next_action,
            human_review_required=human_review_required,
            human_review_reasons=human_review_reasons,
            structure_routing=routing,
            execution=execution,
            action_history=report_history,
            run_id=selected_run_id,
            stage_transition_summary=stage_transition_summary,
            branch_summaries=branch_summaries,
            extra=extra,
            model_building_preflight=mb_preflight,
            picking_preflight=pk_preflight,
            heterogeneity_preflight=het_preflight,
            cryodrgn_ai_preflight=cryodrgn_ai_preflight,
            reconstruction_preflight=cryodrgn_ai_preflight,
            capability_matrix=capability_matrix or {},
            backend_selection={
                "picking": (next_action.parameters.get("picking_selection_decision") if next_action and isinstance(next_action.parameters, dict) else None),
                "heterogeneity": (next_action.parameters.get("heterogeneity_selection_decision") if next_action and isinstance(next_action.parameters, dict) else None),
                "reconstruction": (next_action.parameters.get("reconstruction_selection_decision") if next_action and isinstance(next_action.parameters, dict) else None),
                "model_building": (next_action.parameters.get("model_building_selection_decision") if next_action and isinstance(next_action.parameters, dict) else None),
            },
            benchmark_summary=benchmark_summary,
        )

    @staticmethod
    def _pick_run_history(
        history: Sequence[ActionHistoryEntry],
        current_run_id: Optional[str],
    ) -> list[ActionHistoryEntry]:
        if not history:
            return []
        run_id = current_run_id or history[-1].run_id
        if not run_id:
            return list(history)
        return [x for x in history if x.run_id == run_id] or list(history)

    @staticmethod
    def _summarize_stage_transitions(history: Sequence[ActionHistoryEntry]) -> str:
        if not history:
            return "N/A"
        main = [x for x in history if (x.branch_name or "main") == "main"]
        src = main or list(history)
        stages: list[str] = []
        for item in src:
            s = Reporter._ACTION_STAGE_LABELS.get(item.action_name, item.stage or "unknown")
            if not stages or stages[-1] != s:
                stages.append(s)
        return " -> ".join(stages) if stages else "N/A"

    @staticmethod
    def _summarize_branches(history: Sequence[ActionHistoryEntry]) -> Dict[str, str]:
        grouped: Dict[str, list[str]] = {}
        for item in history:
            branch = item.branch_name or "main"
            grouped.setdefault(branch, []).append(item.action_name)
        return {k: " -> ".join(v) for k, v in grouped.items()}

    def _build_routing_section(
        self,
        state: SampleState,
        qc: QCMetrics,
        next_action: Optional[Action],
        structure_tool: Optional[str],
        structure_reason: Optional[str],
    ) -> RoutingDecision:
        chosen = structure_tool
        reason = structure_reason
        seq = state.sequence_path

        if next_action and next_action.name.value in {"run_modelangelo", "run_emodelx", "run_mica"}:
            chosen = next_action.name.value
            reason = next_action.parameters.get("routing_reason") or next_action.reason
            return RoutingDecision(
                triggered=True,
                sequence_path=seq,
                selected_tool=chosen,
                rationale=reason or "structure_router",
                alternatives_considered=["run_modelangelo", "run_emodelx", "run_mica"],
            )

        if next_action and next_action.name.value == "run_validation":
            decline_reason = str(next_action.parameters.get("routing_reason", "missing_sequence"))
            return RoutingDecision(
                triggered=False,
                sequence_path=seq,
                selected_tool=None,
                rationale=decline_reason,
                alternatives_considered=["run_modelangelo", "run_emodelx", "run_mica"],
            )

        if chosen:
            return RoutingDecision(
                triggered=True,
                sequence_path=seq,
                selected_tool=chosen,
                rationale=reason or "routing_triggered",
                alternatives_considered=["run_modelangelo", "run_emodelx", "run_mica"],
            )

        if not state.map_path:
            decline_reason = "missing_map_path"
        elif not state.sequence_path:
            decline_reason = "missing_sequence"
        elif not (qc.refine3d_completed or qc.current_resolution is not None):
            decline_reason = "insufficient_qc"
        else:
            decline_reason = structure_reason or "router_declined"

        return RoutingDecision(
            triggered=False,
            sequence_path=seq,
            selected_tool=None,
            rationale=decline_reason,
            alternatives_considered=["run_modelangelo", "run_emodelx", "run_mica"],
        )

    def write_json(self, report: PipelineReport, out_path: Path) -> bool:
        """Write report to JSON file."""
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = report.model_dump(mode="json", exclude_none=False)
            out_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("JSON report written: %s", out_path)
            return True
        except OSError as exc:
            logger.error("Failed to write JSON report %s: %s", out_path, exc)
            return False

    def write_markdown(self, report: PipelineReport, out_path: Path) -> bool:
        """Write report to Markdown file."""
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(report.to_markdown(), encoding="utf-8")
            logger.info("Markdown report written: %s", out_path)
            return True
        except OSError as exc:
            logger.error("Failed to write Markdown report %s: %s", out_path, exc)
            return False
