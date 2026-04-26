"""Report schemas for structured JSON and Markdown generation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from schemas.action import Action
from schemas.action_history import ActionHistoryEntry
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState


class ReportMeta(BaseModel):
    """Metadata for generated reports."""

    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent_version: str = Field(default="0.1.0")
    report_type: str = Field(default="cryoem_relion_agent_report")


class ExecutionSummary(BaseModel):
    """Last run-next execution details (e.g. ModelAngelo)."""

    action_name: Optional[str] = None
    status: Optional[str] = None
    command: Optional[str] = None
    dry_run: Optional[bool] = None
    return_code: Optional[int] = None
    output_dir: Optional[str] = None
    generated_model_path: Optional[str] = None
    error_message: Optional[str] = None


class RoutingDecision(BaseModel):
    """Optional structure-routing decision details."""

    triggered: bool = False
    sequence_path: Optional[str] = None
    selected_tool: Optional[str] = None
    rationale: Optional[str] = None
    alternatives_considered: List[str] = Field(default_factory=list)


class PipelineReport(BaseModel):
    """Top-level report model."""

    meta: ReportMeta = Field(default_factory=ReportMeta)
    state: SampleState
    qc: QCMetrics
    recent_action: Optional[Action] = None
    next_action: Optional[Action] = None
    human_review_required: bool = False
    human_review_reasons: List[str] = Field(default_factory=list)
    structure_routing: Optional[RoutingDecision] = None
    execution: Optional[ExecutionSummary] = None
    action_history: List[ActionHistoryEntry] = Field(default_factory=list)
    run_id: Optional[str] = None
    stage_transition_summary: Optional[str] = None
    branch_summaries: Dict[str, str] = Field(default_factory=dict)
    extra: Dict[str, str] = Field(default_factory=dict)
    model_building_preflight: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured model-building preflight (backend, availability, execution_mode, blocked_reason, ...).",
    )
    picking_preflight: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured Topaz / picking preflight (availability, executable_path, blocked_reason, ...).",
    )
    picking_qc: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured picking QC gate result (qc_status, n_picks, qc_flags, ground_truth_evaluation, ...).",
    )
    heterogeneity_preflight: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured cryoDRGN heterogeneity preflight (availability, execution_mode, blocked_reason, ...).",
    )
    reconstruction_preflight: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured reconstruction preflight for optional backends such as cryodrgn_ai.",
    )
    candidate_map_routing: List[Dict[str, Any]] = Field(default_factory=list)
    capability_matrix: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    backend_selection: Dict[str, Optional[Dict[str, Any]]] = Field(default_factory=dict)
    benchmark_summary: Optional[Dict[str, Any]] = None

    def to_markdown(self) -> str:
        """Render a concise markdown report."""
        lines: List[str] = []
        lines.append("# Cryo-EM RELION Agent Report")
        lines.append("")
        lines.append(f"- Generated at: {self.meta.generated_at.isoformat()}")
        lines.append(f"- Agent version: {self.meta.agent_version}")
        lines.append("")
        lines.append("## Project")
        lines.append(f"- Project ID: {self.state.project_id}")
        lines.append(f"- Project Path: {self.state.project_path}")
        lines.append(f"- Stage: {self.state.stage}")
        lines.append(f"- Status: {self.state.status}")
        lines.append(f"- Sequence path: {self.state.sequence_path or 'N/A'}")
        lines.append("")
        lines.append("## QC Summary")
        lines.append(f"- MotionCorr completed: {self.qc.motioncorr_completed}")
        lines.append(f"- CTF completed: {self.qc.ctf_completed}")
        lines.append(f"- Picking completed: {self.qc.picking_completed}")
        lines.append(f"- 2D completed: {self.qc.class2d_completed}")
        lines.append(f"- Initial 3D completed: {self.qc.init3d_completed}")
        lines.append(f"- Refine 3D completed: {self.qc.refine3d_completed}")
        lines.append(f"- Current resolution: {self.qc.current_resolution}")
        lines.append(f"- Confidence score: {self.qc.confidence_score}")
        lines.append("")
        lines.append("## Actions")
        lines.append(f"- Recent action: {self.recent_action.name.value if self.recent_action else 'N/A'}")
        lines.append(f"- Next action: {self.next_action.name.value if self.next_action else 'N/A'}")
        if self.next_action:
            lines.append(f"- Next action reason: {self.next_action.reason}")
        if (
            self.next_action
            and self.next_action.name.value == "request_human_review"
            and isinstance(self.next_action.parameters, dict)
            and self.next_action.parameters.get("branch_name") == "picking_qc"
        ):
            params = self.next_action.parameters
            lines.append("")
            lines.append("## Picking QC Gate")
            lines.append(f"- Status: {params.get('picking_qc_status', 'N/A')}")
            flags = params.get("picking_qc_flags") or []
            if flags:
                lines.append(f"- Flags: {', '.join(str(f) for f in flags)}")
            lines.append(f"- Recommendation: {params.get('recommended_next_action', 'N/A')}")
            if params.get("f1") is not None:
                lines.append(f"- F1: {params.get('f1')}")
        if self.extra.get("picking_backend"):
            lines.append(f"- Picking backend: {self.extra.get('picking_backend')}")
        if self.extra.get("picking_selection_reason"):
            lines.append(f"- Picking selection reason: {self.extra.get('picking_selection_reason')}")
        pk = self.picking_preflight
        if pk:
            lines.append(f"- Picking availability: {'available' if pk.get('available') else 'unavailable'}")
            if pk.get("execution_mode"):
                lines.append(f"- Picking execution mode: {pk['execution_mode']}")
            if pk.get("blocked_reason"):
                lines.append(f"- Picking blocked reason: {pk['blocked_reason']}")
            if pk.get("executable_path"):
                lines.append(f"- Topaz executable: {pk['executable_path']}")
        elif self.extra.get("picking_availability"):
            lines.append(f"- Picking availability: {self.extra.get('picking_availability')}")
            if self.extra.get("picking_execution_mode"):
                lines.append(f"- Picking execution mode: {self.extra.get('picking_execution_mode')}")
            if self.extra.get("picking_blocked_reason"):
                lines.append(f"- Picking blocked reason: {self.extra.get('picking_blocked_reason')}")
            if self.extra.get("topaz_executable"):
                lines.append(f"- Topaz executable: {self.extra.get('topaz_executable')}")
        if self.extra.get("heterogeneity_backend"):
            lines.append(f"- Heterogeneity backend: {self.extra.get('heterogeneity_backend')}")
        if self.extra.get("heterogeneity_selection_reason"):
            lines.append(f"- Heterogeneity selection reason: {self.extra.get('heterogeneity_selection_reason')}")
        hf = self.heterogeneity_preflight
        if hf:
            lines.append(f"- Heterogeneity availability: {'available' if hf.get('available') else 'unavailable'}")
            if hf.get("execution_mode"):
                lines.append(f"- Heterogeneity execution mode: {hf['execution_mode']}")
            if hf.get("blocked_reason"):
                lines.append(f"- Heterogeneity blocked reason: {hf['blocked_reason']}")
            if hf.get("executable_path"):
                lines.append(f"- cryoDRGN executable: {hf['executable_path']}")
            if hf.get("input_summary"):
                lines.append(f"- cryoDRGN input summary: {hf['input_summary']}")
        elif self.extra.get("heterogeneity_availability"):
            lines.append(f"- Heterogeneity availability: {self.extra.get('heterogeneity_availability')}")
            if self.extra.get("heterogeneity_execution_mode"):
                lines.append(f"- Heterogeneity execution mode: {self.extra.get('heterogeneity_execution_mode')}")
            if self.extra.get("heterogeneity_blocked_reason"):
                lines.append(f"- Heterogeneity blocked reason: {self.extra.get('heterogeneity_blocked_reason')}")
            if self.extra.get("cryodrgn_executable"):
                lines.append(f"- cryoDRGN executable: {self.extra.get('cryodrgn_executable')}")
        if self.extra.get("reconstruction_backend"):
            lines.append(f"- Reconstruction backend: {self.extra.get('reconstruction_backend')}")
        if self.extra.get("reconstruction_selection_reason"):
            lines.append(f"- Reconstruction selection reason: {self.extra.get('reconstruction_selection_reason')}")
        rf = self.reconstruction_preflight
        if rf:
            lines.append(f"- cryodrgn_ai preflight: {'available' if rf.get('available') else 'unavailable'}")
            if rf.get("execution_mode"):
                lines.append(f"- Reconstruction execution mode: {rf['execution_mode']}")
            if rf.get("blocked_reason"):
                lines.append(f"- Reconstruction blocked reason: {rf['blocked_reason']}")
            if rf.get("executable_path"):
                lines.append(f"- cryodrgn_ai executable: {rf['executable_path']}")
        for key, label in (
            ("command_manifest", "Command manifest"),
            ("stdout_log", "stdout log"),
            ("stderr_log", "stderr log"),
            ("output_artifacts", "Output artifacts"),
        ):
            if self.extra.get(key):
                lines.append(f"- {label}: {self.extra.get(key)}")
        if self.extra.get("recommended_downstream_action"):
            lines.append(f"- Recommended downstream action: {self.extra.get('recommended_downstream_action')}")
        if self.extra.get("cryodrgn_ai_sampled_volume_count"):
            lines.append("")
            lines.append("## cryodrgn_ai Sampled Maps")
            lines.append(f"- Sampled volumes: {self.extra.get('cryodrgn_ai_sampled_volume_count', '0')}")
            lines.append(f"- Candidate maps: {self.extra.get('cryodrgn_ai_candidate_map_count', '0')}")
            paths_raw = self.extra.get("cryodrgn_ai_sampled_volume_paths", "")
            if paths_raw:
                for p in paths_raw.split("|")[:3]:
                    lines.append(f"  - {p}")
            if self.extra.get("cryodrgn_ai_recommended_downstream"):
                lines.append(f"- Recommended downstream: {self.extra.get('cryodrgn_ai_recommended_downstream')}")
        lines.append("")
        lines.append("## Candidate Map Routing")
        candidate_count = int(self.extra.get("candidate_map_count", "0") or "0")
        lines.append(f"- Candidate count: {candidate_count}")
        if candidate_count:
            lines.append(f"- cryodrgn_ai produced {candidate_count} candidate maps for downstream routing.")
        if self.candidate_map_routing:
            lines.append(f"- Routing records: {len(self.candidate_map_routing)}")
            for record in self.candidate_map_routing[:5]:
                pf = record.get("preflight") if isinstance(record.get("preflight"), dict) else {}
                status = "available" if pf.get("available") else "unavailable"
                lines.append(f"- candidate {record.get('candidate_index')}:")
                lines.append(f"  - map: {record.get('map_path') or 'N/A'}")
                lines.append(f"  - backend: {record.get('recommended_backend') or 'N/A'}")
                lines.append(
                    "  - action: "
                    f"{record.get('downstream_action') or record.get('recommended_action') or 'N/A'}"
                )
                lines.append(f"  - preflight: {status}")
                lines.append(f"  - latent_source: {record.get('latent_source') or 'N/A'}")
                lines.append(f"  - weights_path: {record.get('weights_path') or 'N/A'}")
                if pf.get("blocked_reason"):
                    lines.append(f"  - blocked_reason: {pf.get('blocked_reason')}")
        else:
            lines.append("- No candidate maps available for routing")
        if self.extra.get("model_building_backend"):
            lines.append(f"- Model-building backend: {self.extra.get('model_building_backend')}")
        if self.extra.get("model_selection_reason"):
            lines.append(f"- Model-building selection reason: {self.extra.get('model_selection_reason')}")
        mb_backend = str(self.extra.get("model_building_backend") or "").lower()
        pf = self.model_building_preflight
        if pf:
            lines.append(f"- Availability: {'available' if pf.get('available') else 'unavailable'}")
            if pf.get("execution_mode"):
                lines.append(f"- Execution mode: {pf['execution_mode']}")
            if pf.get("blocked_reason"):
                lines.append(f"- Blocked reason: {pf['blocked_reason']}")
            if pf.get("repo_path"):
                lines.append(f"- {'MICA' if mb_backend == 'mica' else 'EModelX'} repo: {pf['repo_path']}")
            if pf.get("python_path"):
                lines.append(f"- {'MICA' if mb_backend == 'mica' else 'EModelX'} python: {pf['python_path']}")
            if pf.get("run_py_path"):
                lines.append(f"- {'MICA' if mb_backend == 'mica' else 'EModelX'} run.py: {pf['run_py_path']}")
            if pf.get("command"):
                lines.append(f"- {'MICA' if mb_backend == 'mica' else 'EModelX'} command: {pf['command']}")
            if pf.get("output_path"):
                lines.append(f"- Output path: {pf['output_path']}")
            if pf.get("af3_dir"):
                lines.append(f"- AF3 dir: {pf['af3_dir']}")
        elif self.extra.get("model_building_availability"):
            lines.append(f"- Availability: {self.extra.get('model_building_availability')}")
            if self.extra.get("model_building_execution_mode"):
                lines.append(f"- Execution mode: {self.extra.get('model_building_execution_mode')}")
            if self.extra.get("model_building_blocked_reason"):
                lines.append(f"- Blocked reason: {self.extra.get('model_building_blocked_reason')}")
            if self.extra.get("emodelx_repo"):
                lines.append(f"- EModelX repo: {self.extra.get('emodelx_repo')}")
            if self.extra.get("emodelx_python"):
                lines.append(f"- EModelX python: {self.extra.get('emodelx_python')}")
            if self.extra.get("emodelx_output"):
                lines.append(f"- Output path: {self.extra.get('emodelx_output')}")
            if self.extra.get("mica_repo"):
                lines.append(f"- MICA repo: {self.extra.get('mica_repo')}")
            if self.extra.get("mica_python"):
                lines.append(f"- MICA python: {self.extra.get('mica_python')}")
            if self.extra.get("mica_af3_dir"):
                lines.append(f"- AF3 dir: {self.extra.get('mica_af3_dir')}")
            if self.extra.get("mica_output"):
                lines.append(f"- Output path: {self.extra.get('mica_output')}")
        if self.picking_qc:
            qc = self.picking_qc
            lines.append("")
            lines.append("## Picking QC")
            lines.append(f"- Backend: {self.extra.get('picking_backend') or 'N/A'}")
            lines.append(f"- QC status: {qc.get('qc_status') or 'N/A'}")
            lines.append(f"- Picks: {qc.get('n_picks')}")
            n_with = qc.get("n_micrographs_with_picks")
            n_total = qc.get("n_micrographs_total")
            with_str = "N/A" if n_with is None else str(n_with)
            if n_total is not None:
                lines.append(f"- Micrographs with picks: {with_str} / {n_total}")
            elif n_with is not None:
                lines.append(f"- Micrographs with picks: {with_str}")
            mp = qc.get("mean_picks_per_micrograph")
            if mp is not None:
                lines.append(f"- Mean picks/micrograph: {mp}")
            if qc.get("zero_pick_micrograph_ratio") is not None:
                lines.append(f"- Zero-pick micrograph ratio: {qc.get('zero_pick_micrograph_ratio')}")
            ge = qc.get("ground_truth_evaluation") or {}
            if ge:
                tol = ge.get("matching_tolerance_px")
                if tol is None:
                    tol = qc.get("matching_tolerance_px")
                tol_str = "N/A" if tol is None else f"{tol}px"
                lines.append(
                    f"- GT match: precision={ge.get('precision')} recall={ge.get('recall')} f1={ge.get('f1')}"
                    f" tol={tol_str}"
                )
            flags = qc.get("qc_flags") or []
            if flags:
                lines.append(f"- Flags: {', '.join(flags)}")
            rec = qc.get("recommended_next_action")
            if rec:
                lines.append(f"- Recommendation: {rec}")
        lines.append("")
        lines.append("## Backend Capabilities")
        lines.append(f"- Backend entries: {len(self.capability_matrix)}")
        lines.append("")
        lines.append("## Backend Selection")
        for cat, decision in self.backend_selection.items():
            if isinstance(decision, dict) and decision.get("selected_backend"):
                lines.append(
                    f"- {cat}: selected={decision.get('selected_backend')} reason={decision.get('selection_reason')}"
                )
        lines.append("")
        lines.append("## Benchmark")
        if self.benchmark_summary:
            if self.benchmark_summary.get("run_id"):
                lines.append(f"- Run ID: {self.benchmark_summary.get('run_id')}")
            if self.benchmark_summary.get("timestamp"):
                lines.append(f"- Timestamp: {self.benchmark_summary.get('timestamp')}")
            lines.append(f"- Cases: {self.benchmark_summary.get('case_count', 'N/A')}")
            lines.append(f"- Completed: {self.benchmark_summary.get('completed', 'N/A')}")
            lines.append(f"- Blocked: {self.benchmark_summary.get('blocked', 'N/A')}")
            lines.append(f"- Failed: {self.benchmark_summary.get('failed', 'N/A')}")
            if self.benchmark_summary.get("backends_covered"):
                lines.append(f"- Backends: {', '.join(self.benchmark_summary.get('backends_covered') or [])}")
            if self.benchmark_summary.get("categories_covered"):
                lines.append(f"- Categories: {', '.join(self.benchmark_summary.get('categories_covered') or [])}")
            if self.benchmark_summary.get("first_run"):
                lines.append("- No previous benchmark run to compare.")
            elif self.benchmark_summary.get("previous_run_id"):
                lines.append(f"- Compared to: {self.benchmark_summary.get('previous_run_id')}")
                lines.append(f"- Improved: {len(self.benchmark_summary.get('improved_cases') or [])}")
                lines.append(f"- Regressed: {len(self.benchmark_summary.get('regressed_cases') or [])}")
                lines.append(f"- Unchanged: {len(self.benchmark_summary.get('unchanged_cases') or [])}")
                lines.append(f"- New cases: {len(self.benchmark_summary.get('new_cases') or [])}")
                lines.append(f"- Removed cases: {len(self.benchmark_summary.get('removed_cases') or [])}")
                if self.benchmark_summary.get("regressed_cases"):
                    lines.append(f"- Regressed cases: {', '.join(self.benchmark_summary.get('regressed_cases') or [])}")
        else:
            lines.append("- No benchmark summary")
        lines.append("")
        lines.append("## Execution")
        if self.execution:
            lines.append(f"- Executed action: {self.execution.action_name or 'N/A'}")
            lines.append(f"- Status: {self.execution.status or 'N/A'}")
            lines.append(f"- Dry run: {self.execution.dry_run if self.execution.dry_run is not None else 'N/A'}")
            lines.append(f"- Command: {self.execution.command or 'N/A'}")
            lines.append(f"- Return code: {self.execution.return_code}")
            lines.append(f"- Output dir: {self.execution.output_dir or 'N/A'}")
            lines.append(f"- Generated model: {self.execution.generated_model_path or 'N/A'}")
            le = self.state.last_execution or {}
            if isinstance(le, dict) and le.get("action_name") == "run_cryodrgn_ai":
                if le.get("stdout_log"):
                    lines.append(f"- stdout log: {le.get('stdout_log')}")
                if le.get("stderr_log"):
                    lines.append(f"- stderr log: {le.get('stderr_log')}")
                if le.get("output_artifacts"):
                    lines.append(f"- Output artifacts: {le.get('output_artifacts')}")
            if self.execution.error_message:
                lines.append(f"- Error: {self.execution.error_message}")
        else:
            lines.append("- No execution recorded")
        lines.append("")
        lines.append("## Action History")
        if self.action_history:
            for item in self.action_history[-10:]:
                lines.append(
                    f"- {item.timestamp} | stage={item.stage} | action={item.action_name} "
                    f"| status={item.execution_status} | reason={item.decision_reason}"
                )
        else:
            lines.append("- No action history recorded")
        lines.append("")
        lines.append("## Stage Transitions")
        lines.append(f"- Run ID: {self.run_id or 'N/A'}")
        lines.append(f"- {self.stage_transition_summary or 'N/A'}")
        lines.append("")
        lines.append("## Branch History")
        if self.branch_summaries:
            for branch, summary in self.branch_summaries.items():
                lines.append(f"- Branch: {branch} -> {summary}")
        else:
            lines.append("- No branch summary available")
        lines.append("")
        lines.append("## Human Review")
        lines.append(f"- Required: {self.human_review_required}")
        if self.human_review_reasons:
            for reason in self.human_review_reasons:
                lines.append(f"- Reason: {reason}")
        lines.append("")
        lines.append("## Structure Routing")
        if self.structure_routing:
            lines.append(f"- Triggered: {self.structure_routing.triggered}")
            lines.append(f"- Sequence path: {self.structure_routing.sequence_path or 'N/A'}")
            lines.append(f"- Chosen action: {self.structure_routing.selected_tool or 'N/A'}")
            lines.append(f"- Routing reason: {self.structure_routing.rationale or 'N/A'}")
            if self.structure_routing.alternatives_considered:
                lines.append(
                    "- Alternatives: " + ", ".join(self.structure_routing.alternatives_considered)
                )
        else:
            lines.append("- Not triggered")
        lines.append("")
        return "\n".join(lines)
