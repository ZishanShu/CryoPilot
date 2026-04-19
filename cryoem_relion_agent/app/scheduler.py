"""Top-level scheduler orchestration for cryoem_relion_agent."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from agent.backend_registry import model_action_to_backend, model_building_backend_to_action_name
from agent.backend_registry import get_backend_capability_matrix
from agent.model_building_preflight import preflight_model_building
from agent.heterogeneity_preflight import enrich_and_preflight_cryodrgn_params
from agent.picking_preflight import enrich_and_preflight_topaz_params
from agent.history_manager import ActionHistoryManager
from agent.policy_engine import PolicyEngine
from agent.reporter import Reporter
from agent.state_manager import StateManager
from agent.structure_router import ModelingContext, StructureRouter
from agent.benchmark_harness import BenchmarkHarness, default_smoke_cases
from agent.benchmark_persistence import load_latest_benchmark, persist_benchmark_run
from relion.qc_extractor import QCExtractor
from relion.relion_actions import RelionActionExecutor
from schemas.action import Action, ActionName
from schemas.action_history import ActionHistoryEntry
from schemas.report import PipelineReport

logger = logging.getLogger(__name__)


class Scheduler:
    """Coordinate scan -> decision -> execution -> reporting."""

    def __init__(self, project_root: Path, base_dir: Optional[Path] = None) -> None:
        self.project_root = project_root
        self.base_dir = base_dir or Path(__file__).resolve().parents[1]
        self.config_dir = self.base_dir / "configs"
        self.state_file = self.project_root / ".cryoem_agent" / "sample_state.json"
        self.history_file = self.project_root / ".cryoem_agent" / "action_history.jsonl"

        self.qc_extractor = QCExtractor()
        self.policy_engine = PolicyEngine(self.config_dir / "policy_rules.yaml")
        self.structure_router = StructureRouter(self.config_dir / "router_rules.yaml")
        self.action_executor = RelionActionExecutor(self.config_dir / "schemes.yaml")
        self.state_manager = StateManager(self.state_file)
        self.history_manager = ActionHistoryManager(self.history_file)
        self.reporter = Reporter()
        self.policy_engine.set_history_manager(self.history_manager)
        self._active_run_id: Optional[str] = None
        self._active_previous_entry_id: Optional[str] = None
        self._last_benchmark_summary: Optional[Dict[str, Any]] = None

    def _begin_run_trace(self) -> str:
        run_id = uuid4().hex
        self._active_run_id = run_id
        self._active_previous_entry_id = None
        self.policy_engine.set_trace_context(run_id=run_id, previous_entry_id=None, branch_name="main")
        return run_id

    @staticmethod
    def _infer_branch_name(action_name: str, action_params: Optional[dict] = None) -> str:
        if action_params and str(action_params.get("branch_name", "")).strip():
            return str(action_params.get("branch_name")).strip()
        if action_name in {"route_structure", "delegate_structure_router"}:
            return "structure_router"
        if action_name in {"request_human_review"}:
            return "heterogeneity"
        if action_name in {"run_cryodrgn"}:
            return "heterogeneity"
        if action_name in {"run_modelangelo"}:
            return "modelangelo"
        if action_name in {"run_emodelx"}:
            return "emodelx"
        if action_name in {"run_mica"}:
            return "mica"
        if "router" in action_name:
            return "structure_router"
        return "main"

    def _record_action_history(self, entry: ActionHistoryEntry) -> Optional[str]:
        entry_id = self.history_manager.add_action_to_history(entry)
        if entry_id:
            self._active_previous_entry_id = entry_id
        return entry_id

    @staticmethod
    def _backend_tags(action: Action) -> str:
        if not isinstance(action.parameters, dict):
            return ""
        tags: list[str] = []
        picking_backend = str(action.parameters.get("picking_backend", "")).strip()
        if picking_backend:
            tags.append(f"picking_backend={picking_backend}")
        pk_av = str(action.parameters.get("picking_availability", "")).strip()
        if pk_av:
            tags.append(f"picking_availability={pk_av}")
        pk_mode = str(action.parameters.get("picking_execution_mode", "")).strip()
        if pk_mode:
            tags.append(f"picking_execution_mode={pk_mode}")
        pk_br = str(action.parameters.get("picking_blocked_reason", "")).strip()
        if pk_br:
            short = pk_br if len(pk_br) <= 160 else pk_br[:157] + "..."
            tags.append(f"picking_blocked_reason={short}")
        if isinstance(action.parameters.get("topaz_preflight"), dict):
            ex = (action.parameters["topaz_preflight"] or {}).get("executable_path")
            if ex:
                tags.append(f"topaz_executable={str(ex)[:120]}")
        if isinstance(action.parameters.get("picking_selection_decision"), dict):
            sd = action.parameters["picking_selection_decision"] or {}
            if sd.get("selection_reason"):
                tags.append(f"picking_selection_reason={str(sd.get('selection_reason'))[:80]}")
        heterogeneity_backend = str(action.parameters.get("heterogeneity_backend", "")).strip()
        if heterogeneity_backend:
            tags.append(f"heterogeneity_backend={heterogeneity_backend}")
        het_av = str(action.parameters.get("heterogeneity_availability", "")).strip()
        if het_av:
            tags.append(f"heterogeneity_availability={het_av}")
        het_mode = str(action.parameters.get("heterogeneity_execution_mode", "")).strip()
        if het_mode:
            tags.append(f"heterogeneity_execution_mode={het_mode}")
        het_br = str(action.parameters.get("heterogeneity_blocked_reason", "")).strip()
        if het_br:
            short = het_br if len(het_br) <= 160 else het_br[:157] + "..."
            tags.append(f"heterogeneity_blocked_reason={short}")
        if isinstance(action.parameters.get("cryodrgn_preflight"), dict):
            ex = (action.parameters["cryodrgn_preflight"] or {}).get("executable_path")
            if ex:
                tags.append(f"cryodrgn_executable={str(ex)[:120]}")
        if isinstance(action.parameters.get("heterogeneity_selection_decision"), dict):
            sd = action.parameters["heterogeneity_selection_decision"] or {}
            if sd.get("selection_reason"):
                tags.append(f"heterogeneity_selection_reason={str(sd.get('selection_reason'))[:80]}")
        model_building_backend = str(action.parameters.get("model_building_backend", "")).strip()
        if model_building_backend:
            tags.append(f"model_building_backend={model_building_backend}")
        mb_avail = str(action.parameters.get("model_building_availability", "")).strip()
        if mb_avail:
            tags.append(f"model_building_availability={mb_avail}")
        mb_mode = str(action.parameters.get("model_building_execution_mode", "")).strip()
        if mb_mode:
            tags.append(f"model_building_execution_mode={mb_mode}")
        mb_br = str(action.parameters.get("model_building_blocked_reason", "")).strip()
        if mb_br:
            short = mb_br if len(mb_br) <= 160 else mb_br[:157] + "..."
            tags.append(f"model_building_blocked_reason={short}")
        if isinstance(action.parameters.get("model_building_preflight"), dict):
            mpf = action.parameters.get("model_building_preflight") or {}
            er = mpf.get("repo_path")
            ep = mpf.get("python_path")
            eo = mpf.get("output_path")
            af3 = mpf.get("af3_dir")
            if er:
                key = "mica_repo" if model_building_backend == "mica" else "emodelx_repo"
                tags.append(f"{key}={str(er)[:120]}")
            if ep:
                key = "mica_python" if model_building_backend == "mica" else "emodelx_python"
                tags.append(f"{key}={str(ep)[:120]}")
            if eo:
                key = "mica_output" if model_building_backend == "mica" else "emodelx_output"
                tags.append(f"{key}={str(eo)[:120]}")
            if af3 and model_building_backend == "mica":
                tags.append(f"mica_af3_dir={str(af3)[:120]}")
        if isinstance(action.parameters.get("model_building_selection_decision"), dict):
            sd = action.parameters["model_building_selection_decision"] or {}
            if sd.get("selection_reason"):
                tags.append(f"model_selection_reason={str(sd.get('selection_reason'))[:80]}")
        return f" ({', '.join(tags)})" if tags else ""

    def _selection_availability_snapshot(self, params: dict, category: str) -> Dict[str, Dict[str, Any]]:
        sec = self.policy_engine.backend_registry.section(category)
        allowed = [str(x).strip().lower() for x in sec.get("allowed_backends", []) if str(x).strip()]
        out: Dict[str, Dict[str, Any]] = {}
        if category == "picking":
            backend = str(params.get("picking_backend") or "").strip().lower()
            av = str(params.get("picking_availability") or "").strip().lower()
            if backend:
                out[backend] = {"available": av == "available" if av else True}
        elif category == "heterogeneity":
            backend = str(params.get("heterogeneity_backend") or "").strip().lower()
            av = str(params.get("heterogeneity_availability") or "").strip().lower()
            if backend:
                out[backend] = {"available": av == "available" if av else True}
        elif category == "model_building":
            backend = str(params.get("model_building_backend") or "").strip().lower()
            av = str(params.get("model_building_availability") or "").strip().lower()
            if backend:
                out[backend] = {"available": av == "available" if av else True}
        for b in allowed:
            out.setdefault(b, {})
        return out

    def scan_project(
        self,
        sequence_override: Optional[Path] = None,
        *,
        persist_state: bool = True,
    ):
        """Scan project, merge sequence path, optionally persist ``sample_state.json``."""
        state, qc = self.qc_extractor.extract(self.project_root)
        persisted = self.state_manager.load_state()

        if sequence_override is not None:
            state.sequence_path = str(sequence_override.resolve())
        elif persisted is not None and persisted.sequence_path:
            state.sequence_path = persisted.sequence_path

        if persist_state:
            self.state_manager.save_state(state)
        return state, qc

    def _enrich_autopick_action(self, state, action: Action) -> Action:
        """Topaz: merge defaults + structured preflight before execution/report."""
        if action.name != ActionName.RUN_AUTOPICK:
            return action
        if str(action.parameters.get("picking_backend", "")).strip().lower() != "topaz":
            return action
        merged = enrich_and_preflight_topaz_params(
            dict(action.parameters),
            project_root=Path(state.project_path),
            topaz_config_path=self.config_dir / "topaz.yaml",
        )
        decision = self.policy_engine.select_backend_for_category(
            category="picking",
            state=state,
            context={"micrographs_star_path": merged.get("micrographs_star_path")},
            preferred_backend=str(action.parameters.get("picking_backend") or "").strip() or None,
            availability_snapshot=self._selection_availability_snapshot(merged, "picking"),
        )
        merged["picking_selection_decision"] = decision
        return action.model_copy(update={"parameters": merged})

    def _enrich_cryodrgn_action(self, state, action: Action) -> Action:
        """cryoDRGN: merge defaults + structured preflight before execution/report."""
        if action.name != ActionName.RUN_CRYODRGN:
            return action
        if str(action.parameters.get("heterogeneity_backend", "")).strip().lower() != "cryodrgn":
            return action
        merged = enrich_and_preflight_cryodrgn_params(
            dict(action.parameters),
            project_root=Path(state.project_path),
            cryodrgn_config_path=self.config_dir / "cryodrgn.yaml",
        )
        decision = self.policy_engine.select_backend_for_category(
            category="heterogeneity",
            state=state,
            context={"particles_star_path": merged.get("particles_star_path")},
            preferred_backend=str(action.parameters.get("heterogeneity_backend") or "").strip() or None,
            availability_snapshot=self._selection_availability_snapshot(merged, "heterogeneity"),
        )
        merged["heterogeneity_selection_decision"] = decision
        return action.model_copy(update={"parameters": merged})

    def _enrich_model_building_action(self, state, action: Action) -> Action:
        """Model-building: run structured preflight for the resolved backend."""
        if action.name not in {ActionName.RUN_MODELANGELO, ActionName.RUN_EMODELX, ActionName.RUN_MICA}:
            return action
        backend_id = str(action.parameters.get("model_building_backend") or model_action_to_backend(action.name.value)).strip().lower()
        merged = dict(action.parameters)
        if action.name == ActionName.RUN_EMODELX:
            merged.setdefault("chain_count", merged.get("chain_count") or 1)
            merged.setdefault(
                "output_dir",
                str(Path(state.project_path) / "ModelBuilding" / "EModelX" / "job_agent"),
            )
            cfg_path = self.config_dir / "emodelx.yaml"
            pf = preflight_model_building(backend_id, merged, emodelx_config_path=cfg_path)
            merged["emodelx_preflight"] = pf
        elif action.name == ActionName.RUN_MICA:
            merged.setdefault("chain_count", merged.get("chain_count") or 1)
            merged.setdefault(
                "output_dir",
                str(Path(state.project_path) / "ModelBuilding" / "MICA" / "job_agent"),
            )
            merged.setdefault("protocol", "AF3_struct")
            cfg_path = self.config_dir / "mica.yaml"
            default_repo = self.base_dir.parent / "MICA"
            if default_repo.is_dir():
                merged.setdefault("mica_repo_dir", str(default_repo))
            pf = preflight_model_building(backend_id, merged, mica_config_path=cfg_path)
            merged["mica_preflight"] = pf
        else:
            pf = preflight_model_building(backend_id, merged)
        merged["model_building_preflight"] = pf
        merged["model_building_backend"] = backend_id
        merged["model_building_availability"] = "available" if pf.get("available") else "unavailable"
        merged["model_building_execution_mode"] = str(pf.get("execution_mode") or "unavailable")
        merged["model_building_blocked_reason"] = pf.get("blocked_reason") or ""
        merged["model_building_missing_inputs"] = list(pf.get("missing_inputs") or [])
        merged["model_building_missing_dependencies"] = list(pf.get("missing_dependencies") or [])
        decision = self.policy_engine.select_backend_for_category(
            category="model_building",
            state=state,
            context={"af3_dir": merged.get("af3_dir")},
            preferred_backend=backend_id,
            availability_snapshot=self._selection_availability_snapshot(merged, "model_building"),
        )
        merged["model_building_selection_decision"] = decision
        if action.name == ActionName.RUN_EMODELX:
            if pf.get("repo_path"):
                merged.setdefault("emodelx_repo_dir", pf.get("repo_path"))
            if pf.get("python_path"):
                merged.setdefault("emodelx_python", pf.get("python_path"))
        if action.name == ActionName.RUN_MICA:
            if pf.get("repo_path"):
                merged.setdefault("mica_repo_dir", pf.get("repo_path"))
            if pf.get("python_path"):
                merged.setdefault("mica_python", pf.get("python_path"))
            if pf.get("af3_dir"):
                merged.setdefault("af3_dir", pf.get("af3_dir"))
        return action.model_copy(update={"parameters": merged})

    def _resolve_next_action(
        self,
        state,
        qc,
        modeling_context: Optional[ModelingContext] = None,
        log_history: bool = True,
    ) -> Action:
        """Apply policy; if it delegates structure routing, call ``StructureRouter``."""
        action = self.policy_engine.decide_next_action(state, qc, log_history=log_history)
        if action is not None:
            if action.name == ActionName.RUN_AUTOPICK:
                action = self._enrich_autopick_action(state, action)
            elif action.name == ActionName.RUN_CRYODRGN:
                action = self._enrich_cryodrgn_action(state, action)
            elif action.name in {ActionName.RUN_MODELANGELO, ActionName.RUN_EMODELX, ActionName.RUN_MICA}:
                action = self._enrich_model_building_action(state, action)
            return action

        ctx = modeling_context or ModelingContext(
            known_sequence=bool(state.sequence_path),
            af3_available=False,
            chain_count=None,
        )
        routed = self.structure_router.route(state=state, qc=qc, context=ctx)
        if routed.name not in (
            ActionName.RUN_MODELANGELO,
            ActionName.RUN_EMODELX,
            ActionName.RUN_MICA,
        ):
            return routed

        merged_params = dict(routed.parameters)
        merged_params.setdefault("routing_triggered", True)
        merged_params.setdefault("routing_reason", routed.reason)

        explicit = None
        raw_pref = getattr(ctx, "preferred_model_building_backend", None)
        if raw_pref is not None and str(raw_pref).strip():
            explicit = str(raw_pref).strip().lower()

        mb_from_router = model_action_to_backend(routed.name.value)
        preferred = explicit if explicit else mb_from_router
        mb_resolved, mb_note = self.policy_engine.backend_registry.resolve(
            "model_building",
            preferred,
            state.project_id,
        )
        merged_params["model_building_backend"] = mb_resolved
        if mb_note:
            merged_params["model_building_resolution_note"] = mb_note

        if state.map_path:
            merged_params.setdefault("map_path", state.map_path)
        if state.sequence_path:
            merged_params.setdefault("sequence_path", state.sequence_path)

        action_name_str = model_building_backend_to_action_name(mb_resolved)
        try:
            new_name = ActionName(action_name_str)
        except ValueError:
            new_name = ActionName.RUN_MODELANGELO

        if new_name == ActionName.RUN_EMODELX:
            merged_params.setdefault("chain_count", merged_params.get("chain_count") or 1)
            merged_params.setdefault("output_dir", str(Path(state.project_path) / "ModelBuilding" / "EModelX" / "job_agent"))
            merged_params.setdefault("emodelx_protocol", "temp_free")
            default_repo = self.base_dir.parent / "EModelX"
            if default_repo.is_dir():
                merged_params.setdefault("emodelx_repo_dir", str(default_repo))
        if new_name == ActionName.RUN_MICA:
            merged_params.setdefault("chain_count", merged_params.get("chain_count") or 1)
            merged_params.setdefault("output_dir", str(Path(state.project_path) / "ModelBuilding" / "MICA" / "job_agent"))
            merged_params.setdefault("protocol", "AF3_struct")
            default_repo = self.base_dir.parent / "MICA"
            if default_repo.is_dir():
                merged_params.setdefault("mica_repo_dir", str(default_repo))

        raw_af3 = getattr(ctx, "af3_dir", None)
        if raw_af3 is not None and str(raw_af3).strip():
            merged_params.setdefault("af3_dir", str(raw_af3).strip())

        pf = preflight_model_building(
            mb_resolved,
            merged_params,
            emodelx_config_path=(self.config_dir / "emodelx.yaml") if mb_resolved == "emodelx" else None,
            mica_config_path=(self.config_dir / "mica.yaml") if mb_resolved == "mica" else None,
        )
        merged_params["model_building_preflight"] = pf
        if mb_resolved == "emodelx":
            merged_params["emodelx_preflight"] = pf
            if pf.get("repo_path"):
                merged_params.setdefault("emodelx_repo_dir", pf.get("repo_path"))
            if pf.get("python_path"):
                merged_params.setdefault("emodelx_python", pf.get("python_path"))
        if mb_resolved == "mica":
            merged_params["mica_preflight"] = pf
            if pf.get("repo_path"):
                merged_params.setdefault("mica_repo_dir", pf.get("repo_path"))
            if pf.get("python_path"):
                merged_params.setdefault("mica_python", pf.get("python_path"))
            if pf.get("af3_dir"):
                merged_params.setdefault("af3_dir", pf.get("af3_dir"))
        merged_params["model_building_availability"] = "available" if pf.get("available") else "unavailable"
        merged_params["model_building_execution_mode"] = str(pf.get("execution_mode") or "unavailable")
        merged_params["model_building_blocked_reason"] = pf.get("blocked_reason") or ""
        merged_params["model_building_missing_inputs"] = list(pf.get("missing_inputs") or [])
        merged_params["model_building_missing_dependencies"] = list(pf.get("missing_dependencies") or [])
        decision = self.policy_engine.select_backend_for_category(
            category="model_building",
            state=state,
            context={"af3_dir": merged_params.get("af3_dir")},
            preferred_backend=preferred,
            availability_snapshot=self._selection_availability_snapshot(merged_params, "model_building"),
        )
        merged_params["model_building_selection_decision"] = decision

        priorities = self.structure_router.rules.get("priorities", {})
        new_priority = int(priorities.get(action_name_str, routed.priority))

        reason = routed.reason
        if mb_note:
            reason = f"{routed.reason} (model_building policy: requested {preferred!r}, using {mb_resolved!r})"

        return Action(
            name=new_name,
            reason=reason,
            parameters=merged_params,
            priority=new_priority,
        )

    def decide_next(
        self,
        sequence_override: Optional[Path] = None,
        modeling_context: Optional[ModelingContext] = None,
    ) -> Action:
        """Decide next pipeline action (including structure-tool routing when applicable)."""
        self._begin_run_trace()
        state, qc = self.scan_project(sequence_override=sequence_override)
        action = self._resolve_next_action(state, qc, modeling_context=modeling_context)
        state.last_action = action.name.value
        self.state_manager.save_state(state)
        return action

    def run_next(
        self,
        dry_run: bool = False,
        sequence_override: Optional[Path] = None,
        modeling_context: Optional[ModelingContext] = None,
    ):
        """Decide and execute next action."""
        run_id = self._begin_run_trace()
        state, qc = self.scan_project(sequence_override=sequence_override)
        action = self._resolve_next_action(state, qc, modeling_context=modeling_context)
        decision_entry_id = self.policy_engine.get_last_decision_entry_id()
        # PolicyEngine may have already persisted the decision entry directly.
        # Keep scheduler-local chain pointer in sync so execution.previous_entry_id
        # links to the latest entry in this run.
        if decision_entry_id:
            self._active_previous_entry_id = decision_entry_id
        if not decision_entry_id:
            decision_entry_id = self._record_action_history(
                ActionHistoryEntry(
                    run_id=run_id,
                    previous_entry_id=self._active_previous_entry_id,
                    branch_name=self._infer_branch_name(action.name.value, action.parameters),
                    stage=state.stage,
                    action_name=action.name.value,
                    decision_reason=action.reason,
                    qc_snapshot=qc.model_dump(mode="json"),
                    execution_status="decided",
                    output_summary=f"scheduler decided action{self._backend_tags(action)}",
                )
            )
        route_entry_id: Optional[str] = None
        if (
            decision_entry_id
            and isinstance(action.parameters, dict)
            and bool(action.parameters.get("routing_triggered"))
        ):
            route_entry_id = self._record_action_history(
                ActionHistoryEntry(
                    run_id=run_id,
                    parent_decision_id=decision_entry_id,
                    previous_entry_id=self._active_previous_entry_id,
                    branch_name="structure_router",
                    stage=state.stage,
                    action_name="route_structure",
                    decision_reason=str(action.parameters.get("routing_reason") or action.reason),
                    qc_snapshot=qc.model_dump(mode="json"),
                    execution_status="routed",
                    output_summary=f"selected={action.name.value}",
                )
            )
        result = self.action_executor.execute(
            action=action,
            project_root=self.project_root,
            dry_run=dry_run,
            state=state,
        )
        self._record_action_history(
            ActionHistoryEntry(
                run_id=run_id,
                parent_decision_id=route_entry_id or decision_entry_id,
                previous_entry_id=self._active_previous_entry_id,
                branch_name=self._infer_branch_name(action.name.value, action.parameters),
                stage=state.stage,
                action_name=action.name.value,
                decision_reason=action.reason,
                qc_snapshot=qc.model_dump(mode="json"),
                execution_status=str(result.get("status", "unknown")) if isinstance(result, dict) else "unknown",
                output_summary=(
                    (
                        str(result.get("error_message") or result.get("output_dir") or "")
                        + self._backend_tags(action)
                    )
                    if isinstance(result, dict)
                    else ""
                ),
            )
        )
        state.last_action = action.name.value
        if isinstance(result, dict):
            result.setdefault("run_id", run_id)
            result.setdefault("parent_decision_id", decision_entry_id)
            state.last_execution = result
            if result.get("action_name") == "run_modelangelo":
                note = (
                    f"run_modelangelo: status={result.get('status')} "
                    f"output_dir={result.get('output_dir')}"
                )
                if result.get("generated_model_path"):
                    note += f" model={result.get('generated_model_path')}"
                state.notes.append(note)
        self.state_manager.save_state(state)
        return action, result

    def route_structure(self, context: ModelingContext) -> Action:
        """Route structure modeling action."""
        state, qc = self.scan_project()
        return self.structure_router.route(state=state, qc=qc, context=context)

    def run_benchmarks(self, cases: Optional[list[dict]] = None, *, dry_run: bool = False) -> Dict[str, Any]:
        harness = BenchmarkHarness(executor=self.action_executor, project_root=self.project_root)
        summary = harness.run_cases(cases or default_smoke_cases(), dry_run=dry_run)
        summary["markdown"] = harness.summary_markdown(summary)
        # Persist to .cryoem_agent/benchmarks (latest + history) with capability snapshot.
        persist_benchmark_run(
            project_root=self.project_root,
            summary=summary,
            capability_matrix=get_backend_capability_matrix(),
            dry_run=dry_run,
        )
        latest_bench = load_latest_benchmark(self.project_root)
        if isinstance(latest_bench, dict):
            bs = latest_bench.get("benchmark_summary") or {}
            reg = latest_bench.get("regression_summary") or {}
            prev = latest_bench.get("previous_run_metadata") or {}
            self._last_benchmark_summary = {
                "case_count": bs.get("total_cases"),
                "completed": bs.get("completed"),
                "blocked": bs.get("blocked"),
                "failed": bs.get("failed"),
                "run_id": (latest_bench.get("run_metadata") or {}).get("run_id"),
                "timestamp": (latest_bench.get("run_metadata") or {}).get("timestamp"),
                "backends_covered": bs.get("backends_covered") or [],
                "categories_covered": bs.get("categories_covered") or [],
                "previous_run_id": prev.get("previous_run_id"),
                "first_run": bool(reg.get("first_run")),
                "improved_cases": reg.get("improved_cases") or [],
                "regressed_cases": reg.get("regressed_cases") or [],
                "unchanged_cases": reg.get("unchanged_cases") or [],
                "new_cases": reg.get("new_cases") or [],
                "removed_cases": reg.get("removed_cases") or [],
            }
        else:
            self._last_benchmark_summary = summary
        return summary

    def generate_report(
        self,
        out_json: Optional[Path] = None,
        out_md: Optional[Path] = None,
        sequence_override: Optional[Path] = None,
        modeling_context: Optional[ModelingContext] = None,
        *,
        refresh_state: bool = False,
    ) -> PipelineReport:
        """Generate report and optionally write JSON/Markdown files.

        By default ``sample_state.json`` is not written (read-only reporting). Pass
        ``refresh_state=True`` to persist the scanned state like ``scan-project``.
        """
        state, qc = self.scan_project(
            sequence_override=sequence_override,
            persist_state=refresh_state,
        )
        next_action = self._resolve_next_action(state, qc, modeling_context=modeling_context, log_history=False)
        latest_bench = load_latest_benchmark(self.project_root)
        bench_summary = self._last_benchmark_summary
        if bench_summary is None and isinstance(latest_bench, dict):
            # Derive compact summary from persisted payload when no in-memory summary.
            bs = latest_bench.get("benchmark_summary") or {}
            reg = latest_bench.get("regression_summary") or {}
            prev = latest_bench.get("previous_run_metadata") or {}
            bench_summary = {
                "case_count": bs.get("total_cases"),
                "completed": bs.get("completed"),
                "blocked": bs.get("blocked"),
                "failed": bs.get("failed"),
                "run_id": (latest_bench.get("run_metadata") or {}).get("run_id"),
                "timestamp": (latest_bench.get("run_metadata") or {}).get("timestamp"),
                "backends_covered": bs.get("backends_covered") or [],
                "categories_covered": bs.get("categories_covered") or [],
                "previous_run_id": prev.get("previous_run_id"),
                "first_run": bool(reg.get("first_run")),
                "improved_cases": reg.get("improved_cases") or [],
                "regressed_cases": reg.get("regressed_cases") or [],
                "unchanged_cases": reg.get("unchanged_cases") or [],
                "new_cases": reg.get("new_cases") or [],
                "removed_cases": reg.get("removed_cases") or [],
            }
        report = self.reporter.build_report(
            state=state,
            qc=qc,
            recent_action=None,
            next_action=next_action,
            action_history=self.history_manager.retrieve_action_history(),
            current_run_id=self._active_run_id,
            capability_matrix=get_backend_capability_matrix(),
            benchmark_summary=bench_summary,
        )
        if out_json:
            self.reporter.write_json(report, out_json)
        if out_md:
            self.reporter.write_markdown(report, out_md)
        return report
