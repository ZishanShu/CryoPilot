"""Action dispatcher for RELION pipeline commands and structure builders."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from agent.heterogeneity_preflight import enrich_and_preflight_cryodrgn_params
from agent.model_building_preflight import preflight_model_building
from agent.picking_preflight import enrich_and_preflight_topaz_params
from backends.reconstruction.cryodrgn_ai_backend import run as run_cryodrgn_ai_backend
from backends.reconstruction.cryodrgn_ai_backend import update_state as update_cryodrgn_ai_state
from relion.cryodrgn_runner import run_cryodrgn_pipeline
from relion.emodelx_runner import run_emodelx_pipeline
from relion.mica_runner import run_mica_pipeline
from relion.model_angelo_runner import run_model_angelo
from relion.scheme_runner import SchemeRunner
from relion.topaz_runner import run_topaz_pick
from schemas.action import Action
from schemas.execution_result import ExecutionResult
from schemas.sample_state import SampleState

logger = logging.getLogger(__name__)


def _model_building_preflight_blocked(action: Action) -> bool:
    return (
        isinstance(action.parameters, dict)
        and str(action.parameters.get("model_building_availability", "")).strip().lower() == "unavailable"
    )


def _model_building_preflight_runtime_only(action: Action) -> bool:
    if not isinstance(action.parameters, dict):
        return False
    missing_inputs = list(action.parameters.get("model_building_missing_inputs") or [])
    missing_deps = list(action.parameters.get("model_building_missing_dependencies") or [])
    return bool(missing_deps) and not missing_inputs


def _picking_topaz_preflight_blocked(action: Action) -> bool:
    if str(action.parameters.get("picking_backend", "")).strip().lower() != "topaz":
        return False
    return str(action.parameters.get("picking_availability", "")).strip().lower() == "unavailable"


def _heterogeneity_cryodrgn_preflight_blocked(action: Action) -> bool:
    if str(action.parameters.get("heterogeneity_backend", "")).strip().lower() != "cryodrgn":
        return False
    return str(action.parameters.get("heterogeneity_availability", "")).strip().lower() == "unavailable"


def _merge_picking_preflight_fields(out: Dict[str, Any], action: Action) -> None:
    if not isinstance(action.parameters, dict):
        return
    pf = action.parameters.get("topaz_preflight")
    if isinstance(pf, dict):
        out["topaz_preflight"] = pf
    for k in (
        "picking_backend",
        "picking_availability",
        "picking_execution_mode",
        "picking_blocked_reason",
        "picking_missing_inputs",
        "picking_missing_input_paths",
        "picking_missing_dependencies",
    ):
        if k in action.parameters and action.parameters[k] not in (None, "", []):
            out[k] = action.parameters[k]


def _merge_heterogeneity_preflight_fields(out: Dict[str, Any], action: Action) -> None:
    if not isinstance(action.parameters, dict):
        return
    pf = action.parameters.get("cryodrgn_preflight")
    if isinstance(pf, dict):
        out["cryodrgn_preflight"] = pf
    for k in (
        "heterogeneity_backend",
        "heterogeneity_availability",
        "heterogeneity_execution_mode",
        "heterogeneity_blocked_reason",
        "heterogeneity_missing_inputs",
        "heterogeneity_missing_dependencies",
    ):
        if k in action.parameters and action.parameters[k] not in (None, "", []):
            out[k] = action.parameters[k]


def _merge_model_building_preflight_fields(out: Dict[str, Any], action: Action) -> None:
    if not isinstance(action.parameters, dict):
        return
    pf = action.parameters.get("model_building_preflight")
    if isinstance(pf, dict):
        out["model_building_preflight"] = pf
    epf = action.parameters.get("emodelx_preflight")
    if isinstance(epf, dict):
        out["emodelx_preflight"] = epf
    mpf = action.parameters.get("mica_preflight")
    if isinstance(mpf, dict):
        out["mica_preflight"] = mpf
    for k in (
        "model_building_backend",
        "model_building_availability",
        "model_building_execution_mode",
        "model_building_blocked_reason",
        "model_building_missing_inputs",
        "model_building_missing_dependencies",
    ):
        if k in action.parameters and action.parameters[k] not in (None, "", []):
            out[k] = action.parameters[k]


def _run_relion_cli_action(
    *,
    action_name: str,
    runner: SchemeRunner,
    command: Sequence[str],
    project_dir: Path,
    output_dir: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    timeout_sec: int = 3600,
    dry_run: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> ExecutionResult:
    """Shared thin wrapper for RELION external command actions."""
    return runner.run_command(
        action_name=action_name,
        argv=command,
        dry_run=dry_run,
        cwd=project_dir,
        env_overrides=env,
        timeout_sec=timeout_sec,
        output_dir=output_dir,
        metadata=metadata or {},
    )


def run_autopick_action(
    *,
    runner: SchemeRunner,
    project_dir: Path,
    output_dir: Optional[Path] = None,
    extra_args: Optional[Sequence[str]] = None,
    env: Optional[Mapping[str, str]] = None,
    timeout_sec: int = 3600,
    dry_run: bool = True,
    backend: str = "relion_log",
) -> ExecutionResult:
    backend_norm = str(backend or "relion_log").strip().lower()
    if backend_norm == "topaz":
        cmd = ["topaz", "predict", "--model", "resnet16"]
        if output_dir:
            cmd += ["--output", str(output_dir / "topaz_particles.star")]
    elif backend_norm == "cryolo":
        cmd = ["cryolo_predict.py", "-w", "general_model.h5"]
        if output_dir:
            cmd += ["-o", str(output_dir)]
    else:
        backend_norm = "relion_log"
        cmd = ["relion_autopick"]
        if output_dir:
            cmd += ["--o", str(output_dir)]
    if extra_args:
        cmd.extend(str(x) for x in extra_args)
    return _run_relion_cli_action(
        action_name="run_autopick",
        runner=runner,
        command=cmd,
        project_dir=project_dir,
        output_dir=output_dir,
        env=env,
        timeout_sec=timeout_sec,
        dry_run=dry_run,
        metadata={"picking_backend": backend_norm},
    )


def run_extract_action(
    *,
    runner: SchemeRunner,
    project_dir: Path,
    output_dir: Optional[Path] = None,
    extra_args: Optional[Sequence[str]] = None,
    env: Optional[Mapping[str, str]] = None,
    timeout_sec: int = 3600,
    dry_run: bool = True,
) -> ExecutionResult:
    cmd = ["relion_preprocess"]
    if output_dir:
        cmd += ["--o", str(output_dir)]
    if extra_args:
        cmd.extend(str(x) for x in extra_args)
    return _run_relion_cli_action(
        action_name="run_extract",
        runner=runner,
        command=cmd,
        project_dir=project_dir,
        output_dir=output_dir,
        env=env,
        timeout_sec=timeout_sec,
        dry_run=dry_run,
    )


def run_class2d_action(
    *,
    runner: SchemeRunner,
    project_dir: Path,
    output_dir: Optional[Path] = None,
    extra_args: Optional[Sequence[str]] = None,
    env: Optional[Mapping[str, str]] = None,
    timeout_sec: int = 3600,
    dry_run: bool = True,
) -> ExecutionResult:
    cmd = ["relion_refine", "--o", str(output_dir) if output_dir else "Class2D/jobXXX", "--K", "50"]
    if extra_args:
        cmd.extend(str(x) for x in extra_args)
    return _run_relion_cli_action(
        action_name="run_class2d",
        runner=runner,
        command=cmd,
        project_dir=project_dir,
        output_dir=output_dir,
        env=env,
        timeout_sec=timeout_sec,
        dry_run=dry_run,
    )


def run_class3d_or_initmodel_action(
    *,
    runner: SchemeRunner,
    project_dir: Path,
    output_dir: Optional[Path] = None,
    extra_args: Optional[Sequence[str]] = None,
    env: Optional[Mapping[str, str]] = None,
    timeout_sec: int = 3600,
    dry_run: bool = True,
) -> ExecutionResult:
    cmd = ["relion_refine", "--o", str(output_dir) if output_dir else "InitialModel/jobXXX", "--K", "5"]
    if extra_args:
        cmd.extend(str(x) for x in extra_args)
    return _run_relion_cli_action(
        action_name="run_class3d_or_initmodel",
        runner=runner,
        command=cmd,
        project_dir=project_dir,
        output_dir=output_dir,
        env=env,
        timeout_sec=timeout_sec,
        dry_run=dry_run,
    )


def run_init3d_or_class3d_action(
    *,
    runner: SchemeRunner,
    project_dir: Path,
    output_dir: Optional[Path] = None,
    extra_args: Optional[Sequence[str]] = None,
    env: Optional[Mapping[str, str]] = None,
    timeout_sec: int = 3600,
    dry_run: bool = True,
) -> ExecutionResult:
    """Minimal structured wrapper for Init3D/Class3D stage."""
    return run_class3d_or_initmodel_action(
        runner=runner,
        project_dir=project_dir,
        output_dir=output_dir,
        extra_args=extra_args,
        env=env,
        timeout_sec=timeout_sec,
        dry_run=dry_run,
    )


def run_refine3d_action(
    *,
    runner: SchemeRunner,
    project_dir: Path,
    output_dir: Optional[Path] = None,
    extra_args: Optional[Sequence[str]] = None,
    env: Optional[Mapping[str, str]] = None,
    timeout_sec: int = 3600,
    dry_run: bool = True,
) -> ExecutionResult:
    cmd = ["relion_refine", "--o", str(output_dir) if output_dir else "Refine3D/jobXXX"]
    if extra_args:
        cmd.extend(str(x) for x in extra_args)
    return _run_relion_cli_action(
        action_name="run_refine3d",
        runner=runner,
        command=cmd,
        project_dir=project_dir,
        output_dir=output_dir,
        env=env,
        timeout_sec=timeout_sec,
        dry_run=dry_run,
    )


def run_ctf_refine_action(
    *,
    runner: SchemeRunner,
    project_dir: Path,
    output_dir: Optional[Path] = None,
    extra_args: Optional[Sequence[str]] = None,
    env: Optional[Mapping[str, str]] = None,
    timeout_sec: int = 3600,
    dry_run: bool = True,
) -> ExecutionResult:
    cmd = ["relion_ctf_refine"]
    if output_dir:
        cmd += ["--o", str(output_dir)]
    if extra_args:
        cmd.extend(str(x) for x in extra_args)
    return _run_relion_cli_action(
        action_name="run_ctf_refine",
        runner=runner,
        command=cmd,
        project_dir=project_dir,
        output_dir=output_dir,
        env=env,
        timeout_sec=timeout_sec,
        dry_run=dry_run,
    )


def run_polish_action(
    *,
    runner: SchemeRunner,
    project_dir: Path,
    output_dir: Optional[Path] = None,
    extra_args: Optional[Sequence[str]] = None,
    env: Optional[Mapping[str, str]] = None,
    timeout_sec: int = 3600,
    dry_run: bool = True,
) -> ExecutionResult:
    cmd = ["relion_particle_polish"]
    if output_dir:
        cmd += ["--o", str(output_dir)]
    if extra_args:
        cmd.extend(str(x) for x in extra_args)
    return _run_relion_cli_action(
        action_name="run_polish",
        runner=runner,
        command=cmd,
        project_dir=project_dir,
        output_dir=output_dir,
        env=env,
        timeout_sec=timeout_sec,
        dry_run=dry_run,
    )


def run_postprocess_action(
    *,
    runner: SchemeRunner,
    project_dir: Path,
    output_dir: Optional[Path] = None,
    extra_args: Optional[Sequence[str]] = None,
    env: Optional[Mapping[str, str]] = None,
    timeout_sec: int = 3600,
    dry_run: bool = True,
) -> ExecutionResult:
    cmd = ["relion_postprocess"]
    if output_dir:
        cmd += ["--o", str(output_dir)]
    if extra_args:
        cmd.extend(str(x) for x in extra_args)
    return _run_relion_cli_action(
        action_name="run_postprocess",
        runner=runner,
        command=cmd,
        project_dir=project_dir,
        output_dir=output_dir,
        env=env,
        timeout_sec=timeout_sec,
        dry_run=dry_run,
    )


def run_structure_router_action(
    *,
    runner: SchemeRunner,
    project_dir: Path,
    selected_tool: str,
    dry_run: bool = True,
) -> ExecutionResult:
    """Minimal structured wrapper for structure-router selection event."""
    return _run_relion_cli_action(
        action_name="route_structure",
        runner=runner,
        command=["echo", f"route_structure -> {selected_tool}"],
        project_dir=project_dir,
        dry_run=dry_run,
        metadata={"selected_tool": selected_tool, "branch_name": "structure_router"},
    )


def run_validation_action(
    *,
    runner: SchemeRunner,
    project_dir: Path,
    output_dir: Optional[Path] = None,
    dry_run: bool = True,
) -> ExecutionResult:
    """Minimal structured wrapper for validation-only path."""
    cmd = ["echo", "run validation"]
    if output_dir:
        cmd += [str(output_dir)]
    return _run_relion_cli_action(
        action_name="run_validation",
        runner=runner,
        command=cmd,
        project_dir=project_dir,
        output_dir=output_dir,
        dry_run=dry_run,
        metadata={"branch_name": "structure_router"},
    )


def run_cryodrgn_action(
    *,
    runner: SchemeRunner,
    project_dir: Path,
    output_dir: Optional[Path] = None,
    extra_args: Optional[Sequence[str]] = None,
    env: Optional[Mapping[str, str]] = None,
    timeout_sec: int = 3600,
    dry_run: bool = True,
) -> ExecutionResult:
    """Minimal structured wrapper for cryoDRGN heterogeneity branch."""
    cmd = ["cryodrgn", "train_vae"]
    if output_dir:
        cmd += ["-o", str(output_dir)]
    if extra_args:
        cmd.extend(str(x) for x in extra_args)
    return _run_relion_cli_action(
        action_name="run_cryodrgn",
        runner=runner,
        command=cmd,
        project_dir=project_dir,
        output_dir=output_dir,
        env=env,
        timeout_sec=timeout_sec,
        dry_run=dry_run,
        metadata={"heterogeneity_backend": "cryodrgn", "branch_name": "heterogeneity"},
    )


class RelionActionExecutor:
    """Execute policy-decided actions via SchemeRunner or specialized runners."""

    def __init__(self, schemes_yaml: Path) -> None:
        self.runner = SchemeRunner(schemes_yaml)
        self.model_builders_yaml = schemes_yaml.parent / "model_builders.yaml"
        self.topaz_config_path = schemes_yaml.parent / "topaz.yaml"
        self.cryodrgn_config_path = schemes_yaml.parent / "cryodrgn.yaml"
        self.mica_config_path = schemes_yaml.parent / "mica.yaml"

    def execute(
        self,
        action: Action,
        project_root: Path,
        dry_run: bool = True,
        state: Optional[SampleState] = None,
    ) -> Dict[str, Any]:
        """Execute an Action and return structured command result."""
        try:
            name = action.name.value

            if name == "run_modelangelo":
                if _model_building_preflight_blocked(action) and not (
                    dry_run and _model_building_preflight_runtime_only(action)
                ):
                    out: Dict[str, Any] = {
                        "action_name": name,
                        "status": "dry_run" if dry_run else "blocked",
                        "dry_run": dry_run,
                        "success": bool(dry_run),
                        "command": None,
                        "return_code": None,
                        "stdout": "[DRY-RUN] Real run would be blocked by model_building preflight." if dry_run else "",
                        "stderr": "",
                        "error_message": str(
                            (action.parameters or {}).get("model_building_blocked_reason")
                            or "model_building preflight failed"
                        ),
                        "reason": action.reason,
                        "priority": action.priority,
                    }
                    _merge_model_building_preflight_fields(out, action)
                    return out
                out = run_model_angelo(
                    action=action,
                    project_root=project_root,
                    state=state,
                    dry_run=dry_run,
                    model_builders_yaml=self.model_builders_yaml,
                )
                if isinstance(out, dict):
                    _merge_model_building_preflight_fields(out, action)
                return out

            if name == "run_autopick":
                backend = str(action.parameters.get("picking_backend", "relion_log")).strip().lower()
                if backend == "topaz":
                    params = dict(action.parameters or {})
                    if not isinstance(params.get("topaz_preflight"), dict):
                        params = enrich_and_preflight_topaz_params(
                            params,
                            project_root=project_root,
                            topaz_config_path=self.topaz_config_path,
                        )
                        act = action.model_copy(update={"parameters": params})
                    else:
                        act = action
                    if _picking_topaz_preflight_blocked(act):
                        out_pb: Dict[str, Any] = {
                            "action_name": name,
                            "status": "preflight_blocked",
                            "dry_run": dry_run,
                            "success": False,
                            "command": None,
                            "return_code": None,
                            "stdout": "",
                            "stderr": "",
                            "error_message": str(
                                (act.parameters or {}).get("picking_blocked_reason") or "topaz preflight failed"
                            ),
                            "reason": action.reason,
                            "priority": action.priority,
                            "execution_mode": "unavailable",
                        }
                        _merge_picking_preflight_fields(out_pb, act)
                        return out_pb
                    pf = act.parameters.get("topaz_preflight")
                    if not isinstance(pf, dict):
                        out_pb = {
                            "action_name": name,
                            "status": "preflight_blocked",
                            "dry_run": dry_run,
                            "success": False,
                            "error_message": "topaz_preflight missing",
                            "reason": action.reason,
                            "priority": action.priority,
                        }
                        _merge_picking_preflight_fields(out_pb, act)
                        return out_pb
                    er = run_topaz_pick(
                        params=dict(act.parameters),
                        project_dir=project_root,
                        topaz_config_path=self.topaz_config_path,
                        preflight=pf,
                        dry_run=dry_run,
                    )
                    out = er.to_serializable_dict()
                    out["success"] = er.is_success
                    _merge_picking_preflight_fields(out, act)
                    md = out.get("metadata") if isinstance(out.get("metadata"), dict) else {}
                    for k in ("topaz_command", "topaz_output_path", "topaz_executable", "execution_mode"):
                        if md.get(k):
                            out[k] = md[k]
                    extra = out.get("extra") if isinstance(out.get("extra"), dict) else {}
                    for k in ("topaz_radius_used", "model_source", "command_mode"):
                        if md.get(k) is not None:
                            extra[k] = md.get(k)
                    if extra:
                        out["extra"] = extra
                    out["picking_backend"] = "topaz"
                    if dry_run:
                        out["execution_mode"] = out.get("execution_mode") or "dry_run"
                    elif out.get("status") == "success":
                        out["execution_mode"] = "real"
                    return out
                return run_autopick_action(
                    runner=self.runner,
                    project_dir=project_root,
                    output_dir=project_root / "AutoPick" / "job_agent",
                    dry_run=dry_run,
                    backend=backend,
                ).to_serializable_dict()

            if name == "run_extract":
                return run_extract_action(
                    runner=self.runner,
                    project_dir=project_root,
                    output_dir=project_root / "Extract" / "job_agent",
                    dry_run=dry_run,
                ).to_serializable_dict()

            if name == "run_class2d":
                return run_class2d_action(
                    runner=self.runner,
                    project_dir=project_root,
                    output_dir=project_root / "Class2D" / "job_agent",
                    dry_run=dry_run,
                ).to_serializable_dict()

            if name in ("run_init3d_or_class3d", "run_initmodel_3dclass"):
                return run_init3d_or_class3d_action(
                    runner=self.runner,
                    project_dir=project_root,
                    output_dir=project_root / "InitialModel" / "job_agent",
                    dry_run=dry_run,
                ).to_serializable_dict()

            if name in ("run_refine3d", "run_refine_ctf_polish"):
                return run_refine3d_action(
                    runner=self.runner,
                    project_dir=project_root,
                    output_dir=project_root / "Refine3D" / "job_agent",
                    dry_run=dry_run,
                ).to_serializable_dict()

            if name == "run_ctf_refine":
                return run_ctf_refine_action(
                    runner=self.runner,
                    project_dir=project_root,
                    output_dir=project_root / "CtfRefine" / "job_agent",
                    dry_run=dry_run,
                ).to_serializable_dict()

            if name == "run_polish":
                return run_polish_action(
                    runner=self.runner,
                    project_dir=project_root,
                    output_dir=project_root / "Polish" / "job_agent",
                    dry_run=dry_run,
                ).to_serializable_dict()

            if name == "run_postprocess":
                return run_postprocess_action(
                    runner=self.runner,
                    project_dir=project_root,
                    output_dir=project_root / "PostProcess" / "job_agent",
                    dry_run=dry_run,
                ).to_serializable_dict()

            if name == "run_cryodrgn":
                params = dict(action.parameters or {})
                if not isinstance(params.get("cryodrgn_preflight"), dict):
                    params = enrich_and_preflight_cryodrgn_params(
                        params,
                        project_root=project_root,
                        cryodrgn_config_path=self.cryodrgn_config_path,
                    )
                    act = action.model_copy(update={"parameters": params})
                else:
                    act = action
                if _heterogeneity_cryodrgn_preflight_blocked(act):
                    out_cd: Dict[str, Any] = {
                        "action_name": name,
                        "status": "preflight_blocked",
                        "dry_run": dry_run,
                        "success": False,
                        "command": None,
                        "return_code": None,
                        "stdout": "",
                        "stderr": "",
                        "error_message": str(
                            (act.parameters or {}).get("heterogeneity_blocked_reason")
                            or "cryodrgn preflight failed"
                        ),
                        "reason": action.reason,
                        "priority": action.priority,
                        "execution_mode": "unavailable",
                    }
                    _merge_heterogeneity_preflight_fields(out_cd, act)
                    return out_cd
                pf = act.parameters.get("cryodrgn_preflight")
                if not isinstance(pf, dict):
                    out_cd = {
                        "action_name": name,
                        "status": "preflight_blocked",
                        "dry_run": dry_run,
                        "success": False,
                        "command": None,
                        "return_code": None,
                        "stdout": "",
                        "stderr": "",
                        "error_message": "cryodrgn_preflight missing",
                        "reason": action.reason,
                        "priority": action.priority,
                    }
                    _merge_heterogeneity_preflight_fields(out_cd, act)
                    return out_cd
                res = run_cryodrgn_pipeline(
                    params=dict(act.parameters),
                    preflight=pf,
                    project_root=project_root,
                    dry_run=dry_run,
                )
                st = str(res.get("status") or "unknown")
                success = st == "completed" or (dry_run and st == "dry_run")
                out_cd = {
                    "action_name": name,
                    "status": st,
                    "dry_run": dry_run,
                    "success": success,
                    "heterogeneity_backend": "cryodrgn",
                    "execution_mode": res.get("execution_mode") or ("dry_run" if dry_run else "real"),
                    "commands": res.get("commands"),
                    "cryodrgn_steps": res.get("steps"),
                    "output_dir": res.get("output_dir"),
                    "train_outdir": res.get("train_outdir"),
                    "pose_pkl": res.get("pose_pkl"),
                    "ctf_pkl": res.get("ctf_pkl"),
                    "error_message": res.get("error"),
                    "reason": action.reason,
                    "priority": action.priority,
                    "stdout": "",
                    "stderr": "",
                    "return_code": None,
                }
                cmds = res.get("commands")
                if isinstance(cmds, list) and cmds:
                    out_cd["command"] = " | ".join(str(c) for c in cmds)[:4000]
                _merge_heterogeneity_preflight_fields(out_cd, act)
                return out_cd

            if name == "run_cryodrgn_ai":
                params = dict(action.parameters or {})
                run_dir = Path(
                    str(
                        params.get("run_dir")
                        or params.get("output_dir")
                        or project_root / "CryoDRGN_AI" / "job_agent"
                    )
                )
                params.setdefault("execution_mode", "dry_run" if dry_run else "real")
                params.setdefault("project_path", str(project_root))
                params.setdefault("datadir", str(project_root))
                if not params.get("particles_star_or_mrcs") and params.get("particles_star_path"):
                    params["particles_star_or_mrcs"] = params["particles_star_path"]
                if not params.get("particles_star_or_mrcs") and params.get("particles_mrcs_path"):
                    params["particles_star_or_mrcs"] = params["particles_mrcs_path"]
                if not params.get("ctf_metadata") and params.get("ctf_pkl_path"):
                    params["ctf_metadata"] = params["ctf_pkl_path"]
                res = run_cryodrgn_ai_backend(params, run_dir)
                artifacts = res.get("output_artifacts") if isinstance(res.get("output_artifacts"), dict) else {}
                out_ai: Dict[str, Any] = {
                    "action_name": name,
                    "status": res.get("status"),
                    "dry_run": dry_run,
                    "success": bool(res.get("success")),
                    "selected_backend": "cryodrgn_ai",
                    "reconstruction_backend": "cryodrgn_ai",
                    "selection_reason": params.get("selection_reason"),
                    "execution_mode": res.get("execution_mode"),
                    "command_manifest": res.get("command_manifest") or artifacts.get("command_manifest"),
                    "commands": res.get("commands"),
                    "command": " | ".join(
                        " ".join(str(x) for x in (c.get("argv") or []))
                        for c in (res.get("commands") or [])
                        if isinstance(c, dict)
                    )[:4000],
                    "output_dir": artifacts.get("run_dir") or str(run_dir),
                    "output_artifacts": artifacts,
                    "stdout_log": res.get("stdout_log") or artifacts.get("stdout_log"),
                    "stderr_log": res.get("stderr_log") or artifacts.get("stderr_log"),
                    "return_code": res.get("return_code"),
                    "blocked_reason": res.get("blocked_reason"),
                    "failed_reason": res.get("failed_reason"),
                    "error_message": res.get("blocked_reason") or res.get("failed_reason"),
                    "cryodrgn_ai_preflight": res.get("preflight"),
                    "recommended_downstream_action": (
                        "sample representative latent volumes and route candidate maps to model-building backends"
                    ),
                    "reason": action.reason,
                    "priority": action.priority,
                }
                if state is not None:
                    update_cryodrgn_ai_state(state, out_ai)
                return out_ai

            if name == "request_human_review":
                params = action.parameters or {}
                branch = str(params.get("branch_name", "heterogeneity")).strip() or "heterogeneity"
                if branch == "picking_qc":
                    metadata = {
                        "branch_name": "picking_qc",
                        "picking_backend": str(params.get("picking_backend", "topaz")),
                    }
                else:
                    metadata = {
                        "heterogeneity_backend": str(
                            params.get("heterogeneity_backend", "human_review")
                        ),
                        "branch_name": branch,
                    }
                er = _run_relion_cli_action(
                    action_name="request_human_review",
                    runner=self.runner,
                    command=["echo", f"[DRY-RUN] Human review requested for {branch} branch"],
                    project_dir=project_root,
                    dry_run=dry_run,
                    metadata=metadata,
                )
                out = er.to_serializable_dict()
                out["reason"] = action.reason
                out["priority"] = action.priority
                return out

            if name in ("run_emodelx", "run_mica"):
                if name == "run_emodelx":
                    params = dict(action.parameters or {})
                    if not isinstance(params.get("model_building_preflight"), dict):
                        pf = preflight_model_building(
                            "emodelx",
                            params,
                            emodelx_config_path=self.model_builders_yaml.parent / "emodelx.yaml",
                        )
                        params["model_building_preflight"] = pf
                        params["emodelx_preflight"] = pf
                        params["model_building_backend"] = "emodelx"
                        params["model_building_availability"] = "available" if pf.get("available") else "unavailable"
                        params["model_building_execution_mode"] = str(pf.get("execution_mode") or "unavailable")
                        params["model_building_blocked_reason"] = pf.get("blocked_reason") or ""
                        params["model_building_missing_inputs"] = list(pf.get("missing_inputs") or [])
                        params["model_building_missing_dependencies"] = list(pf.get("missing_dependencies") or [])
                        act = action.model_copy(update={"parameters": params})
                    else:
                        act = action
                    if _model_building_preflight_blocked(act) and not dry_run:
                        out = {
                            "action_name": name,
                            "status": "preflight_blocked",
                            "dry_run": dry_run,
                            "success": False,
                            "command": None,
                            "return_code": None,
                            "stdout": "",
                            "stderr": "",
                            "execution_mode": "unavailable",
                            "error_message": str(
                                (act.parameters or {}).get("model_building_blocked_reason")
                                or "model_building preflight failed"
                            ),
                            "reason": action.reason,
                            "priority": action.priority,
                        }
                        _merge_model_building_preflight_fields(out, act)
                        return out
                    pf = act.parameters.get("model_building_preflight")
                    if not isinstance(pf, dict):
                        out = {
                            "action_name": name,
                            "status": "preflight_blocked",
                            "dry_run": dry_run,
                            "success": False,
                            "execution_mode": "unavailable",
                            "error_message": "model_building_preflight missing",
                            "reason": action.reason,
                            "priority": action.priority,
                        }
                        _merge_model_building_preflight_fields(out, act)
                        return out
                    res = run_emodelx_pipeline(params=dict(act.parameters), preflight=pf, dry_run=dry_run)
                    st = str(res.get("status") or "unknown")
                    out = {
                        "action_name": name,
                        "status": st,
                        "dry_run": dry_run,
                        "success": st == "completed" or (dry_run and st == "dry_run"),
                        "model_building_backend": "emodelx",
                        "execution_mode": str(res.get("execution_mode") or ("dry_run" if dry_run else "real")),
                        "command": str(res.get("command") or " ".join(str(x) for x in (res.get("command_argv") or [])))[:4000],
                        "command_argv": res.get("command_argv"),
                        "cwd": res.get("cwd"),
                        "output_dir": res.get("output_dir"),
                        "repo_path": res.get("repo_path"),
                        "python_path": res.get("python_path"),
                        "artifact_paths": res.get("artifact_paths"),
                        "generated_paths": res.get("generated_paths") or res.get("artifact_paths"),
                        "return_code": res.get("return_code"),
                        "stdout": res.get("stdout") or "",
                        "stderr": res.get("stderr") or "",
                        "error_message": res.get("error"),
                        "reason": action.reason,
                        "priority": action.priority,
                    }
                    _merge_model_building_preflight_fields(out, act)
                    return out

                if name == "run_mica":
                    params = dict(action.parameters or {})
                    if not isinstance(params.get("model_building_preflight"), dict):
                        pf = preflight_model_building(
                            "mica",
                            params,
                            mica_config_path=self.mica_config_path,
                        )
                        params["model_building_preflight"] = pf
                        params["mica_preflight"] = pf
                        params["model_building_backend"] = "mica"
                        params["model_building_availability"] = "available" if pf.get("available") else "unavailable"
                        params["model_building_execution_mode"] = str(pf.get("execution_mode") or "unavailable")
                        params["model_building_blocked_reason"] = pf.get("blocked_reason") or ""
                        params["model_building_missing_inputs"] = list(pf.get("missing_inputs") or [])
                        params["model_building_missing_dependencies"] = list(pf.get("missing_dependencies") or [])
                        act = action.model_copy(update={"parameters": params})
                    else:
                        act = action
                    if _model_building_preflight_blocked(act) and not dry_run:
                        out = {
                            "action_name": name,
                            "status": "preflight_blocked",
                            "dry_run": dry_run,
                            "success": False,
                            "command": None,
                            "return_code": None,
                            "stdout": "",
                            "stderr": "",
                            "execution_mode": "unavailable",
                            "error_message": str(
                                (act.parameters or {}).get("model_building_blocked_reason")
                                or "model_building preflight failed"
                            ),
                            "reason": action.reason,
                            "priority": action.priority,
                        }
                        _merge_model_building_preflight_fields(out, act)
                        return out
                    pf = act.parameters.get("model_building_preflight")
                    if not isinstance(pf, dict):
                        out = {
                            "action_name": name,
                            "status": "preflight_blocked",
                            "dry_run": dry_run,
                            "success": False,
                            "execution_mode": "unavailable",
                            "error_message": "model_building_preflight missing",
                            "reason": action.reason,
                            "priority": action.priority,
                        }
                        _merge_model_building_preflight_fields(out, act)
                        return out
                    res = run_mica_pipeline(params=dict(act.parameters), preflight=pf, dry_run=dry_run)
                    st = str(res.get("status") or "unknown")
                    out = {
                        "action_name": name,
                        "status": st,
                        "dry_run": dry_run,
                        "success": st == "completed" or (dry_run and st == "dry_run"),
                        "model_building_backend": "mica",
                        "execution_mode": str(res.get("execution_mode") or ("dry_run" if dry_run else "real")),
                        "command": str(res.get("command") or " ".join(str(x) for x in (res.get("command_argv") or [])))[:4000],
                        "command_argv": res.get("command_argv"),
                        "cwd": res.get("cwd"),
                        "output_dir": res.get("output_dir"),
                        "repo_path": res.get("repo_path"),
                        "python_path": res.get("python_path"),
                        "af3_dir": res.get("af3_dir"),
                        "artifact_paths": res.get("artifact_paths"),
                        "generated_paths": res.get("generated_paths") or res.get("artifact_paths"),
                        "return_code": res.get("return_code"),
                        "stdout": res.get("stdout") or "",
                        "stderr": res.get("stderr") or "",
                        "error_message": res.get("error"),
                        "reason": action.reason,
                        "priority": action.priority,
                    }
                    _merge_model_building_preflight_fields(out, act)
                    return out

                if _model_building_preflight_blocked(action) and not dry_run:
                    out = {
                        "action_name": name,
                        "status": "preflight_blocked",
                        "dry_run": dry_run,
                        "success": False,
                        "command": None,
                        "return_code": None,
                        "stdout": "",
                        "stderr": "",
                        "placeholder": True,
                        "execution_mode": "unavailable",
                        "error_message": str(
                            (action.parameters or {}).get("model_building_blocked_reason")
                            or "model_building preflight failed"
                        ),
                        "reason": action.reason,
                        "priority": action.priority,
                    }
                    _merge_model_building_preflight_fields(out, action)
                    return out
                # First-party integrations are not wired; always emit scheme-based dry-run only (no subprocess).
                er = self.runner.run_scheme(
                    action_name=name,
                    params=dict(action.parameters),
                    dry_run=True,
                    cwd=project_root,
                    timeout_sec=3600,
                )
                out = er.to_serializable_dict()
                out["reason"] = action.reason
                out["priority"] = action.priority
                out["placeholder"] = True
                out["execution_mode"] = "dry_run_placeholder"
                _merge_model_building_preflight_fields(out, action)
                return out

            result = self.runner.run(
                action_name=name,
                params=action.parameters,
                dry_run=dry_run,
                cwd=project_root,
            )
            result["reason"] = action.reason
            result["priority"] = action.priority
            if "status" not in result:
                result["status"] = "success" if result.get("success") else "failed"
            return result
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to execute action %s: %s", action.name.value, exc)
            return {
                "action_name": action.name.value,
                "dry_run": dry_run,
                "command": None,
                "return_code": None,
                "stdout": "",
                "stderr": "",
                "success": False,
                "status": "error",
                "error": str(exc),
                "error_message": str(exc),
                "reason": action.reason,
                "priority": action.priority,
            }
