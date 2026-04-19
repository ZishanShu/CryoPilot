"""Lightweight unified benchmark harness for backend smoke runs."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

from relion.relion_actions import RelionActionExecutor
from schemas.action import Action, ActionName


def _action_name_for(category: str, backend: str) -> ActionName:
    c = str(category).strip().lower()
    b = str(backend).strip().lower()
    if c == "picking":
        return ActionName.RUN_AUTOPICK
    if c == "heterogeneity":
        return ActionName.RUN_CRYODRGN if b == "cryodrgn" else ActionName.REQUEST_HUMAN_REVIEW
    if c == "model_building":
        if b == "emodelx":
            return ActionName.RUN_EMODELX
        if b == "mica":
            return ActionName.RUN_MICA
        return ActionName.RUN_MODELANGELO
    return ActionName.STOP_PIPELINE


def default_smoke_cases() -> List[Dict[str, Any]]:
    return [
        {
            "benchmark_id": "topaz_smoke_case",
            "category": "picking",
            "backend": "topaz",
            "inputs": {"picking_backend": "topaz"},
            "expected_status": "preflight_blocked",
            "expected_artifacts": [],
            "notes": "No real micrographs in unit tests; should block honestly.",
        },
        {
            "benchmark_id": "cryodrgn_smoke_case",
            "category": "heterogeneity",
            "backend": "cryodrgn",
            "inputs": {"heterogeneity_backend": "cryodrgn"},
            "expected_status": "preflight_blocked",
            "expected_artifacts": [],
            "notes": "No cryodrgn env in unit tests; should block honestly.",
        },
        {
            "benchmark_id": "emodelx_smoke_case",
            "category": "model_building",
            "backend": "emodelx",
            "inputs": {"model_building_backend": "emodelx"},
            "expected_status": "preflight_blocked",
            "expected_artifacts": [],
            "notes": "No repo/weights in unit tests; should block honestly.",
        },
        {
            "benchmark_id": "mica_smoke_case",
            "category": "model_building",
            "backend": "mica",
            "inputs": {"model_building_backend": "mica"},
            "expected_status": "preflight_blocked",
            "expected_artifacts": [],
            "notes": "No repo/weights/af3 in unit tests; should block honestly.",
        },
    ]


class BenchmarkHarness:
    def __init__(self, *, executor: RelionActionExecutor, project_root: Path) -> None:
        self.executor = executor
        self.project_root = project_root

    def run_case(self, case: Dict[str, Any], *, dry_run: bool = False) -> Dict[str, Any]:
        start = time.time()
        category = str(case.get("category") or "")
        backend = str(case.get("backend") or "")
        params = dict(case.get("inputs") or {})
        action = Action(
            name=_action_name_for(category, backend),
            reason=f"benchmark:{case.get('benchmark_id')}",
            parameters=params,
            priority=50,
        )
        result = self.executor.execute(action=action, project_root=self.project_root, dry_run=dry_run, state=None)
        elapsed = time.time() - start
        artifacts: List[Any] = []
        if isinstance(result.get("artifact_paths"), list):
            artifacts = list(result.get("artifact_paths") or [])
        elif isinstance(result.get("generated_paths"), list):
            artifacts = list(result.get("generated_paths") or [])
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        preflight = (
            result.get("topaz_preflight")
            or result.get("cryodrgn_preflight")
            or result.get("emodelx_preflight")
            or result.get("mica_preflight")
            or result.get("model_building_preflight")
        )
        selection = (
            params.get("picking_selection_decision")
            or params.get("heterogeneity_selection_decision")
            or params.get("model_building_selection_decision")
        )
        return {
            "benchmark_id": case.get("benchmark_id"),
            "category": category,
            "backend": backend,
            "status": result.get("status"),
            "execution_mode": result.get("execution_mode"),
            "runtime_seconds": round(float(elapsed), 4),
            "command": result.get("command"),
            "artifacts": artifacts,
            "blocked_reason": result.get("error_message") or result.get("model_building_blocked_reason"),
            "preflight": preflight,
            "selection_decision": selection,
            "notes": case.get("notes"),
            "expected_status": case.get("expected_status"),
            "expected_artifacts": list(case.get("expected_artifacts") or []),
            "generated_paths": list(result.get("generated_paths") or []),
            "topaz_output_path": result.get("topaz_output_path") or metadata.get("topaz_output_path"),
            "stdout": result.get("stdout"),
            "stderr": result.get("stderr"),
            "return_code": result.get("return_code"),
            "extra": dict(case.get("extra") or {}),
        }

    def run_cases(self, cases: List[Dict[str, Any]], *, dry_run: bool = False) -> Dict[str, Any]:
        results = [self.run_case(c, dry_run=dry_run) for c in cases]
        summary = {
            "case_count": len(results),
            "completed": sum(1 for r in results if r.get("status") in {"completed", "success"}),
            "blocked": sum(1 for r in results if r.get("status") == "preflight_blocked"),
            "failed": sum(1 for r in results if r.get("status") in {"failed", "error"}),
            "results": results,
        }
        return summary

    @staticmethod
    def summary_markdown(summary: Dict[str, Any]) -> str:
        lines = [
            "# Backend Benchmark Summary",
            "",
            f"- Cases: {summary.get('case_count', 0)}",
            f"- Completed: {summary.get('completed', 0)}",
            f"- Blocked: {summary.get('blocked', 0)}",
            f"- Failed: {summary.get('failed', 0)}",
            "",
            "## Results",
        ]
        for item in summary.get("results", []):
            lines.append(
                f"- {item.get('benchmark_id')}: backend={item.get('backend')} "
                f"status={item.get('status')} mode={item.get('execution_mode')}"
            )
        lines.append("")
        return "\n".join(lines)

