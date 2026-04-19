"""Run RELION schemes/commands from YAML config with dry-run support."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml

from schemas.execution_result import ExecutionResult

logger = logging.getLogger(__name__)


class SchemeRunner:
    """Load command templates and run selected scheme safely."""

    def __init__(self, schemes_config: Path) -> None:
        self.schemes_config = schemes_config
        self.scheme_map = self._load_schemes()

    def _load_schemes(self) -> Dict[str, str]:
        default = {
            "run_import_preprocess": "echo 'run import preprocess'",
            "run_picking_extract": "echo 'run picking extract'",
            "run_class2d_cleanup": "echo 'run class2d cleanup'",
            "run_initmodel_3dclass": "echo 'run initmodel 3dclass'",
            "run_refine_ctf_polish": "echo 'run refine ctf polish'",
            "run_postprocess": "echo 'run postprocess'",
            "run_validation": "echo 'run validation'",
        }
        try:
            if not self.schemes_config.exists():
                return default
            data = yaml.safe_load(self.schemes_config.read_text(encoding="utf-8")) or {}
            schemes = data.get("schemes", {})
            return {**default, **schemes}
        except (OSError, yaml.YAMLError) as exc:
            logger.error("Failed to load schemes config: %s", exc)
            return default

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _argv_to_command(argv: Sequence[str]) -> str:
        return " ".join(shlex.quote(str(a)) for a in argv)

    def run_command(
        self,
        action_name: str,
        argv: Sequence[str],
        *,
        dry_run: bool = True,
        cwd: Optional[Path] = None,
        env_overrides: Optional[Mapping[str, str]] = None,
        timeout_sec: Optional[int] = 3600,
        output_dir: Optional[Path] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Run a resolved command argv and return ``ExecutionResult``."""
        started_at = self._now_iso()
        command = self._argv_to_command(argv)
        meta: Dict[str, Any] = dict(metadata or {})

        if dry_run:
            return ExecutionResult(
                action_name=action_name,
                command=command,
                command_argv=[str(a) for a in argv],
                dry_run=True,
                status="dry_run",
                return_code=0,
                stdout=f"[DRY-RUN] Would execute: {command}",
                stderr="",
                output_dir=str(output_dir) if output_dir else None,
                started_at=started_at,
                finished_at=self._now_iso(),
                metadata=meta,
            )

        run_env = os.environ.copy()
        if env_overrides:
            run_env.update({str(k): str(v) for k, v in env_overrides.items()})

        try:
            completed = subprocess.run(
                [str(a) for a in argv],
                cwd=str(cwd) if cwd else None,
                env=run_env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
            status = "success" if completed.returncode == 0 else "failed"
            return ExecutionResult(
                action_name=action_name,
                command=command,
                command_argv=[str(a) for a in argv],
                dry_run=False,
                status=status,
                return_code=completed.returncode,
                stdout=completed.stdout or "",
                stderr=completed.stderr or "",
                output_dir=str(output_dir) if output_dir else None,
                started_at=started_at,
                finished_at=self._now_iso(),
                metadata=meta,
            )
        except FileNotFoundError as exc:
            return ExecutionResult(
                action_name=action_name,
                command=command,
                command_argv=[str(a) for a in argv],
                dry_run=False,
                status="failed",
                return_code=127,
                stdout="",
                stderr=str(exc),
                output_dir=str(output_dir) if output_dir else None,
                error_message=f"command not found: {argv[0]}",
                started_at=started_at,
                finished_at=self._now_iso(),
                metadata=meta,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                action_name=action_name,
                command=command,
                command_argv=[str(a) for a in argv],
                dry_run=False,
                status="failed",
                return_code=None,
                stdout=exc.stdout or "",
                stderr=exc.stderr or str(exc),
                output_dir=str(output_dir) if output_dir else None,
                error_message=f"Command timeout after {timeout_sec}s",
                started_at=started_at,
                finished_at=self._now_iso(),
                metadata=meta,
            )
        except Exception as exc:  # noqa: BLE001
            return ExecutionResult(
                action_name=action_name,
                command=command,
                command_argv=[str(a) for a in argv],
                dry_run=False,
                status="failed",
                return_code=None,
                stdout="",
                stderr=str(exc),
                output_dir=str(output_dir) if output_dir else None,
                error_message=f"Unexpected command failure: {exc}",
                started_at=started_at,
                finished_at=self._now_iso(),
                metadata=meta,
            )

    def run_scheme(
        self,
        action_name: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        dry_run: bool = True,
        cwd: Optional[Path] = None,
        env_overrides: Optional[Mapping[str, str]] = None,
        timeout_sec: Optional[int] = 3600,
        output_dir: Optional[Path] = None,
    ) -> ExecutionResult:
        """Resolve one configured scheme action and run it."""
        params = params or {}
        try:
            template = self.scheme_map.get(action_name)
            if not template:
                return ExecutionResult(
                    action_name=action_name,
                    dry_run=dry_run,
                    status="skipped",
                    error_message=f"Action not configured: {action_name}",
                    started_at=self._now_iso(),
                    finished_at=self._now_iso(),
                    metadata={"runner": "scheme"},
                )
            command = template.format(**params)
            logger.info("Prepared command for %s: %s", action_name, command)
            return self.run_command(
                action_name=action_name,
                argv=shlex.split(command),
                dry_run=dry_run,
                cwd=cwd,
                env_overrides=env_overrides,
                timeout_sec=timeout_sec,
                output_dir=output_dir,
                metadata={"runner": "scheme", "template": template, "params": params},
            )
        except Exception as exc:  # noqa: BLE001
            return ExecutionResult(
                action_name=action_name,
                dry_run=dry_run,
                status="failed",
                error_message=f"Failed to resolve scheme: {exc}",
                stderr=str(exc),
                started_at=self._now_iso(),
                finished_at=self._now_iso(),
                metadata={"runner": "scheme"},
            )

    def run(
        self,
        action_name: str,
        params: Optional[Dict[str, Any]] = None,
        dry_run: bool = True,
        cwd: Optional[Path] = None,
        timeout_sec: int = 3600,
    ) -> Dict[str, Any]:
        """
        Backward-compatible wrapper returning dict payload.

        Existing callers historically expected ``success`` and ``error`` keys.
        """
        er = self.run_scheme(
            action_name=action_name,
            params=params,
            dry_run=dry_run,
            cwd=cwd,
            timeout_sec=timeout_sec,
        )
        out = er.to_serializable_dict()
        out["success"] = er.is_success
        out["error"] = er.error_message
        return out
