"""Structured result for execution-layer actions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ExecutionResult(BaseModel):
    """Unified result of running one action command/scheme."""

    action_name: str = Field(..., description="Action enum value string.")
    command: Optional[str] = Field(default=None, description="Human-readable shell-style command.")
    command_argv: Optional[List[str]] = Field(default=None, description="Argv passed to subprocess.")
    dry_run: bool = False
    status: str = Field(
        default="unknown",
        description="success | failed | dry_run | skipped | error_validation | not_implemented | error",
    )
    return_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    output_dir: Optional[str] = None
    generated_paths: List[str] = Field(default_factory=list)
    generated_model_path: Optional[str] = None
    log_path: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """Whether execution should be treated as successful."""
        return self.status in {"success", "dry_run", "skipped"}

    def to_serializable_dict(self) -> Dict[str, Any]:
        """JSON-friendly dict for SampleState.last_execution."""
        return self.model_dump(mode="json", exclude_none=False)

    def to_dict(self) -> Dict[str, Any]:
        """Compatibility alias for callers expecting ``to_dict``."""
        return self.to_serializable_dict()
