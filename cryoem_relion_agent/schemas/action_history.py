"""Schema for one action-history record."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ActionHistoryEntry(BaseModel):
    """One decision/execution trace entry for auditability."""

    entry_id: str = Field(default_factory=lambda: uuid4().hex)
    run_id: Optional[str] = None
    parent_decision_id: Optional[str] = None
    previous_entry_id: Optional[str] = None
    branch_name: Optional[str] = "main"
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    stage: str = "unknown"
    action_name: str
    decision_reason: str = ""
    qc_snapshot: Dict[str, Any] = Field(default_factory=dict)
    execution_status: str = "decided"
    output_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dictionary."""
        return self.model_dump(mode="json", exclude_none=False)

