"""File-based action history manager (JSONL)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from schemas.action_history import ActionHistoryEntry

logger = logging.getLogger(__name__)


class ActionHistoryManager:
    """Append/retrieve action history entries for one project."""

    def __init__(self, history_file: Path) -> None:
        self.history_file = history_file

    def add_action_to_history(self, entry: ActionHistoryEntry) -> Optional[str]:
        """Append one action record as JSONL; return ``entry_id`` on success."""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(entry.to_dict(), ensure_ascii=False)
            with self.history_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            return entry.entry_id
        except OSError as exc:
            logger.error("Failed to append action history %s: %s", self.history_file, exc)
            return None

    def retrieve_action_history(
        self,
        stage: Optional[str] = None,
        action_name: Optional[str] = None,
    ) -> List[ActionHistoryEntry]:
        """Read and filter action history records."""
        entries: List[ActionHistoryEntry] = []
        try:
            if not self.history_file.exists():
                return entries
            for raw in self.history_file.read_text(encoding="utf-8").splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    item = ActionHistoryEntry.model_validate(json.loads(raw))
                except Exception:  # noqa: BLE001
                    continue
                if stage is not None and item.stage != stage:
                    continue
                if action_name is not None and item.action_name != action_name:
                    continue
                entries.append(item)
            return entries
        except OSError as exc:
            logger.error("Failed to read action history %s: %s", self.history_file, exc)
            return entries

    def summarize_history(self) -> str:
        """Return a short textual summary."""
        entries = self.retrieve_action_history()
        if not entries:
            return "No action history available."
        latest = entries[-1]
        return (
            f"entries={len(entries)}, latest={latest.action_name}, "
            f"status={latest.execution_status}, stage={latest.stage}"
        )

    def get_history_by_run(self, run_id: str) -> List[ActionHistoryEntry]:
        """Return all entries for one run id."""
        return [x for x in self.retrieve_action_history() if x.run_id == run_id]

    def get_history_by_branch(self, branch_name: str, run_id: Optional[str] = None) -> List[ActionHistoryEntry]:
        """Return history filtered by branch name (missing branch -> ``main``)."""
        items = self.get_history_by_run(run_id) if run_id else self.retrieve_action_history()
        return [x for x in items if (x.branch_name or "main") == branch_name]

    def summarize_stage_transitions(self, run_id: Optional[str] = None) -> str:
        """Compact stage-transition view based on history ordering."""
        items = self.get_history_by_run(run_id) if run_id else self.retrieve_action_history()
        if not items:
            return "N/A"
        main_items = [x for x in items if (x.branch_name or "main") == "main"]
        source = main_items or items
        stages: List[str] = []
        for item in source:
            s = item.stage or "unknown"
            if not stages or stages[-1] != s:
                stages.append(s)
        return " -> ".join(stages) if stages else "N/A"

    def summarize_branches(self, run_id: Optional[str] = None) -> Dict[str, str]:
        """Compact branch summary, grouped as action arrow chains."""
        items = self.get_history_by_run(run_id) if run_id else self.retrieve_action_history()
        grouped: Dict[str, List[str]] = {}
        for item in items:
            b = item.branch_name or "main"
            grouped.setdefault(b, []).append(item.action_name)
        return {k: " -> ".join(v) for k, v in grouped.items()}

