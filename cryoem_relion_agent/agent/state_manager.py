"""State persistence and recovery utilities for SampleState."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from schemas.sample_state import SampleState

logger = logging.getLogger(__name__)


class StateManager:
    """Manage on-disk state snapshots for one project."""

    def __init__(self, state_file: Path) -> None:
        self.state_file = state_file

    def load_state(self) -> Optional[SampleState]:
        """Load state from JSON file. Return None when missing/invalid."""
        try:
            if not self.state_file.exists():
                logger.info("State file not found: %s", self.state_file)
                return None

            content = self.state_file.read_text(encoding="utf-8")
            data = json.loads(content)
            state = SampleState.model_validate(data)
            logger.info("State loaded: %s", self.state_file)
            return state
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            logger.error("Failed to load state from %s: %s", self.state_file, exc)
            return None

    def save_state(self, state: SampleState) -> bool:
        """Save state as JSON file. Return success boolean."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            state.touch()
            payload = state.model_dump(mode="json", exclude_none=False)
            self.state_file.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("State saved: %s", self.state_file)
            return True
        except OSError as exc:
            logger.error("Failed to save state to %s: %s", self.state_file, exc)
            return False

    def backup_state(self, state: SampleState, suffix: str = ".bak") -> Optional[Path]:
        """Save a backup copy of state, returning backup path on success."""
        try:
            backup_path = self.state_file.with_suffix(self.state_file.suffix + suffix)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            payload = state.model_dump(mode="json", exclude_none=False)
            backup_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("State backup saved: %s", backup_path)
            return backup_path
        except OSError as exc:
            logger.error("Failed to backup state: %s", exc)
            return None
