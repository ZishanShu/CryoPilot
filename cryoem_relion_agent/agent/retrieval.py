"""Simple retrieval helper over case memory for similar historical projects."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from memory.case_store import CaseStore

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Retrieve similar cases from sqlite case memory."""

    def __init__(self, case_store: CaseStore) -> None:
        self.case_store = case_store

    def retrieve_similar_cases(
        self,
        sample_type: Optional[str] = None,
        resolution_upper: Optional[float] = None,
        orientation_warning: Optional[bool] = None,
        heterogeneity_warning: Optional[bool] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Query similar historical cases using simple filters."""
        try:
            filters: Dict[str, Any] = {
                "sample_type": sample_type,
                "resolution_upper": resolution_upper,
                "orientation_warning": orientation_warning,
                "heterogeneity_warning": heterogeneity_warning,
            }
            filters = {k: v for k, v in filters.items() if v is not None}
            return self.case_store.query_similar(filters=filters, top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to retrieve similar cases: %s", exc)
            return []
