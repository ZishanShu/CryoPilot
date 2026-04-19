"""SQLite-backed lightweight case memory store."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from memory.schema import CaseRecord

logger = logging.getLogger(__name__)


class CaseStore:
    """CRUD and simple retrieval for historical cryo-EM cases."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def init_db(self) -> bool:
        """Initialize sqlite schema."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cases (
                        case_id TEXT PRIMARY KEY,
                        project_id TEXT NOT NULL,
                        sample_type TEXT NOT NULL,
                        n_micrographs INTEGER NOT NULL,
                        n_particles_after_2d INTEGER NOT NULL,
                        estimated_resolution REAL,
                        orientation_warning INTEGER NOT NULL,
                        heterogeneity_warning INTEGER NOT NULL,
                        chosen_model_builder TEXT,
                        final_status TEXT NOT NULL,
                        notes TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                conn.commit()
            return True
        except sqlite3.Error as exc:
            logger.error("Failed to init DB %s: %s", self.db_path, exc)
            return False

    def insert_case(self, record: CaseRecord) -> bool:
        """Insert or replace a case record."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cases (
                        case_id, project_id, sample_type, n_micrographs, n_particles_after_2d,
                        estimated_resolution, orientation_warning, heterogeneity_warning,
                        chosen_model_builder, final_status, notes, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.case_id,
                        record.project_id,
                        record.sample_type,
                        record.n_micrographs,
                        record.n_particles_after_2d,
                        record.estimated_resolution,
                        int(record.orientation_warning),
                        int(record.heterogeneity_warning),
                        record.chosen_model_builder,
                        record.final_status,
                        record.notes,
                        record.created_at.isoformat(),
                    ),
                )
                conn.commit()
            return True
        except sqlite3.Error as exc:
            logger.error("Failed to insert case %s: %s", record.case_id, exc)
            return False

    def query_similar(self, filters: Dict[str, Any] | None = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query similar cases using simple field filters."""
        filters = filters or {}
        where_clauses: List[str] = []
        params: List[Any] = []
        if "sample_type" in filters:
            where_clauses.append("sample_type = ?")
            params.append(filters["sample_type"])
        if "resolution_upper" in filters:
            where_clauses.append("(estimated_resolution IS NOT NULL AND estimated_resolution <= ?)")
            params.append(float(filters["resolution_upper"]))
        if "orientation_warning" in filters:
            where_clauses.append("orientation_warning = ?")
            params.append(int(bool(filters["orientation_warning"])))
        if "heterogeneity_warning" in filters:
            where_clauses.append("heterogeneity_warning = ?")
            params.append(int(bool(filters["heterogeneity_warning"])))
        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        sql = f"""
            SELECT case_id, project_id, sample_type, n_micrographs, n_particles_after_2d,
                   estimated_resolution, orientation_warning, heterogeneity_warning,
                   chosen_model_builder, final_status, notes, created_at
            FROM cases
            {where_sql}
            ORDER BY
                CASE WHEN estimated_resolution IS NULL THEN 1 ELSE 0 END ASC,
                estimated_resolution ASC,
                n_particles_after_2d DESC,
                created_at DESC
            LIMIT ?
        """
        params.append(max(1, int(top_k)))
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(sql, params).fetchall()
            return [
                {
                    "case_id": row["case_id"],
                    "project_id": row["project_id"],
                    "sample_type": row["sample_type"],
                    "n_micrographs": row["n_micrographs"],
                    "n_particles_after_2d": row["n_particles_after_2d"],
                    "estimated_resolution": row["estimated_resolution"],
                    "orientation_warning": bool(row["orientation_warning"]),
                    "heterogeneity_warning": bool(row["heterogeneity_warning"]),
                    "chosen_model_builder": row["chosen_model_builder"],
                    "final_status": row["final_status"],
                    "notes": row["notes"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]
        except sqlite3.Error as exc:
            logger.error("Failed to query similar cases: %s", exc)
            return []

    def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Get one case by case_id."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
            if row is None:
                return None
            return dict(row)
        except sqlite3.Error as exc:
            logger.error("Failed to get case %s: %s", case_id, exc)
            return None
