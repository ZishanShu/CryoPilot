"""Minimal and extensible STAR parser for RELION outputs.

Resolution and micrograph counts use structured, keyword-aware parsing — not blind
regex over the whole file — to avoid FSC-table values (e.g. 1.000000) and
particle-stack references being mistaken for global resolution or micrograph counts.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Cryo-EM single-particle map resolution in Angstrom — conservative band for sanity checks.
PLAUSIBLE_RESOLUTION_MIN_ANGSTROM = 0.5
PLAUSIBLE_RESOLUTION_MAX_ANGSTROM = 50.0

# RELION single-value fields that unambiguously store global estimated resolution (Angstrom).
_TRUSTED_RESOLUTION_FIELD_NAMES: Tuple[str, ...] = (
    "_rlnFinalResolution",
    "_rlnEstimatedResolution",
    "_rlnGoldStandardResolution",
    "_rlnCurrentResolution",
    "_rlnOverallResolution",
)

_FLOAT_TOKEN = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def is_plausible_global_resolution_angstrom(value: float, trusted_field: bool = False) -> bool:
    """
    Return True if value is in a plausible Angstrom range for a published map.

    If ``trusted_field`` is False, very round values like 1.0 are rejected because they
    often come from FSC/column noise rather than a reported resolution.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if math.isnan(v) or v <= 0.0:
        return False
    if not (PLAUSIBLE_RESOLUTION_MIN_ANGSTROM < v < PLAUSIBLE_RESOLUTION_MAX_ANGSTROM):
        return False
    if not trusted_field and abs(v - 1.0) < 1e-6:
        # 1.0 Å without a trusted field is a common false positive (e.g. FSC=1.0).
        return False
    return True


def normalize_resolution_float(value: object, trusted_field: bool = False) -> Optional[float]:
    """Normalize a single resolution candidate to float or None."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v):
        return None
    if not is_plausible_global_resolution_angstrom(v, trusted_field=trusted_field):
        return None
    return v


def extract_column_names_from_loop_header(lines: List[str], start_idx: int) -> Tuple[List[str], int]:
    """Parse RELION loop column names starting after ``loop_``; return (names, index after header)."""
    cols: List[str] = []
    i = start_idx
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        if stripped.startswith("_"):
            token = stripped.split()[0]
            cols.append(token)
            i += 1
            continue
        break
    return cols, i


def iter_star_loop_blocks(text: str) -> Iterator[Tuple[Optional[str], List[str], List[str]]]:
    """
    Yield (current_data_block_name, column_names, data_row_lines) for each ``loop_`` section.

    ``data_*`` lines set the active block name for subsequent loops.
    """
    lines = text.splitlines()
    i = 0
    current_data: Optional[str] = None
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("data_"):
            current_data = stripped
            i += 1
            continue
        if stripped.startswith("loop_"):
            i += 1
            cols, i = extract_column_names_from_loop_header(lines, i)
            row_lines: List[str] = []
            while i < len(lines):
                row = lines[i].strip()
                if not row or row.startswith("#"):
                    i += 1
                    continue
                if row.startswith("data_") or row.startswith("loop_"):
                    break
                if row.startswith("_"):
                    break
                row_lines.append(lines[i])
                i += 1
            yield current_data, cols, row_lines
            continue
        i += 1


def count_unique_micrographs_from_star_text(text: str) -> Optional[int]:
    """
    Count unique ``_rlnMicrographName`` entries when that column exists in a loop.

    Particle-level STAR files may list many rows; we deduplicate by micrograph path.
    """
    unique: Set[str] = set()
    found_column = False
    for _data_block, cols, row_lines in iter_star_loop_blocks(text):
        if "_rlnMicrographName" not in cols:
            continue
        found_column = True
        idx = cols.index("_rlnMicrographName")
        for raw in row_lines:
            parts = raw.split()
            if len(parts) <= idx:
                continue
            name = parts[idx].strip()
            if name:
                unique.add(name)
    if not found_column:
        return None
    return len(unique) if unique else 0


def count_micrograph_rows_data_micrographs(text: str) -> Optional[int]:
    """
    Count rows in a ``data_micrographs`` loop (one row ≈ one micrograph when no dedup needed).
    """
    rows_for_block: List[str] = []
    for data_block, cols, row_lines in iter_star_loop_blocks(text):
        if data_block and "data_micrographs" in data_block and cols and row_lines:
            # Prefer counting rows when _rlnMicrographName is present (movie pipeline).
            if "_rlnMicrographName" in cols:
                return len(row_lines)
            rows_for_block.extend(row_lines)
    if rows_for_block:
        return len(rows_for_block)
    return None


def extract_resolution_from_star_text(text: str) -> Optional[float]:
    """
    Extract global map resolution (Angstrom) from STAR text using trusted fields only.

    Does **not** scrape arbitrary numbers from FSC tables (``_rlnResolution`` in ``data_fsc``).
    """
    # 1) Single-line RELION metadata: _rlnFinalResolution <value>
    for field in _TRUSTED_RESOLUTION_FIELD_NAMES:
        pattern = re.compile(
            rf"^\s*{re.escape(field)}\s+({_FLOAT_TOKEN})\s*$",
            re.MULTILINE | re.IGNORECASE,
        )
        m = pattern.search(text)
        if m:
            val = normalize_resolution_float(m.group(1), trusted_field=True)
            if val is not None:
                return val

    # 2) Optional: legacy "estimated resolution" in comments (still keyword-scoped)
    for line in text.splitlines():
        lower = line.lower()
        if "estimated resolution" in lower and "_rln" not in line:
            m2 = re.search(rf"({_FLOAT_TOKEN})\s*[aå]?", line, re.IGNORECASE)
            if m2:
                val = normalize_resolution_float(m2.group(1), trusted_field=False)
                if val is not None:
                    return val
    return None


class StarParser:
    """Parse selected metrics from STAR files with graceful fallback."""

    def parse(self, star_path: Path) -> Dict[str, Optional[float | int | bool | str]]:
        result: Dict[str, Optional[float | int | bool | str]] = {
            "particle_count": None,
            "resolution": None,
            "job_completed": None,
            "map_path": None,
            "orientation_warning": None,
            "heterogeneity_warning": None,
        }
        try:
            if not star_path.exists():
                return result
            text = star_path.read_text(encoding="utf-8", errors="ignore")

            # Resolution: trusted fields / keywords only (never blind global regex).
            result["resolution"] = extract_resolution_from_star_text(text)

            lines = text.splitlines()
            particle_count = 0
            for line in lines:
                s = line.strip()
                if not s:
                    continue
                if s.startswith(("#", "_", "data_", "loop_")):
                    continue
                particle_count += 1
            result["particle_count"] = particle_count

            lower = text.lower()
            result["job_completed"] = "finished" in lower or "completed" in lower
            result["orientation_warning"] = any(
                x in lower for x in ["preferred orientation", "orientation bias", "anisotropy"]
            )
            result["heterogeneity_warning"] = any(
                x in lower for x in ["heterogeneity", "multiple conformations"]
            )

            # Map path: prefer explicit RELION post-process map fields when present.
            map_match = re.search(
                r"^\s*_rlnPostprocessedMapMasked\s+(\S+)\s*$",
                text,
                re.MULTILINE | re.IGNORECASE,
            )
            if not map_match:
                map_match = re.search(
                    r"^\s*_rlnPostprocessedMap\s+(\S+)\s*$",
                    text,
                    re.MULTILINE | re.IGNORECASE,
                )
            if map_match:
                result["map_path"] = map_match.group(1)

            return result
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to parse STAR file %s: %s", star_path, exc)
            return result
