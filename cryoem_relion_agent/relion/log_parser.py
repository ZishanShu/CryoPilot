"""Parse RELION log files for key metrics and warnings.

Resolution is only read from lines that clearly discuss resolution (not job IDs,
iteration numbers, or FSC correlation values).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from relion.star_parser import normalize_resolution_float

logger = logging.getLogger(__name__)

RE_PARTICLE = re.compile(r"(?:particles?|n_particles)[^0-9]*([0-9]+)", re.IGNORECASE)
RE_MAP = re.compile(r"(\S+(?:\.mrc|\.map))", re.IGNORECASE)
_FLOAT = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

# Lines must mention resolution semantics before we trust any number on that line.
_RESOLUTION_LINE_HINTS: Tuple[str, ...] = (
    "final resolution",
    "estimated resolution",
    "global resolution",
    "overall resolution",
    "gold-standard",
    "gold standard",
    "fsc=0.143",
    "fsc = 0.143",
    "fsc 0.143",
    "at fsc",
    "nyquist resolution",
    "angstrom resolution",
    "resolution:",
    "resolution =",
    "resolution is",
    "_rlnfinalresolution",
    "_rlnestimatedresolution",
)


def _extract_resolution_from_log_line(line: str) -> Optional[float]:
    """Pick the most plausible resolution number from a line already known to be resolution-related."""
    lower = line.lower()
    numbers: List[float] = []
    for m in _FLOAT.finditer(line):
        try:
            numbers.append(float(m.group(0)))
        except ValueError:
            continue
    if not numbers:
        return None

    # Prefer the number closest to a resolution keyword phrase.
    best_val: Optional[float] = None
    best_dist = 10**9
    for kw in ("resolution", "angstrom", "fsc", "nyquist"):
        pos = lower.find(kw)
        if pos < 0:
            continue
        for m in _FLOAT.finditer(line):
            try:
                v = float(m.group(0))
            except ValueError:
                continue
            dist = abs(m.start() - pos)
            if dist < best_dist:
                best_dist = dist
                best_val = v
    candidate = best_val if best_val is not None else numbers[0]
    return normalize_resolution_float(candidate, trusted_field=False)


class LogParser:
    """Extract conservative indicators from RELION job logs."""

    def parse(self, log_path: Path) -> Dict[str, Optional[float | int | bool | str]]:
        result: Dict[str, Optional[float | int | bool | str]] = {
            "particle_count": None,
            "resolution": None,
            "job_completed": None,
            "map_path": None,
            "orientation_warning": False,
            "heterogeneity_warning": False,
        }
        try:
            if not log_path.exists():
                return result
            text = log_path.read_text(encoding="utf-8", errors="ignore")
            lower = text.lower()
            result["job_completed"] = any(
                x in lower for x in ["job finished", "finished successfully", "completed"]
            )
            m_particle = RE_PARTICLE.search(text)
            if m_particle:
                try:
                    result["particle_count"] = int(m_particle.group(1))
                except ValueError:
                    pass

            # Resolution: only scan lines that mention resolution semantics.
            best_res: Optional[float] = None
            for line in text.splitlines():
                line_lower = line.lower()
                if not any(hint in line_lower for hint in _RESOLUTION_LINE_HINTS):
                    continue
                # Skip obvious non-resolution lines (job/iteration noise).
                if "iteration" in line_lower and "resolution" not in line_lower:
                    continue
                cand = _extract_resolution_from_log_line(line)
                if cand is None:
                    continue
                if best_res is None or cand < best_res:
                    best_res = cand
            result["resolution"] = best_res

            m_map = RE_MAP.search(text)
            if m_map:
                result["map_path"] = m_map.group(1)
            result["orientation_warning"] = any(
                k in lower for k in ["preferred orientation", "orientation bias", "anisotropy"]
            )
            result["heterogeneity_warning"] = any(
                k in lower
                for k in ["heterogeneity", "multiple conformations", "compositional heterogeneity"]
            )
            return result
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to parse log file %s: %s", log_path, exc)
            return result
