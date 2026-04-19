"""Minimal GT-vs-predicted coordinate evaluation for picking smoke runs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple


Coord = Tuple[float, float]


def _try_float(text: str) -> Optional[float]:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def read_coordinate_file(path: Path) -> List[Coord]:
    coords: List[Coord] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("_") or line.startswith("data_") or line.startswith("loop_"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        # Support both "x y" and "micrograph x y [score]".
        if _try_float(parts[0]) is not None and _try_float(parts[1]) is not None:
            x = float(parts[0])
            y = float(parts[1])
            coords.append((x, y))
            continue
        if len(parts) >= 3 and _try_float(parts[1]) is not None and _try_float(parts[2]) is not None:
            x = float(parts[1])
            y = float(parts[2])
            coords.append((x, y))
    return coords


def _greedy_match_count(predicted: List[Coord], gt: List[Coord], radius_px: float) -> int:
    if radius_px <= 0 or not predicted or not gt:
        return 0
    radius2 = radius_px * radius_px
    used_gt: set[int] = set()
    matched = 0
    for px, py in predicted:
        best_idx = None
        best_dist2 = None
        for idx, (gx, gy) in enumerate(gt):
            if idx in used_gt:
                continue
            d2 = (px - gx) * (px - gx) + (py - gy) * (py - gy)
            if d2 <= radius2 and (best_dist2 is None or d2 < best_dist2):
                best_idx = idx
                best_dist2 = d2
        if best_idx is not None:
            used_gt.add(best_idx)
            matched += 1
    return matched


def evaluate_picking_predictions(
    *,
    predicted_coord_file: Path,
    gt_coord_files: List[Path],
    match_radius_px: float,
) -> Dict[str, float]:
    predicted = read_coordinate_file(predicted_coord_file) if predicted_coord_file.is_file() else []
    gt: List[Coord] = []
    for path in gt_coord_files:
        if path.is_file():
            gt.extend(read_coordinate_file(path))
    matched = _greedy_match_count(predicted, gt, float(match_radius_px))
    predicted_count = len(predicted)
    gt_count = len(gt)
    precision = (matched / predicted_count) if predicted_count > 0 else 0.0
    recall = (matched / gt_count) if gt_count > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "predicted_count": predicted_count,
        "gt_count": gt_count,
        "matched_count": matched,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "evaluation_available": bool(predicted_count or gt_count),
        "match_radius_px": float(match_radius_px),
    }

