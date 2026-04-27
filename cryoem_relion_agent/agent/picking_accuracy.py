"""Picking-accuracy evaluator using greedy nearest-neighbor coordinate matching.

This module is intentionally small and dependency-light: it only reads TSV
files, evaluates predictions against ground truth at one or more matching
radii, and writes a CSV / JSON summary.  It does **not** invoke any picking
backend.  Use ``scripts.convert_topaz_output_to_pred_tsv`` to convert Topaz's
STAR output into the ``predicted_coordinates.tsv`` format consumed here.

The matching algorithm matches the existing ``agent.picking_qc`` greedy
nearest-neighbor logic but exposes per-image precision/recall/F1.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# --------------------------------------------------------------------------- #
# TSV parsing
# --------------------------------------------------------------------------- #


def _read_tsv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"TSV not found: {p}")
    with p.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    return fieldnames, rows


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _group_by_image(
    rows: Iterable[Mapping[str, str]],
    *,
    image_name_col: str,
    x_col: str,
    y_col: str,
    split_col: Optional[str] = None,
    split_filter: Optional[str] = None,
) -> Dict[str, List[Tuple[float, float]]]:
    grouped: Dict[str, List[Tuple[float, float]]] = {}
    for row in rows:
        if split_filter is not None and split_col is not None:
            if str(row.get(split_col, "")).strip() != split_filter:
                continue
        name = str(row.get(image_name_col, "")).strip()
        if not name:
            continue
        x = _coerce_float(row.get(x_col))
        y = _coerce_float(row.get(y_col))
        if x is None or y is None:
            continue
        grouped.setdefault(name, []).append((x, y))
    return grouped


# --------------------------------------------------------------------------- #
# Greedy matching
# --------------------------------------------------------------------------- #


def _greedy_match_per_image(
    preds: List[Tuple[float, float]],
    gts: List[Tuple[float, float]],
    radius_px: float,
) -> int:
    """Return number of matched pairs.

    Each prediction matches at most one GT; each GT matches at most one
    prediction.  We sort all candidate (pred, gt) pairs by distance and
    greedily match shortest-first.
    """
    if not preds or not gts:
        return 0
    r2 = float(radius_px) ** 2
    pairs: List[Tuple[float, int, int]] = []
    for pi, (px, py) in enumerate(preds):
        for gi, (gx, gy) in enumerate(gts):
            d2 = (px - gx) ** 2 + (py - gy) ** 2
            if d2 <= r2:
                pairs.append((d2, pi, gi))
    if not pairs:
        return 0
    pairs.sort(key=lambda t: (t[0], t[1], t[2]))
    used_pred: set = set()
    used_gt: set = set()
    matched = 0
    for _d2, pi, gi in pairs:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        matched += 1
    return matched


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PickingAccuracyMetrics:
    n_micrographs: int
    n_gt: int
    n_pred: int
    n_matched: int
    precision: float
    recall: float
    f1: float
    matching_radius_px: float
    empty_output: bool
    split: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_micrographs": int(self.n_micrographs),
            "n_gt": int(self.n_gt),
            "n_pred": int(self.n_pred),
            "n_matched": int(self.n_matched),
            "precision": round(float(self.precision), 6),
            "recall": round(float(self.recall), 6),
            "f1": round(float(self.f1), 6),
            "matching_radius_px": float(self.matching_radius_px),
            "empty_output": bool(self.empty_output),
            "split": self.split,
        }


def _compute_metrics(
    gt_by_image: Dict[str, List[Tuple[float, float]]],
    pred_by_image: Dict[str, List[Tuple[float, float]]],
    *,
    radius_px: float,
    split: Optional[str],
) -> PickingAccuracyMetrics:
    image_names = sorted(set(gt_by_image) | set(pred_by_image))
    n_gt = sum(len(v) for v in gt_by_image.values())
    n_pred = sum(len(v) for v in pred_by_image.values())
    n_matched = 0
    for name in image_names:
        n_matched += _greedy_match_per_image(
            pred_by_image.get(name, []),
            gt_by_image.get(name, []),
            radius_px,
        )
    precision = (n_matched / n_pred) if n_pred > 0 else 0.0
    recall = (n_matched / n_gt) if n_gt > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return PickingAccuracyMetrics(
        n_micrographs=len(image_names),
        n_gt=int(n_gt),
        n_pred=int(n_pred),
        n_matched=int(n_matched),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        matching_radius_px=float(radius_px),
        empty_output=bool(n_pred == 0),
        split=split,
    )


def evaluate_picking_accuracy(
    gt_tsv: Path,
    pred_tsv: Path,
    *,
    split: Optional[str] = None,
    matching_radius_px: float = 30.0,
    image_name_col: str = "image_name",
    gt_x_col: str = "x_coord",
    gt_y_col: str = "y_coord",
    pred_x_col: str = "x_coord",
    pred_y_col: str = "y_coord",
) -> Dict[str, Any]:
    """Evaluate one prediction TSV against one GT TSV at a single radius."""
    _gt_fields, gt_rows = _read_tsv(Path(gt_tsv))
    _pred_fields, pred_rows = _read_tsv(Path(pred_tsv))

    gt_by_image = _group_by_image(
        gt_rows,
        image_name_col=image_name_col,
        x_col=gt_x_col,
        y_col=gt_y_col,
        split_col="split",
        split_filter=split,
    )
    pred_by_image = _group_by_image(
        pred_rows,
        image_name_col=image_name_col,
        x_col=pred_x_col,
        y_col=pred_y_col,
        # Predictions usually have no split column; only filter when present.
        split_col="split" if any("split" in r for r in pred_rows) else None,
        split_filter=split if any("split" in r for r in pred_rows) else None,
    )

    metrics = _compute_metrics(
        gt_by_image,
        pred_by_image,
        radius_px=float(matching_radius_px),
        split=split,
    )
    return metrics.to_dict()


def evaluate_picking_accuracy_radius_sensitivity(
    gt_tsv: Path,
    pred_tsv: Path,
    *,
    split: Optional[str] = None,
    radii_px: Sequence[float] = (20, 30, 40),
    image_name_col: str = "image_name",
    gt_x_col: str = "x_coord",
    gt_y_col: str = "y_coord",
    pred_x_col: str = "x_coord",
    pred_y_col: str = "y_coord",
) -> List[Dict[str, Any]]:
    """Run :func:`evaluate_picking_accuracy` at each requested radius."""
    out: List[Dict[str, Any]] = []
    for r in radii_px:
        out.append(
            evaluate_picking_accuracy(
                gt_tsv=Path(gt_tsv),
                pred_tsv=Path(pred_tsv),
                split=split,
                matching_radius_px=float(r),
                image_name_col=image_name_col,
                gt_x_col=gt_x_col,
                gt_y_col=gt_y_col,
                pred_x_col=pred_x_col,
                pred_y_col=pred_y_col,
            )
        )
    return out


def write_radius_sensitivity_outputs(
    rows: Sequence[Mapping[str, Any]],
    *,
    csv_path: Path,
    summary_json_path: Path,
    primary_radius_px: Optional[float] = None,
) -> Tuple[Path, Path]:
    """Write ``picking_accuracy_by_radius.csv`` and ``picking_accuracy_summary.json``."""
    csv_path = Path(csv_path)
    summary_json_path = Path(summary_json_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "matching_radius_px",
        "split",
        "n_micrographs",
        "n_gt",
        "n_pred",
        "n_matched",
        "precision",
        "recall",
        "f1",
        "empty_output",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    primary: Optional[Mapping[str, Any]] = None
    if primary_radius_px is not None:
        for row in rows:
            try:
                if abs(float(row.get("matching_radius_px") or 0) - float(primary_radius_px)) < 1e-9:
                    primary = row
                    break
            except (TypeError, ValueError):
                continue
    summary: Dict[str, Any] = {
        "primary_radius_px": (float(primary_radius_px) if primary_radius_px is not None else None),
        "primary_metrics": dict(primary) if primary else None,
        "radii": [dict(r) for r in rows],
    }
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, summary_json_path


__all__ = [
    "PickingAccuracyMetrics",
    "evaluate_picking_accuracy",
    "evaluate_picking_accuracy_radius_sensitivity",
    "write_radius_sensitivity_outputs",
]
