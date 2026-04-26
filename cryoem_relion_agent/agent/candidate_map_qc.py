"""Lightweight QC and ranking utilities for candidate density maps."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _as_path(map_path: str | Path | None) -> Path | None:
    if map_path is None:
        return None
    raw = str(map_path).strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def _empty_metrics(map_path: str | Path | None) -> Dict[str, Any]:
    return {
        "map_path": str(map_path) if map_path is not None else None,
        "file_exists": False,
        "file_size_bytes": 0,
        "mrc_shape": None,
        "finite_fraction": 0.0,
        "density_mean": 0.0,
        "density_std": 0.0,
        "density_min": 0.0,
        "density_max": 0.0,
        "nonzero_fraction": 0.0,
        "contrast_score": 0.0,
        "edge_artifact_score": 0.0,
        "qc_score": 0.0,
        "qc_status": "failed",
        "qc_reason": "",
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _edge_artifact_score(data: np.ndarray) -> float:
    if data.ndim < 2 or min(data.shape) < 3:
        return 0.0
    edge_mask = np.zeros(data.shape, dtype=bool)
    edge_mask[0, ...] = True
    edge_mask[-1, ...] = True
    edge_mask[:, 0, ...] = True
    edge_mask[:, -1, ...] = True
    if data.ndim >= 3:
        edge_mask[:, :, 0, ...] = True
        edge_mask[:, :, -1, ...] = True
    edge = np.abs(data[edge_mask])
    interior = np.abs(data[~edge_mask])
    if interior.size == 0:
        return 0.0
    edge_mean = float(np.mean(edge)) if edge.size else 0.0
    interior_mean = float(np.mean(interior))
    ratio = edge_mean / (interior_mean + 1e-8)
    return max(0.0, ratio - 1.0)


def _qc_score(
    *,
    finite_fraction: float,
    nonzero_fraction: float,
    density_std: float,
    density_min: float,
    density_max: float,
    edge_artifact_score: float,
) -> float:
    # Scoring principle:
    # - Reward finite voxels, non-empty maps, and useful density variance.
    # - Penalize excessive edge artifacts.
    # - Clamp score to [0, 100].
    finite_component = 30.0 * min(1.0, max(0.0, finite_fraction))
    nonzero_component = 25.0 * min(1.0, max(0.0, nonzero_fraction * 2.0))
    dynamic_range = max(0.0, density_max - density_min)
    contrast_component = 25.0 * min(1.0, density_std / 0.5)
    range_component = 20.0 * min(1.0, dynamic_range / 2.0)
    edge_penalty = 30.0 * min(1.0, max(0.0, edge_artifact_score))
    score = finite_component + nonzero_component + contrast_component + range_component - edge_penalty
    return max(0.0, min(100.0, score))


def evaluate_candidate_map(map_path: str | Path) -> Dict[str, Any]:
    """Evaluate one candidate MRC map with deterministic lightweight QC."""

    metrics = _empty_metrics(map_path)
    path = _as_path(map_path)
    if path is None:
        metrics["qc_reason"] = "missing_map_path"
        return metrics

    metrics["map_path"] = str(path)
    if not path.is_file():
        metrics["qc_reason"] = "map_path_unavailable"
        return metrics

    metrics["file_exists"] = True
    try:
        metrics["file_size_bytes"] = int(path.stat().st_size)
    except OSError:
        metrics["file_size_bytes"] = 0

    try:
        import mrcfile  # type: ignore
    except Exception:
        metrics["qc_status"] = "unavailable"
        metrics["qc_reason"] = "mrcfile_unavailable"
        return metrics

    try:
        with mrcfile.open(path, permissive=True) as mrc:
            data = np.asarray(mrc.data)
    except Exception as exc:
        metrics["qc_reason"] = f"unreadable_mrc: {exc}"
        return metrics

    if data.size == 0:
        metrics["qc_reason"] = "empty_mrc_data"
        return metrics

    metrics["mrc_shape"] = [int(x) for x in data.shape]
    finite_mask = np.isfinite(data)
    finite_fraction = float(np.mean(finite_mask))
    metrics["finite_fraction"] = finite_fraction
    finite_values = data[finite_mask]
    if finite_values.size == 0:
        metrics["qc_reason"] = "no_finite_voxels"
        return metrics

    finite_values = finite_values.astype(np.float64, copy=False)
    density_mean = float(np.mean(finite_values))
    density_std = float(np.std(finite_values))
    density_min = float(np.min(finite_values))
    density_max = float(np.max(finite_values))
    nonzero_fraction = float(np.mean(np.abs(finite_values) > 1e-8))
    dynamic_range = density_max - density_min
    contrast_score = 0.0
    if dynamic_range > 1e-12:
        contrast_score = float(min(1.0, density_std / (dynamic_range + 1e-8)))
    edge_score = _edge_artifact_score(data.astype(np.float64, copy=False))

    metrics["density_mean"] = density_mean
    metrics["density_std"] = density_std
    metrics["density_min"] = density_min
    metrics["density_max"] = density_max
    metrics["nonzero_fraction"] = nonzero_fraction
    metrics["contrast_score"] = contrast_score
    metrics["edge_artifact_score"] = edge_score

    if density_std < 1e-8 or dynamic_range < 1e-6 or nonzero_fraction < 1e-4:
        metrics["qc_score"] = 5.0
        metrics["qc_status"] = "warning"
        metrics["qc_reason"] = "near_constant_or_empty_density"
        return metrics

    score = _qc_score(
        finite_fraction=finite_fraction,
        nonzero_fraction=nonzero_fraction,
        density_std=density_std,
        density_min=density_min,
        density_max=density_max,
        edge_artifact_score=edge_score,
    )
    metrics["qc_score"] = score
    if finite_fraction < 0.95:
        metrics["qc_status"] = "warning"
        metrics["qc_reason"] = "contains_non_finite_voxels"
    elif score >= 50.0:
        metrics["qc_status"] = "passed"
        metrics["qc_reason"] = "map_qc_passed"
    else:
        metrics["qc_status"] = "warning"
        metrics["qc_reason"] = "low_qc_score"
    return metrics


def evaluate_candidate_maps(candidate_maps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach candidate-map QC payload to each candidate dictionary."""

    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(candidate_maps):
        row = dict(item)
        row.setdefault("candidate_index", idx)
        map_path = row.get("map_path") or row.get("path")
        qc = evaluate_candidate_map(str(map_path) if map_path is not None else "")
        row["candidate_map_qc"] = qc
        row["qc_score"] = _safe_float(qc.get("qc_score"), default=0.0)
        row["qc_status"] = str(qc.get("qc_status") or "failed")
        row["qc_reason"] = str(qc.get("qc_reason") or "")
        out.append(row)
    return out


def rank_candidate_maps(candidate_maps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return candidates ordered by descending qc_score (deterministic tie-breaks)."""

    evaluated: List[Dict[str, Any]] = []
    for idx, item in enumerate(candidate_maps):
        row = dict(item)
        row.setdefault("candidate_index", idx)
        qc = row.get("candidate_map_qc")
        if not isinstance(qc, dict):
            map_path = row.get("map_path") or row.get("path")
            qc = evaluate_candidate_map(str(map_path) if map_path is not None else "")
            row["candidate_map_qc"] = qc
        row["qc_score"] = _safe_float(row.get("qc_score"), _safe_float(qc.get("qc_score"), 0.0))
        row["qc_status"] = str(row.get("qc_status") or qc.get("qc_status") or "failed")
        row["qc_reason"] = str(row.get("qc_reason") or qc.get("qc_reason") or "")
        evaluated.append(row)

    ranked = sorted(
        evaluated,
        key=lambda x: (
            -_safe_float(x.get("qc_score"), 0.0),
            int(x.get("candidate_index", 0)),
            str(x.get("map_path") or ""),
        ),
    )
    for rank, row in enumerate(ranked):
        row["qc_rank"] = rank
    return ranked
