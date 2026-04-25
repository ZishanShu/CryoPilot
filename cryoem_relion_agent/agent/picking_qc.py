"""Picking QC Gate.

Parse Topaz ``picks.star`` (or TSV-style picks file) and optionally compare
against CryoPPP-style ground-truth coordinate CSVs.  Returns a JSON-serializable
QC summary that downstream consumers (scheduler, reporter) attach to the
``run_autopick`` execution metadata.

Stays pure-stdlib: no pandas / numpy.
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "min_total_picks": 100.0,
    "min_mean_picks_per_micrograph": 5.0,
    "max_zero_pick_micrograph_ratio": 0.5,
    "min_f1": 0.05,
}


# ----------------------------- parsing helpers ------------------------------


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _resolve_column(columns: List[str], candidates: Iterable[str]) -> Optional[str]:
    if not columns:
        return None
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _column_index(header: List[str], candidates: Iterable[str]) -> Optional[int]:
    if not header:
        return None
    lower = [c.lower() for c in header]
    for cand in candidates:
        if cand in header:
            return header.index(cand)
        if cand.lower() in lower:
            return lower.index(cand.lower())
    return None


def _parse_loop_columns(lines: List[str], start_idx: int) -> Tuple[List[str], int]:
    columns: List[str] = []
    i = start_idx
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if s.startswith("_"):
            tok = s.split()[0]
            columns.append(tok.lstrip("_"))
            i += 1
            continue
        break
    return columns, i


def _looks_like_tsv(text: str) -> bool:
    head = text.lstrip()
    snippet = head[:128]
    if "data_" in snippet or "loop_" in snippet:
        return False
    first_line = head.split("\n", 1)[0]
    return ("\t" in first_line) or ("," in first_line)


def _to_pick_record(
    rec: Dict[str, str],
    name_key: Optional[str],
    x_key: Optional[str],
    y_key: Optional[str],
    score_key: Optional[str],
) -> Dict[str, Any]:
    name = str(rec.get(name_key, "")) if name_key else ""
    x = _safe_float(rec.get(x_key)) if x_key else None
    y = _safe_float(rec.get(y_key)) if y_key else None
    score = _safe_float(rec.get(score_key)) if score_key else None
    return {"micrograph": name, "x": x, "y": y, "score": score}


def _parse_tsv_picks(text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        return [], []
    sep = "\t" if "\t" in lines[0] else ","
    header = [h.strip() for h in lines[0].split(sep)]
    name_key = _resolve_column(header, ("image_name", "micrograph", "rlnMicrographName"))
    x_key = _resolve_column(header, ("x_coord", "x", "rlnCoordinateX"))
    y_key = _resolve_column(header, ("y_coord", "y", "rlnCoordinateY"))
    score_key = _resolve_column(header, ("score", "rlnAutopickFigureOfMerit", "fom"))
    rows: List[Dict[str, Any]] = []
    for ln in lines[1:]:
        parts = [p.strip() for p in ln.split(sep)]
        if len(parts) != len(header):
            continue
        rec = dict(zip(header, parts))
        rows.append(_to_pick_record(rec, name_key, x_key, y_key, score_key))
    return rows, header


def _parse_star_picks(text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    lines = text.splitlines()
    columns: List[str] = []
    data_start = -1
    for i, ln in enumerate(lines):
        if ln.strip().startswith("loop_"):
            columns, data_start = _parse_loop_columns(lines, i + 1)
            break
    if data_start < 0:
        return [], columns
    name_key = _resolve_column(columns, ("rlnMicrographName", "image_name", "micrograph"))
    x_key = _resolve_column(columns, ("rlnCoordinateX", "x_coord", "x"))
    y_key = _resolve_column(columns, ("rlnCoordinateY", "y_coord", "y"))
    score_key = _resolve_column(columns, ("rlnAutopickFigureOfMerit", "score", "fom"))
    rows: List[Dict[str, Any]] = []
    for ln in lines[data_start:]:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < len(columns):
            continue
        rec = dict(zip(columns, parts))
        rows.append(_to_pick_record(rec, name_key, x_key, y_key, score_key))
    return rows, columns


def parse_topaz_picks_star(path: Path) -> Dict[str, Any]:
    """Parse a Topaz ``picks.star`` (RELION STAR loop) or TSV picks file.

    Returns ``{"rows": [{micrograph, x, y, score}, ...], "columns": [...]}``.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if _looks_like_tsv(text):
        rows, columns = _parse_tsv_picks(text)
    else:
        rows, columns = _parse_star_picks(text)
    return {"rows": rows, "columns": columns}


def count_micrographs_in_star(path: Path) -> int:
    """Count data rows in a RELION ``micrographs.star`` (loop block)."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    lines = text.splitlines()
    data_start = -1
    for i, ln in enumerate(lines):
        if ln.strip().startswith("loop_"):
            _, data_start = _parse_loop_columns(lines, i + 1)
            break
    if data_start < 0:
        return 0
    n = 0
    for ln in lines[data_start:]:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        n += 1
    return n


def _normalize_micrograph_name(name: Any) -> str:
    """Return basename without extension; tolerant to full paths and bare stems."""
    s = str(name or "").strip()
    if not s:
        return ""
    return Path(s).stem


def _micrograph_names_in_star(path: Path) -> List[str]:
    """Return ordered, deduplicated list of normalized ``rlnMicrographName`` basenames."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lines = text.splitlines()
    columns: List[str] = []
    data_start = -1
    for i, ln in enumerate(lines):
        if ln.strip().startswith("loop_"):
            columns, data_start = _parse_loop_columns(lines, i + 1)
            break
    if data_start < 0:
        return []
    name_idx: Optional[int] = None
    for i, c in enumerate(columns):
        if c.lower() in {"rlnmicrographname", "micrograph", "image_name"}:
            name_idx = i
            break
    seen: set = set()
    out: List[str] = []
    for ln in lines[data_start:]:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if name_idx is not None and len(parts) > name_idx:
            tok = parts[name_idx]
        elif parts:
            tok = parts[0]
        else:
            continue
        norm = _normalize_micrograph_name(tok)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def _parse_gt_csv(path: Path) -> List[Tuple[float, float]]:
    coords: List[Tuple[float, float]] = []
    try:
        with Path(path).open("r", encoding="utf-8", errors="replace", newline="") as fh:
            rows = list(csv.reader(fh))
    except OSError:
        return coords
    if not rows:
        return coords
    header = [c.strip() for c in rows[0]]
    x_idx = _column_index(header, ("x_coord", "x", "X", "rlnCoordinateX"))
    y_idx = _column_index(header, ("y_coord", "y", "Y", "rlnCoordinateY"))
    if x_idx is None or y_idx is None:
        # No header - treat first two columns as x,y.
        x_idx, y_idx = 0, 1
        start = 0
    else:
        start = 1
    for r in rows[start:]:
        if len(r) <= max(x_idx, y_idx):
            continue
        x = _safe_float(r[x_idx])
        y = _safe_float(r[y_idx])
        if x is None or y is None:
            continue
        coords.append((x, y))
    return coords


# ----------------------------- QC computation -------------------------------


def _picks_per_micrograph(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in rows:
        m = str(r.get("micrograph") or "")
        counts[m] = counts.get(m, 0) + 1
    return counts


def _greedy_match(
    preds: List[Tuple[float, float]],
    gts: List[Tuple[float, float]],
    tolerance_px: float,
) -> Tuple[int, int, int]:
    """Greedy nearest-neighbor matching with a single tolerance.

    Returns ``(tp, fp, fn)``.
    """
    if not preds and not gts:
        return 0, 0, 0
    if not gts:
        return 0, len(preds), 0
    if not preds:
        return 0, 0, len(gts)
    used = [False] * len(gts)
    tp = 0
    tol2 = float(tolerance_px) ** 2
    order = sorted(range(len(preds)), key=lambda i: (preds[i][1], preds[i][0]))
    for pi in order:
        px, py = preds[pi]
        best = -1
        best_d2 = tol2
        for gi, (gx, gy) in enumerate(gts):
            if used[gi]:
                continue
            d2 = (gx - px) ** 2 + (gy - py) ** 2
            if d2 <= best_d2:
                best = gi
                best_d2 = d2
        if best >= 0:
            used[best] = True
            tp += 1
    fp = len(preds) - tp
    fn = sum(1 for u in used if not u)
    return tp, fp, fn


def _gt_evaluation(
    pred_by_mic: Dict[str, List[Tuple[float, float]]],
    gt_index: Dict[str, Path],
    eval_names: List[str],
    coordinate_scale: float,
    tolerance_px: float,
) -> Dict[str, Any]:
    """Evaluate predictions vs. GT over an explicit micrograph evaluation set.

    Names in ``pred_by_mic`` / ``gt_index`` / ``eval_names`` must already be
    normalized (basename without extension).  Empty preds against non-empty GT
    contribute fully to FN; preds without GT contribute fully to FP.
    """
    tp_total = fp_total = fn_total = 0
    matched_mics = 0
    pred_total = 0
    gt_total = 0
    for name in eval_names:
        preds_raw = pred_by_mic.get(name, [])
        scaled = [(x * coordinate_scale, y * coordinate_scale) for (x, y) in preds_raw]
        gt_path = gt_index.get(name)
        gts = _parse_gt_csv(gt_path) if gt_path is not None else []
        pred_total += len(scaled)
        gt_total += len(gts)
        if scaled and gts:
            matched_mics += 1
        tp, fp, fn = _greedy_match(scaled, gts, tolerance_px)
        tp_total += tp
        fp_total += fp
        fn_total += fn
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "matching_tolerance_px": float(tolerance_px),
        "coordinate_scale": float(coordinate_scale),
        "matched_micrographs": matched_mics,
        "gt_micrographs_available": len(gt_index),
        "evaluated_micrographs": len(eval_names),
        "gt_particles_total": gt_total,
        "pred_particles_total": pred_total,
    }


def evaluate_picking_output(
    picks_path: Path,
    micrographs_star_path: Optional[Path] = None,
    gt_coords_dir: Optional[Path] = None,
    coordinate_scale: float = 1.0,
    matching_tolerance_px: float = 80.0,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Evaluate picking output and apply QC thresholds.

    Returns a JSON-serializable dict.  See :data:`DEFAULT_THRESHOLDS` for
    threshold keys.
    """
    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        for k, v in thresholds.items():
            try:
                th[k] = float(v)
            except (TypeError, ValueError):
                continue

    parsed = parse_topaz_picks_star(Path(picks_path))
    rows = parsed["rows"]
    n_picks = len(rows)
    counts = _picks_per_micrograph(rows)
    n_mics_with_picks = len([m for m, c in counts.items() if c > 0])
    per_mic_counts = sorted(counts.values())
    mean_pp = statistics.fmean(per_mic_counts) if per_mic_counts else 0.0
    median_pp = float(statistics.median(per_mic_counts)) if per_mic_counts else 0.0

    n_total: Optional[int] = None
    zero_count: Optional[int] = None
    zero_ratio: Optional[float] = None
    if micrographs_star_path is not None:
        n_total = count_micrographs_in_star(Path(micrographs_star_path))
        if n_total <= 0:
            n_total = n_mics_with_picks
        if n_total > 0:
            zero_count = max(0, n_total - n_mics_with_picks)
            zero_ratio = round(zero_count / n_total, 6)

    scores = [r.get("score") for r in rows if r.get("score") is not None]
    score_min = round(float(min(scores)), 6) if scores else None
    score_max = round(float(max(scores)), 6) if scores else None
    score_mean = round(statistics.fmean(scores), 6) if scores else None

    qc_flags: List[str] = []
    if n_picks == 0:
        qc_flags.append("empty_picks_file")
    if n_picks < th["min_total_picks"] and "severe_under_picking" not in qc_flags:
        qc_flags.append("severe_under_picking")
    if (
        n_picks > 0
        and mean_pp < th["min_mean_picks_per_micrograph"]
        and "severe_under_picking" not in qc_flags
    ):
        qc_flags.append("severe_under_picking")
    if zero_ratio is not None and zero_ratio > th["max_zero_pick_micrograph_ratio"]:
        qc_flags.append("many_empty_micrographs")

    gt_eval: Optional[Dict[str, Any]] = None
    if gt_coords_dir is not None and Path(gt_coords_dir).is_dir():
        gt_index: Dict[str, Path] = {}
        for p in Path(gt_coords_dir).glob("*.csv"):
            key = _normalize_micrograph_name(p.name)
            if key:
                gt_index[key] = p
        pred_by_mic: Dict[str, List[Tuple[float, float]]] = {}
        for r in rows:
            x = r.get("x")
            y = r.get("y")
            if x is None or y is None:
                continue
            key = _normalize_micrograph_name(r.get("micrograph"))
            if not key:
                continue
            pred_by_mic.setdefault(key, []).append((float(x), float(y)))
        if micrographs_star_path is not None:
            eval_names = _micrograph_names_in_star(Path(micrographs_star_path))
        else:
            union = set(pred_by_mic.keys()) | set(gt_index.keys())
            union.discard("")
            eval_names = sorted(union)
        if eval_names:
            gt_eval = _gt_evaluation(
                pred_by_mic,
                gt_index,
                eval_names,
                coordinate_scale=float(coordinate_scale),
                tolerance_px=float(matching_tolerance_px),
            )
            if gt_eval.get("f1") is not None and gt_eval["f1"] < th["min_f1"]:
                qc_flags.append("low_f1_against_ground_truth")

    qc_status = "passed" if not qc_flags else "failed_qc"
    recommended = (
        "continue_to_extract" if qc_status == "passed" else "threshold_sweep_or_train_topaz"
    )

    return {
        "qc_status": qc_status,
        "recommended_next_action": recommended,
        "n_picks": n_picks,
        "n_micrographs_with_picks": n_mics_with_picks,
        "n_micrographs_total": n_total,
        "zero_pick_micrograph_count": zero_count,
        "zero_pick_micrograph_ratio": zero_ratio,
        "mean_picks_per_micrograph": round(mean_pp, 6),
        "median_picks_per_micrograph": median_pp,
        "picks_per_micrograph": dict(sorted(counts.items())),
        "score_min": score_min,
        "score_max": score_max,
        "score_mean": score_mean,
        "ground_truth_evaluation": gt_eval,
        "qc_flags": qc_flags,
        "thresholds": th,
        "matching_tolerance_px": float(matching_tolerance_px),
        "coordinate_scale": float(coordinate_scale),
        "picks_path": str(picks_path),
        "micrographs_star_path": (str(micrographs_star_path) if micrographs_star_path else None),
        "gt_coords_dir": (str(gt_coords_dir) if gt_coords_dir else None),
    }
