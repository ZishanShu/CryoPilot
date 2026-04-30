from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


Point = Tuple[float, float]


ID_KEYS = [
    "micrograph_id",
    "image_name",
    "micrograph_name",
    "micrograph",
    "case_id",
]

MRC_KEYS = [
    "mrc_path",
    "local_mrc_path",
    "micrograph_path",
    "image_path",
    "path",
]

GT_KEYS = [
    "gt_coord_path",
    "gt_path",
    "target_coord_path",
    "reference_coord_path",
    "coord_path",
    "ground_truth_coord_path",
    "gt_coordinate_path",
    "manual_coord_path",
]


def parse_threshold_arg(x: str) -> Tuple[str, Path]:
    if "=" not in x:
        raise ValueError(f"Expected THRESHOLD=DIR, got: {x}")
    th, d = x.split("=", 1)
    th = th.strip()
    d = d.strip()
    if not th:
        raise ValueError(f"Empty threshold in: {x}")
    if not d:
        raise ValueError(f"Empty directory in: {x}")
    return th, Path(d)


def row_micrograph_id(row: Dict[str, str]) -> str:
    for k in ID_KEYS:
        v = row.get(k)
        if v:
            return Path(v).stem

    for k in MRC_KEYS:
        v = row.get(k)
        if v:
            return Path(v).stem

    raise ValueError(f"Cannot infer micrograph_id from row keys: {list(row.keys())}")


def row_gt_path(row: Dict[str, str]) -> Optional[Path]:
    for k in GT_KEYS:
        v = row.get(k)
        if v:
            p = Path(v)
            if p.is_file():
                return p
    return None


def to_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def parse_coord_file(path: Path) -> List[Point]:
    pts: List[Point] = []
    if not path.is_file():
        return pts

    with path.open(errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue

            low = s.lower()
            if "x_coord" in low or "rlncoordinatex" in low or "coordinatex" in low:
                continue
            if s in {"data_", "data_particles", "loop_"}:
                continue
            if s.startswith("_"):
                continue

            parts = re.split(r"[\s,]+", s)

            nums: List[float] = []
            for p in parts:
                val = to_float(p)
                if val is not None and math.isfinite(val):
                    nums.append(val)

            if len(nums) >= 2:
                pts.append((nums[0], nums[1]))

    return pts


def grid_match(pred: List[Point], gt: List[Point], radius: float) -> Tuple[int, int, int]:
    """Greedy one-to-one matching within radius, using a grid index over GT."""
    if not pred and not gt:
        return 0, 0, 0
    if not pred:
        return 0, 0, len(gt)
    if not gt:
        return 0, len(pred), 0

    cell = max(radius, 1.0)
    r2 = radius * radius

    grid: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for i, (x, y) in enumerate(gt):
        grid[(int(x // cell), int(y // cell))].append(i)

    matched_gt = set()
    tp = 0

    # 高分已经在 Topaz 文件中通常排在前面；这里按输入顺序贪心匹配。
    for px, py in pred:
        gx0 = int(px // cell)
        gy0 = int(py // cell)

        best_i = None
        best_d2 = None

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for gi in grid.get((gx0 + dx, gy0 + dy), []):
                    if gi in matched_gt:
                        continue
                    gx, gy = gt[gi]
                    d2 = (px - gx) ** 2 + (py - gy) ** 2
                    if d2 <= r2 and (best_d2 is None or d2 < best_d2):
                        best_d2 = d2
                        best_i = gi

        if best_i is not None:
            matched_gt.add(best_i)
            tp += 1

    fp = len(pred) - tp
    fn = len(gt) - tp
    return tp, fp, fn


def metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-csv", required=True, type=Path)
    ap.add_argument("--pred", action="append", required=True, help="THRESHOLD=prediction_dir")
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--match-radius", type=float, default=20.0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    pred_dirs = [parse_threshold_arg(x) for x in args.pred]
    out = args.out_root
    out.mkdir(parents=True, exist_ok=True)

    with args.pairs_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.limit:
        rows = rows[: args.limit]

    usable = []
    missing_gt = 0

    for row in rows:
        mid = row_micrograph_id(row)
        gt_path = row_gt_path(row)
        if gt_path is None:
            missing_gt += 1
            continue
        gt = parse_coord_file(gt_path)
        if not gt:
            missing_gt += 1
            continue
        usable.append((mid, gt_path, gt))

    if not usable:
        raise SystemExit(
            "No usable GT coordinate files found. "
            "Check pairs CSV columns. Expected one of: "
            + ", ".join(GT_KEYS)
        )

    per_micro_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for th, pred_dir in pred_dirs:
        total_tp = total_fp = total_fn = 0
        total_pred = 0
        total_gt = 0
        n_missing_pred = 0
        n_zero_pred = 0
        macro_precisions = []
        macro_recalls = []
        macro_f1s = []

        for mid, gt_path, gt in usable:
            pred_path = pred_dir / f"{mid}.coord"
            pred = parse_coord_file(pred_path)

            if not pred_path.is_file():
                n_missing_pred += 1

            if len(pred) == 0:
                n_zero_pred += 1

            tp, fp, fn = grid_match(pred, gt, args.match_radius)
            m = metrics(tp, fp, fn)

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_pred += len(pred)
            total_gt += len(gt)

            macro_precisions.append(m["precision"])
            macro_recalls.append(m["recall"])
            macro_f1s.append(m["f1"])

            per_micro_rows.append({
                "threshold": th,
                "micrograph_id": mid,
                "pred_path": str(pred_path),
                "gt_path": str(gt_path),
                "n_pred": len(pred),
                "n_gt": len(gt),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
            })

        micro = metrics(total_tp, total_fp, total_fn)
        macro_precision = sum(macro_precisions) / len(macro_precisions) if macro_precisions else 0.0
        macro_recall = sum(macro_recalls) / len(macro_recalls) if macro_recalls else 0.0
        macro_f1 = sum(macro_f1s) / len(macro_f1s) if macro_f1s else 0.0

        summary_rows.append({
            "threshold": th,
            "pred_dir": str(pred_dir),
            "n_micrographs_eval": len(usable),
            "n_missing_pred": n_missing_pred,
            "n_zero_pred": n_zero_pred,
            "zero_pred_ratio": n_zero_pred / len(usable) if usable else 0.0,
            "total_pred": total_pred,
            "total_gt": total_gt,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "micro_precision": micro["precision"],
            "micro_recall": micro["recall"],
            "micro_f1": micro["f1"],
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
        })

    # 选 micro_f1 最高。如果并列，再选 recall 高，再选 fp 少。
    best = sorted(
        summary_rows,
        key=lambda r: (
            float(r["micro_f1"]),
            float(r["micro_recall"]),
            -int(r["fp"]),
        ),
        reverse=True,
    )[0]

    for r in summary_rows:
        r["selected_by_micro_f1"] = str(r["threshold"] == best["threshold"])

    summary_csv = out / "topaz_threshold_sweep_f1_summary.csv"
    per_micro_csv = out / "topaz_threshold_sweep_f1_per_micrograph.csv"
    summary_json = out / "topaz_threshold_sweep_f1_summary.json"
    summary_md = out / "topaz_threshold_sweep_f1_summary.md"

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with per_micro_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_micro_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_micro_rows)

    payload = {
        "pairs_csv": str(args.pairs_csv),
        "match_radius": args.match_radius,
        "n_pairs_rows": len(rows),
        "n_micrographs_eval": len(usable),
        "n_missing_or_empty_gt": missing_gt,
        "best_threshold_by_micro_f1": best["threshold"],
        "best_summary": best,
        "summary_rows": summary_rows,
        "outputs": {
            "summary_csv": str(summary_csv),
            "per_micro_csv": str(per_micro_csv),
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
        },
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Topaz Threshold Sweep F1 Evaluation\n")
    lines.append(f"- Pairs CSV: `{args.pairs_csv}`")
    lines.append(f"- Matching radius: {args.match_radius} px")
    lines.append(f"- Evaluated micrographs with GT: {len(usable)}")
    lines.append(f"- Missing/empty GT rows: {missing_gt}")
    lines.append(f"- Selected threshold by micro-F1: `{best['threshold']}`")
    lines.append("")
    lines.append("## Summary table\n")
    lines.append("| threshold | n_micrographs | total_pred | total_gt | TP | FP | FN | micro_precision | micro_recall | micro_F1 | macro_F1 | selected |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for r in summary_rows:
        lines.append(
            f"| {r['threshold']} | {r['n_micrographs_eval']} | {r['total_pred']} | {r['total_gt']} | "
            f"{r['tp']} | {r['fp']} | {r['fn']} | "
            f"{float(r['micro_precision']):.4f} | {float(r['micro_recall']):.4f} | "
            f"{float(r['micro_f1']):.4f} | {float(r['macro_f1']):.4f} | "
            f"{r['selected_by_micro_f1']} |"
        )
    lines.append("")
    lines.append("## Interpretation\n")
    lines.append(
        "This table evaluates particle-picking quality using ground-truth coordinate matching. "
        "The closed-loop selector should prefer micro-F1 when GT is available, rather than selecting only by extracted particle count."
    )
    lines.append("")
    lines.append("## Output files\n")
    lines.append(f"- Summary CSV: `{summary_csv}`")
    lines.append(f"- Per-micrograph CSV: `{per_micro_csv}`")
    lines.append(f"- JSON: `{summary_json}`")

    summary_md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
