from pathlib import Path
import argparse
import csv
import math
import re
import statistics

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

ID_KEYS = ["micrograph_id", "image_name", "micrograph_name", "micrograph"]
PATH_KEYS = ["mrc_path", "local_mrc_path", "micrograph_path", "image_path", "path"]


def parse_coord(path):
    pts = []
    if not Path(path).is_file():
        return pts
    with open(path, errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("_") or s in {"data_", "loop_"}:
                continue
            low = s.lower()
            if "x_coord" in low or "coordinatex" in low:
                continue
            parts = re.split(r"[\s,]+", s)
            nums = []
            for p in parts:
                try:
                    nums.append(float(p))
                except Exception:
                    pass
            if len(nums) >= 2:
                pts.append((nums[0], nums[1]))
    return pts


def get_mid(row):
    for k in ID_KEYS:
        if row.get(k):
            return Path(row[k]).stem
    for k in PATH_KEYS:
        if row.get(k):
            return Path(row[k]).stem
    raise ValueError("Cannot infer micrograph id")


def get_gt(row):
    for k in GT_KEYS:
        if row.get(k) and Path(row[k]).is_file():
            return Path(row[k])
    return None


def nearest_distances(pred, gt, max_pred=None):
    if max_pred:
        pred = pred[:max_pred]
    ds = []
    dxs = []
    dys = []
    for px, py in pred:
        best = None
        best_dx = None
        best_dy = None
        for gx, gy in gt:
            dx = gx - px
            dy = gy - py
            d = math.hypot(dx, dy)
            if best is None or d < best:
                best = d
                best_dx = dx
                best_dy = dy
        if best is not None:
            ds.append(best)
            dxs.append(best_dx)
            dys.append(best_dy)
    return ds, dxs, dys


def q(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    idx = int(round((len(xs) - 1) * p))
    return xs[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-csv", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--threshold", default="")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--max-pred-per-micrograph", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(args.pairs_csv)))
    per = []
    all_d = []
    all_dx = []
    all_dy = []

    for row in rows:
        mid = get_mid(row)
        gt_path = get_gt(row)
        if gt_path is None:
            continue
        pred_path = Path(args.pred_dir) / f"{mid}.coord"
        pred = parse_coord(pred_path)
        gt = parse_coord(gt_path)
        if not pred or not gt:
            continue

        ds, dxs, dys = nearest_distances(pred, gt, args.max_pred_per_micrograph)
        all_d.extend(ds)
        all_dx.extend(dxs)
        all_dy.extend(dys)

        per.append({
            "micrograph_id": mid,
            "n_pred": len(pred),
            "n_gt": len(gt),
            "median_nearest_dist": q(ds, 0.5),
            "p25_nearest_dist": q(ds, 0.25),
            "p75_nearest_dist": q(ds, 0.75),
            "p90_nearest_dist": q(ds, 0.9),
            "median_dx_gt_minus_pred": q(dxs, 0.5),
            "median_dy_gt_minus_pred": q(dys, 0.5),
        })

    summary = {
        "threshold": args.threshold,
        "n_micrographs": len(per),
        "n_nearest_pairs": len(all_d),
        "median_nearest_dist": q(all_d, 0.5),
        "p25_nearest_dist": q(all_d, 0.25),
        "p75_nearest_dist": q(all_d, 0.75),
        "p90_nearest_dist": q(all_d, 0.9),
        "p95_nearest_dist": q(all_d, 0.95),
        "median_dx_gt_minus_pred": q(all_dx, 0.5),
        "median_dy_gt_minus_pred": q(all_dy, 0.5),
        "mean_dx_gt_minus_pred": statistics.mean(all_dx) if all_dx else None,
        "mean_dy_gt_minus_pred": statistics.mean(all_dy) if all_dy else None,
    }

    with (out / "coordinate_alignment_per_micrograph.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per[0].keys()))
        writer.writeheader()
        writer.writerows(per)

    with (out / "coordinate_alignment_summary.json").open("w") as f:
        import json
        json.dump(summary, f, indent=2)

    md = []
    md.append("# Coordinate Alignment Diagnostic\n")
    md.append("| metric | value |")
    md.append("| --- | ---: |")
    for k, v in summary.items():
        md.append(f"| {k} | {v} |")
    (out / "coordinate_alignment_summary.md").write_text("\n".join(md))

    print(summary)


if __name__ == "__main__":
    main()
