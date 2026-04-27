#!/usr/bin/env python3
"""Convert Topaz ``picks.star`` (or compatible TSV) to ``predicted_coordinates.tsv``.

Output schema::

    image_name <TAB> x_coord <TAB> y_coord <TAB> score

The picking-accuracy evaluator only needs ``image_name``, ``x_coord``, and
``y_coord``; ``score`` is preserved for downstream analyses (it is filled
with an empty string when the source has no Topaz score column).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from relion.star_parser import iter_star_loop_blocks


_STAR_NAME_KEYS = ("_rlnMicrographName",)
_STAR_X_KEYS = ("_rlnCoordinateX",)
_STAR_Y_KEYS = ("_rlnCoordinateY",)
_STAR_SCORE_KEYS = ("_rlnAutopickFigureOfMerit", "_rlnLogLikeliContribution")


def _idx(cols: List[str], candidates: Iterable[str]) -> Optional[int]:
    for cand in candidates:
        if cand in cols:
            return cols.index(cand)
    return None


def _image_name_from_path(value: str) -> str:
    return Path(value).stem


def _parse_topaz_star(text: str) -> List[Tuple[str, float, float, Optional[float]]]:
    rows: List[Tuple[str, float, float, Optional[float]]] = []
    for _data_block, cols, row_lines in iter_star_loop_blocks(text):
        name_idx = _idx(cols, _STAR_NAME_KEYS)
        x_idx = _idx(cols, _STAR_X_KEYS)
        y_idx = _idx(cols, _STAR_Y_KEYS)
        score_idx = _idx(cols, _STAR_SCORE_KEYS)
        if name_idx is None or x_idx is None or y_idx is None:
            continue
        for raw in row_lines:
            parts = raw.split()
            if len(parts) <= max(name_idx, x_idx, y_idx):
                continue
            try:
                x = float(parts[x_idx])
                y = float(parts[y_idx])
            except ValueError:
                continue
            score: Optional[float] = None
            if score_idx is not None and len(parts) > score_idx:
                try:
                    score = float(parts[score_idx])
                except ValueError:
                    score = None
            rows.append((_image_name_from_path(parts[name_idx]), x, y, score))
    return rows


def _parse_topaz_tsv(text: str) -> List[Tuple[str, float, float, Optional[float]]]:
    """Parse Topaz's native TSV (``image_name``, ``x_coord``, ``y_coord``, ``score``)."""
    out: List[Tuple[str, float, float, Optional[float]]] = []
    reader = csv.DictReader(text.splitlines(), delimiter="\t")
    if not reader.fieldnames:
        return out
    name_col = next(
        (c for c in reader.fieldnames if c.strip().lower() in {"image_name", "micrograph", "rlnmicrographname"}),
        reader.fieldnames[0],
    )
    x_col = next((c for c in reader.fieldnames if c.strip().lower() in {"x_coord", "x", "rlncoordinatex"}), None)
    y_col = next((c for c in reader.fieldnames if c.strip().lower() in {"y_coord", "y", "rlncoordinatey"}), None)
    score_col = next((c for c in reader.fieldnames if c.strip().lower() in {"score", "log_likelihood"}), None)
    if x_col is None or y_col is None:
        return out
    for row in reader:
        try:
            x = float(str(row.get(x_col)).strip())
            y = float(str(row.get(y_col)).strip())
        except (TypeError, ValueError):
            continue
        score: Optional[float] = None
        if score_col is not None:
            try:
                score = float(str(row.get(score_col)).strip())
            except (TypeError, ValueError):
                score = None
        out.append((_image_name_from_path(str(row.get(name_col) or "")), x, y, score))
    return out


def convert_topaz_output_to_pred_tsv(
    topaz_output: Path,
    out_tsv: Path,
) -> Path:
    """Convert ``topaz_output`` (STAR or TSV) to ``predicted_coordinates.tsv``."""
    src = Path(topaz_output)
    if not src.is_file():
        raise FileNotFoundError(f"topaz output not found: {src}")
    text = src.read_text(encoding="utf-8", errors="ignore")
    if src.suffix.lower() == ".star" or "_rlnCoordinateX" in text:
        rows = _parse_topaz_star(text)
    else:
        rows = _parse_topaz_tsv(text)

    out_tsv = Path(out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["image_name", "x_coord", "y_coord", "score"])
        for name, x, y, score in rows:
            writer.writerow([
                name,
                f"{x:.6f}",
                f"{y:.6f}",
                "" if score is None else f"{score:.6f}",
            ])
    return out_tsv


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--topaz-output", required=True, help="Topaz picks.star or topaz TSV file.")
    p.add_argument("--out", required=True, help="Output predicted_coordinates.tsv path.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out = convert_topaz_output_to_pred_tsv(Path(args.topaz_output), Path(args.out))
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
