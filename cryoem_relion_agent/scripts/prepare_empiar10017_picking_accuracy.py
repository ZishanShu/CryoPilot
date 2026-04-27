#!/usr/bin/env python3
"""Prepare an EMPIAR-10017 picking-accuracy run.

Reads ``.mrc`` micrographs and matching ``.coord`` ground-truth files from
``<dataset_root>/data``, pairs them by stem, splits into train/dev/test with
a fixed random seed, and writes everything a downstream picking-accuracy
case needs into ``<run_root>/input``:

* ``split_manifest.json``    -- machine-readable split summary
* ``gt_coordinates.tsv``     -- merged GT coordinates with split/image info
* ``micrographs_train.star`` -- RELION ``data_micrographs`` STAR file (train)
* ``micrographs_dev.star``   -- ditto (dev)
* ``micrographs_test.star``  -- ditto (test)
* ``case.yaml``              -- self-describing case spec for the runner

The script does **not** copy or modify the source ``.mrc`` files.  It writes
absolute paths into the manifests so the case file is fully reproducible.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


@dataclass
class PreparedSplit:
    train: List[Path] = field(default_factory=list)
    dev: List[Path] = field(default_factory=list)
    test: List[Path] = field(default_factory=list)


def _parse_coord_file(coord_path: Path) -> List[Tuple[float, float]]:
    """Parse a ``.coord`` file: take first two numeric columns per line."""
    coords: List[Tuple[float, float]] = []
    if not coord_path.is_file():
        return coords
    text = coord_path.read_text(encoding="utf-8", errors="ignore")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
        except ValueError:
            continue
        coords.append((x, y))
    return coords


def _pair_micrographs_with_coords(data_dir: Path) -> List[Tuple[Path, Path]]:
    """Return a list of ``(mrc_path, coord_path)`` pairs sorted by stem."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data directory not found: {data_dir}")
    mrc_by_stem: Dict[str, Path] = {}
    for p in sorted(data_dir.iterdir()):
        if p.is_file() and p.suffix.lower() == ".mrc":
            mrc_by_stem[p.stem] = p
    coord_by_stem: Dict[str, Path] = {}
    for p in sorted(data_dir.iterdir()):
        if p.is_file() and p.suffix.lower() == ".coord":
            coord_by_stem[p.stem] = p
    pairs: List[Tuple[Path, Path]] = []
    for stem in sorted(mrc_by_stem):
        coord = coord_by_stem.get(stem)
        if coord is None:
            continue
        pairs.append((mrc_by_stem[stem], coord))
    return pairs


def _split_pairs(
    pairs: List[Tuple[Path, Path]],
    *,
    train_size: int,
    dev_size: int,
    test_size: int,
    random_seed: int,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    if len(pairs) < (train_size + dev_size + test_size):
        raise ValueError(
            f"requested splits ({train_size}/{dev_size}/{test_size}) exceed "
            f"available paired micrographs ({len(pairs)})"
        )
    rng = random.Random(int(random_seed))
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    train = sorted(shuffled[:train_size], key=lambda x: x[0].stem)
    dev = sorted(shuffled[train_size:train_size + dev_size], key=lambda x: x[0].stem)
    test = sorted(
        shuffled[train_size + dev_size:train_size + dev_size + test_size],
        key=lambda x: x[0].stem,
    )
    return train, dev, test


def _write_micrographs_star(paths: List[Path], star_path: Path) -> Path:
    star_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "data_micrographs",
        "",
        "loop_",
        "_rlnMicrographName #1",
    ]
    lines.extend(str(p) for p in paths)
    lines.append("")
    star_path.write_text("\n".join(lines), encoding="utf-8")
    return star_path


def _write_gt_tsv(
    split_pairs: Dict[str, List[Tuple[Path, Path]]], out_path: Path
) -> Tuple[Path, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[str] = []
    rows.append("\t".join(["split", "image_name", "micrograph_path", "coord_path", "x_coord", "y_coord"]))
    n = 0
    for split, pairs in split_pairs.items():
        for mrc, coord in pairs:
            for x, y in _parse_coord_file(coord):
                rows.append("\t".join([split, mrc.stem, str(mrc), str(coord), f"{x:.6f}", f"{y:.6f}"]))
                n += 1
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return out_path, n


def _write_case_yaml(
    *,
    case_path: Path,
    dataset_root: Path,
    run_root: Path,
    input_dir: Path,
    split_manifest: Path,
    gt_tsv: Path,
    star_train: Path,
    star_dev: Path,
    star_test: Path,
) -> Path:
    case = {
        "case_id": "P1_topaz_empiar10017_picking_accuracy",
        "case_type": "picking_accuracy",
        "dataset": {
            "name": "EMPIAR-10017",
            "local_root": str(dataset_root),
        },
        "input": {
            "split_manifest": str(split_manifest),
            "gt_coordinates": str(gt_tsv),
            "micrographs_train_star": str(star_train),
            "micrographs_dev_star": str(star_dev),
            "micrographs_test_star": str(star_test),
        },
        "evaluation": {
            "type": "coordinate_matching",
            "matching_radius_px": 30,
            "radius_sensitivity_px": [20, 30, 40],
        },
        "metrics": ["precision", "recall", "f1", "n_gt", "n_pred", "n_matched"],
        "run_root": str(run_root),
        "input_dir": str(input_dir),
    }
    case_path.parent.mkdir(parents=True, exist_ok=True)
    case_path.write_text(yaml.safe_dump(case, sort_keys=False), encoding="utf-8")
    return case_path


def prepare_empiar10017_picking_accuracy(
    *,
    dataset_root: Path,
    run_root: Path,
    train_size: int = 60,
    dev_size: int = 12,
    test_size: int = 12,
    random_seed: int = 42,
) -> Dict[str, object]:
    """Build the EMPIAR-10017 picking-accuracy input bundle.

    Returns the materialized split manifest dict.
    """
    dataset_root = Path(dataset_root).resolve()
    run_root = Path(run_root).resolve()
    data_dir = dataset_root / "data"

    pairs = _pair_micrographs_with_coords(data_dir)
    train, dev, test = _split_pairs(
        pairs,
        train_size=int(train_size),
        dev_size=int(dev_size),
        test_size=int(test_size),
        random_seed=int(random_seed),
    )

    input_dir = run_root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    star_train = _write_micrographs_star([m for m, _c in train], input_dir / "micrographs_train.star")
    star_dev = _write_micrographs_star([m for m, _c in dev], input_dir / "micrographs_dev.star")
    star_test = _write_micrographs_star([m for m, _c in test], input_dir / "micrographs_test.star")

    gt_tsv, n_gt_rows = _write_gt_tsv(
        {"train": train, "dev": dev, "test": test},
        input_dir / "gt_coordinates.tsv",
    )

    def _split_summary(label: str, pairs_for: List[Tuple[Path, Path]]) -> Dict[str, object]:
        return {
            "split": label,
            "count": len(pairs_for),
            "items": [
                {
                    "image_name": mrc.stem,
                    "micrograph_path": str(mrc),
                    "coord_path": str(coord),
                }
                for mrc, coord in pairs_for
            ],
        }

    manifest: Dict[str, object] = {
        "case_id": "P1_topaz_empiar10017_picking_accuracy",
        "dataset_root": str(dataset_root),
        "data_dir": str(data_dir),
        "run_root": str(run_root),
        "input_dir": str(input_dir),
        "random_seed": int(random_seed),
        "n_paired": len(pairs),
        "splits": {
            "train": _split_summary("train", train),
            "dev": _split_summary("dev", dev),
            "test": _split_summary("test", test),
        },
        "files": {
            "split_manifest": str(input_dir / "split_manifest.json"),
            "gt_coordinates": str(gt_tsv),
            "micrographs_train_star": str(star_train),
            "micrographs_dev_star": str(star_dev),
            "micrographs_test_star": str(star_test),
            "case_yaml": str(input_dir / "case.yaml"),
        },
        "n_gt_coordinate_rows": int(n_gt_rows),
    }
    manifest_path = input_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_case_yaml(
        case_path=input_dir / "case.yaml",
        dataset_root=dataset_root,
        run_root=run_root,
        input_dir=input_dir,
        split_manifest=manifest_path,
        gt_tsv=gt_tsv,
        star_train=star_train,
        star_dev=star_dev,
        star_test=star_test,
    )

    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-root",
        required=True,
        help="EMPIAR-10017 dataset root (must contain data/*.mrc and data/*.coord).",
    )
    p.add_argument(
        "--run-root",
        required=True,
        help="Run root directory; outputs will be placed under <run-root>/input.",
    )
    p.add_argument("--train-size", type=int, default=60)
    p.add_argument("--dev-size", type=int, default=12)
    p.add_argument("--test-size", type=int, default=12)
    p.add_argument("--random-seed", type=int, default=42)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = prepare_empiar10017_picking_accuracy(
        dataset_root=Path(args.dataset_root),
        run_root=Path(args.run_root),
        train_size=int(args.train_size),
        dev_size=int(args.dev_size),
        test_size=int(args.test_size),
        random_seed=int(args.random_seed),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
