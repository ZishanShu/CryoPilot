#!/usr/bin/env python3
"""Prepare a small CryoPPP subset for one Topaz smoke run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.data_prep.cryoppp_smoke_prep import prepare_cryoppp_topaz_smoke


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True, help="Path to one CryoPPP dataset root")
    p.add_argument("--run-root", required=True, help="Root directory for smoke run outputs")
    p.add_argument("--dataset-id", required=True, help="Dataset id label")
    p.add_argument("--sample-size", type=int, default=32)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--sampling-strategy", choices=["random", "first_n"], default="random")
    p.add_argument("--copy", action="store_true", help="Copy files instead of symlink")
    p.add_argument("--pixel-size-angstrom", type=float, default=None)
    p.add_argument("--particle-diameter-px", type=float, default=None)
    return p


def main() -> int:
    args = build_parser().parse_args()
    manifest = prepare_cryoppp_topaz_smoke(
        dataset_root=Path(args.dataset_root),
        run_root=Path(args.run_root),
        dataset_id=str(args.dataset_id),
        sample_size=int(args.sample_size),
        random_seed=int(args.random_seed),
        sampling_strategy=str(args.sampling_strategy),
        use_symlink=not bool(args.copy),
        pixel_size_angstrom_override=args.pixel_size_angstrom,
        particle_diameter_px_override=args.particle_diameter_px,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

