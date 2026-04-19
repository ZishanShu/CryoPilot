"""Lightweight image conversion helpers for smoke-run preprocessing."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def convert_jpg_to_png(src_path: Path, dst_path: Path) -> Path:
    """Convert one JPEG image to PNG while preserving stem."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as img:
        img.save(dst_path, format="PNG")
    return dst_path

