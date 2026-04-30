"""RELION LoG/autopick runner for the picker fallback stage.

Materialises a minimal RELION project layout from a pairs CSV
(micrograph_path, gt_coord_path), optionally executes relion_autopick in
LoG mode, converts outputs to coord files, and evaluates F1 against GT
using the existing picking evaluator.

Stdlib-only (no numpy/pandas).  Does not require raw MRC/MRCS files or GPU
for dry-run; test isolation is achieved by passing ``dry_run=True`` and
providing a command runner callable.
"""

from __future__ import annotations

import csv
import json
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.eval.picking_eval import evaluate_picking_predictions

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

CommandRunner = Callable[[List[str], Path], Tuple[int, str, str]]


def _default_runner(cmd: List[str], cwd: Path) -> Tuple[int, str, str]:
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=3600,
    )
    return result.returncode, result.stdout, result.stderr


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RelionLogAutopickConfig:
    """Configuration for one RELION LoG autopick run.

    Parameters
    ----------
    relion_autopick_bin:
        Path to the ``relion_autopick`` (or ``relion_autopick_mpi``) binary.
        Required for real runs; ignored in dry-run.
    output_root:
        Root directory for all outputs.  Sub-directories are created
        automatically.
    pairs_csv:
        CSV with columns ``micrograph_path`` and ``gt_coord_path`` (one row
        per micrograph).
    dry_run:
        When True the autopick command is built and recorded but not executed.
        Predicted picks are left empty so F1 is 0 (STOP is preserved if GT F1
        also stays below threshold).
    limit:
        If set, only the first ``limit`` micrographs are processed.
    particle_diameter_angstrom:
        LoG particle diameter hint (Å).
    pixel_size_angstrom:
        Pixel size (Å/px) written into the optics table.
    voltage_kv:
        Accelerating voltage for the optics table.
    cs_mm:
        Spherical aberration (mm) for the optics table.
    amplitude_contrast:
        Amplitude contrast for the optics table.
    match_radius_px:
        Matching radius (px) for F1 evaluation.
    command_runner:
        Callable ``(cmd: List[str], cwd: Path) -> (returncode, stdout, stderr)``.
        Defaults to ``subprocess.run``.  Override in tests to mock execution.
    """

    relion_autopick_bin: Optional[Path] = None
    output_root: Path = Path("reruns/relion_log_autopick")
    pairs_csv: Optional[Path] = None
    dry_run: bool = True
    limit: Optional[int] = None
    particle_diameter_angstrom: float = 120.0
    pixel_size_angstrom: float = 1.0
    voltage_kv: float = 300.0
    cs_mm: float = 2.7
    amplitude_contrast: float = 0.1
    match_radius_px: float = 80.0
    command_runner: CommandRunner = field(default_factory=lambda: _default_runner, repr=False)


# ---------------------------------------------------------------------------
# STAR writers
# ---------------------------------------------------------------------------


_OPTICS_TEMPLATE = """\
data_optics

loop_
_rlnOpticsGroupName #1
_rlnOpticsGroup #2
_rlnMicrographPixelSize #3
_rlnVoltage #4
_rlnSphericalAberration #5
_rlnAmplitudeContrast #6
opticsGroup1 1 {pixel_size} {voltage} {cs} {ac}

data_micrographs

loop_
_rlnOpticsGroup #1
_rlnMicrographName #2
"""


def _write_micrographs_star(
    path: Path,
    mic_names: List[str],
    *,
    pixel_size: float = 1.0,
    voltage: float = 300.0,
    cs: float = 2.7,
    ac: float = 0.1,
) -> None:
    """Write a RELION-4-compatible micrographs.star with an optics table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = _OPTICS_TEMPLATE.format(
        pixel_size=pixel_size,
        voltage=voltage,
        cs=cs,
        ac=ac,
    )
    rows = "\n".join(f"1 {name}" for name in mic_names)
    path.write_text(header + rows + "\n", encoding="utf-8")


def _write_coord_star(path: Path, coords: List[Tuple[float, float]]) -> None:
    """Write a RELION coordinate STAR file from an x/y list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "data_",
        "",
        "loop_",
        "_rlnCoordinateX #1",
        "_rlnCoordinateY #2",
    ]
    for x, y in coords:
        lines.append(f"{x:.2f} {y:.2f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_simple_coord(path: Path, coords: List[Tuple[float, float]]) -> None:
    """Write a plain ``x y`` coord file for the F1 evaluator."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{x:.2f} {y:.2f}" for x, y in coords]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Parsing RELION autopick output STAR files
# ---------------------------------------------------------------------------


def _parse_relion_coord_star(path: Path) -> List[Tuple[float, float]]:
    """Parse a per-micrograph coordinate STAR file produced by relion_autopick."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    in_loop = False
    x_idx = y_idx = None
    coords: List[Tuple[float, float]] = []
    columns: List[str] = []
    data_start = False
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("loop_"):
            in_loop = True
            columns = []
            data_start = False
            continue
        if in_loop and s.startswith("_"):
            tok = s.split()[0].lstrip("_").lower()
            columns.append(tok)
            continue
        if in_loop and columns and not data_start:
            # First non-column, non-empty line after loop_ header = data
            data_start = True
            x_idx = next((i for i, c in enumerate(columns) if "coordinatex" in c), None)
            y_idx = next((i for i, c in enumerate(columns) if "coordinatey" in c), None)
        if data_start and x_idx is not None and y_idx is not None:
            parts = s.split()
            if len(parts) > max(x_idx, y_idx):
                try:
                    coords.append((float(parts[x_idx]), float(parts[y_idx])))
                except ValueError:
                    continue
    return coords


# ---------------------------------------------------------------------------
# CSV pairs reader
# ---------------------------------------------------------------------------


# Column-name aliases.  Order matters: earlier names take priority.
_MIC_ID_KEYS = (
    "micrograph_id",
    "image_name",
    "micrograph_name",
    "micrograph",
    "id",
)
_MIC_PATH_KEYS = (
    "local_mrc_path",
    "mrc_path",
    "micrograph_path",
    "local_micrograph_path",
    "image_path",
    "path",
)
_GT_PATH_KEYS = (
    "local_coord_path",
    "gt_coord_path",
    "gt_path",
    "coord_path",
    "ground_truth_coord_path",
    "gt_coordinate_path",
    "manual_coord_path",
)


def _pick_field(row: Dict[str, str], keys: Tuple[str, ...]) -> str:
    """Return the first non-empty value among ``keys`` (case-insensitive)."""
    lower = {k.lower(): v for k, v in row.items() if k}
    for key in keys:
        v = lower.get(key.lower())
        if v is not None:
            s = str(v).strip()
            if s:
                return s
    return ""


def _read_pairs_csv(
    path: Path, limit: Optional[int] = None
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Read a flexible pairs CSV with column aliasing.

    Returns ``(usable_pairs, diagnostics)`` where ``diagnostics`` is a JSON-
    serialisable dict ready to persist as ``pairs_csv_diagnostics.json``.

    A row is *usable* when both ``micrograph_id`` and a micrograph path are
    resolvable from the recognised aliases.  GT coords are optional (so dry-run
    materialisation works without ground-truth files).
    """
    usable: List[Dict[str, str]] = []
    n_raw = 0
    dropped_reasons: Dict[str, int] = {}
    fieldnames: List[str] = []
    first_raw_row: Optional[Dict[str, str]] = None
    first_usable_row: Optional[Dict[str, str]] = None

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            n_raw += 1
            if first_raw_row is None:
                first_raw_row = {k: row.get(k, "") for k in fieldnames}

            mic_id = _pick_field(row, _MIC_ID_KEYS)
            mic_path = _pick_field(row, _MIC_PATH_KEYS)
            gt_path = _pick_field(row, _GT_PATH_KEYS)

            if not mic_id and not mic_path:
                dropped_reasons["missing_id_and_path"] = (
                    dropped_reasons.get("missing_id_and_path", 0) + 1
                )
                continue
            if not mic_path:
                dropped_reasons["missing_micrograph_path"] = (
                    dropped_reasons.get("missing_micrograph_path", 0) + 1
                )
                continue
            if not mic_id:
                # Synthesise an id from the path so downstream code has a stem.
                mic_id = Path(mic_path).stem

            entry = {
                "micrograph_id": mic_id,
                "micrograph_path": mic_path,
                "gt_coord_path": gt_path,
            }
            usable.append(entry)
            if first_usable_row is None:
                first_usable_row = dict(entry)
            if limit is not None and len(usable) >= limit:
                break

    diagnostics: Dict[str, Any] = {
        "fieldnames": fieldnames,
        "n_rows_raw": n_raw,
        "n_rows_usable": len(usable),
        "n_rows_dropped": n_raw - len(usable),
        "dropped_reasons": dropped_reasons,
        "first_raw_row": first_raw_row,
        "first_usable_row": first_usable_row,
    }
    return usable, diagnostics


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------


def _build_autopick_cmd(
    relion_bin: Path,
    micrographs_star: Path,
    output_root: Path,
    *,
    diameter_angstrom: float = 120.0,
    pixel_size: float = 1.0,
) -> List[str]:
    """Build the relion_autopick LoG command."""
    # Convert diameter from Å to px for the --LoG_diam_min/max range (±25%).
    diam_px = diameter_angstrom / pixel_size
    diam_min_px = round(diam_px * 0.75, 1)
    diam_max_px = round(diam_px * 1.25, 1)
    return [
        str(relion_bin),
        "--i", str(micrographs_star),
        "--odir", str(output_root / "AutoPick" / "job001"),
        "--pickname", "autopick",
        "--LoG",
        "--LoG_diam_min", str(diam_min_px),
        "--LoG_diam_max", str(diam_max_px),
        "--lowpass", "20",
        "--LoG_adjust_threshold", "0",
        "--pipeline_control", str(output_root / "AutoPick" / "job001"),
    ]


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_relion_log_autopick(
    config: RelionLogAutopickConfig,
) -> Dict[str, Any]:
    """Materialise a minimal RELION project, optionally run autopick, evaluate F1.

    Returns a JSON-serialisable dict with keys:

    - ``mode``: ``"dry_run"`` or ``"real_run"``
    - ``f1``: float or None
    - ``precision``, ``recall``: float or None
    - ``predicted_count``, ``gt_count``: int
    - ``command``: List[str] — the autopick command (built even in dry-run)
    - ``returncode``: int or None
    - ``stdout``, ``stderr``: str or None
    - ``output_root``: str
    - ``evaluation_available``: bool
    - ``error``: str or None
    """
    out = Path(config.output_root)
    out.mkdir(parents=True, exist_ok=True)

    if config.pairs_csv is None or not Path(config.pairs_csv).exists():
        return _error_bundle(out, "pairs_csv missing or not found", config.dry_run)

    try:
        pairs, diagnostics = _read_pairs_csv(Path(config.pairs_csv), limit=config.limit)
    except Exception as exc:  # noqa: BLE001
        return _error_bundle(out, f"pairs_csv read error: {exc}", config.dry_run)

    # Persist diagnostics regardless of whether the parse produced usable rows,
    # so failures can be debugged from the output bundle alone.
    (out / "pairs_csv_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
    )

    if diagnostics["n_rows_raw"] == 0:
        return _error_bundle(out, "pairs_csv is empty", config.dry_run)
    if diagnostics["n_rows_usable"] == 0:
        return _error_bundle(
            out,
            "pairs_csv has rows but no usable micrograph paths",
            config.dry_run,
        )

    # --- Materialise project layout -----------------------------------------
    mics_dir = out / "Micrographs"
    mics_dir.mkdir(parents=True, exist_ok=True)
    autopick_dir = out / "AutoPick" / "job001"
    autopick_dir.mkdir(parents=True, exist_ok=True)

    # Use basename of each micrograph as the canonical name in the project.
    mic_names: List[str] = []
    for p in pairs:
        name = Path(p["micrograph_path"]).name
        # Symlink only if source exists; otherwise write a placeholder text
        # file so the project layout is valid for dry-run.
        src = Path(p["micrograph_path"])
        dst = mics_dir / name
        if src.exists() and not dst.exists():
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                dst.write_text("placeholder\n", encoding="utf-8")
        elif not dst.exists():
            dst.write_text("placeholder\n", encoding="utf-8")
        mic_names.append(f"Micrographs/{name}")

    micrographs_star = out / "micrographs.star"
    _write_micrographs_star(
        micrographs_star,
        mic_names,
        pixel_size=config.pixel_size_angstrom,
        voltage=config.voltage_kv,
        cs=config.cs_mm,
        ac=config.amplitude_contrast,
    )

    # --- Build command -------------------------------------------------------
    relion_bin = (
        config.relion_autopick_bin
        if config.relion_autopick_bin is not None
        else Path("relion_autopick")
    )
    cmd = _build_autopick_cmd(
        relion_bin,
        micrographs_star,
        out,
        diameter_angstrom=config.particle_diameter_angstrom,
        pixel_size=config.pixel_size_angstrom,
    )

    # --- Persist the command for auditability --------------------------------
    (out / "autopick_command.txt").write_text(
        shlex.join(cmd) + "\n", encoding="utf-8"
    )

    returncode: Optional[int] = None
    stdout_text: Optional[str] = None
    stderr_text: Optional[str] = None
    run_error: Optional[str] = None

    if not config.dry_run:
        t0 = time.monotonic()
        try:
            returncode, stdout_text, stderr_text = config.command_runner(cmd, out)
        except Exception as exc:  # noqa: BLE001
            run_error = f"command_runner raised: {exc}"
            returncode = -1
        elapsed = time.monotonic() - t0
        (out / "autopick_stdout.txt").write_text(stdout_text or "", encoding="utf-8")
        (out / "autopick_stderr.txt").write_text(stderr_text or "", encoding="utf-8")
    else:
        elapsed = 0.0

    # --- Collect predicted coordinates per micrograph -----------------------
    pred_coord_path = out / "all_predicted_coords.coord"
    all_preds: List[Tuple[float, float]] = []

    if not config.dry_run and run_error is None and returncode == 0:
        for p in pairs:
            stem = Path(p["micrograph_path"]).stem
            # relion_autopick writes <stem>_autopick.star into the odir
            star_path = autopick_dir / f"{stem}_autopick.star"
            coords = _parse_relion_coord_star(star_path)
            all_preds.extend(coords)
    # dry_run or failed run → all_preds stays empty (STOP preserved)

    _write_simple_coord(pred_coord_path, all_preds)

    # --- Collect GT coordinate files ----------------------------------------
    gt_files: List[Path] = []
    for p in pairs:
        gt = p.get("gt_coord_path", "")
        if gt:
            gt_path = Path(gt)
            if gt_path.exists():
                gt_files.append(gt_path)

    # --- Evaluate F1 --------------------------------------------------------
    eval_result = evaluate_picking_predictions(
        predicted_coord_file=pred_coord_path,
        gt_coord_files=gt_files,
        match_radius_px=config.match_radius_px,
    )

    bundle: Dict[str, Any] = {
        "mode": "dry_run" if config.dry_run else "real_run",
        "f1": eval_result["f1"] if eval_result.get("evaluation_available") else None,
        "precision": eval_result.get("precision"),
        "recall": eval_result.get("recall"),
        "predicted_count": eval_result.get("predicted_count", 0),
        "gt_count": eval_result.get("gt_count", 0),
        "matched_count": eval_result.get("matched_count", 0),
        "match_radius_px": config.match_radius_px,
        "command": cmd,
        "returncode": returncode,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "runtime_sec": round(elapsed, 3),
        "output_root": str(out),
        "micrographs_star": str(micrographs_star),
        "pred_coord_path": str(pred_coord_path),
        "n_micrographs": len(pairs),
        "evaluation_available": bool(eval_result.get("evaluation_available")),
        "error": run_error,
    }

    # --- Persist bundle ------------------------------------------------------
    (out / "relion_log_autopick_result.json").write_text(
        json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_result_md(out / "relion_log_autopick_result.md", bundle)

    return bundle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error_bundle(out: Path, error: str, dry_run: bool) -> Dict[str, Any]:
    bundle: Dict[str, Any] = {
        "mode": "dry_run" if dry_run else "real_run",
        "f1": None,
        "precision": None,
        "recall": None,
        "predicted_count": 0,
        "gt_count": 0,
        "matched_count": 0,
        "command": [],
        "returncode": None,
        "stdout": None,
        "stderr": None,
        "runtime_sec": 0.0,
        "output_root": str(out),
        "evaluation_available": False,
        "error": error,
    }
    (out / "relion_log_autopick_result.json").write_text(
        json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8"
    )
    return bundle


def _write_result_md(path: Path, bundle: Dict[str, Any]) -> None:
    lines = [
        "# RELION LoG autopick result",
        "",
        f"- mode: {bundle.get('mode')}",
        f"- f1: {bundle.get('f1')}",
        f"- precision: {bundle.get('precision')}",
        f"- recall: {bundle.get('recall')}",
        f"- predicted_count: {bundle.get('predicted_count')}",
        f"- gt_count: {bundle.get('gt_count')}",
        f"- matched_count: {bundle.get('matched_count')}",
        f"- match_radius_px: {bundle.get('match_radius_px')}",
        f"- returncode: {bundle.get('returncode')}",
        f"- runtime_sec: {bundle.get('runtime_sec')}",
        f"- n_micrographs: {bundle.get('n_micrographs')}",
        f"- evaluation_available: {bundle.get('evaluation_available')}",
        f"- error: {bundle.get('error')}",
        "",
        "## Command",
        "```",
        shlex.join(bundle.get("command") or []),
        "```",
        "",
        "Note: This stage does not claim final structure determination.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
