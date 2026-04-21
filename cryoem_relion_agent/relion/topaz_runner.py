"""Real Topaz ``extract`` execution for particle picking (RELION-adjacent workflow)."""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from agent.picking_preflight import effective_micrographs_star, load_topaz_config
from relion.star_parser import iter_star_loop_blocks
from schemas.execution_result import ExecutionResult

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _argv_to_command(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(a)) for a in argv)


class TopazDeviceValueError(ValueError):
    """Raised when a topaz_device value cannot be normalized to int / None."""


def normalize_topaz_device(value: Any) -> Optional[int]:
    """Normalize a topaz_device value to an ``int`` GPU index or ``None`` for CPU.

    Topaz ``extract``'s ``-d/--device`` flag in current versions expects an
    integer GPU index; passing a string like ``"CPU"`` causes argparse to bail
    with ``invalid int value``.  Callers should:

    * pass ``int`` (e.g. ``0``) for an explicit GPU index,
    * pass ``None`` / ``""`` / ``"cpu"`` / ``"CPU"`` to request CPU execution,
      in which case this helper returns ``None`` and the caller must OMIT the
      ``-d`` flag entirely,
    * pass a numeric string (``"0"``, ``"1"``) which is coerced to ``int``.

    Any other non-empty string raises :class:`TopazDeviceValueError` so we
    never silently drop a value the user set intentionally.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        # bool is a subclass of int; treat as an invalid device specifier so
        # a yaml ``true``/``false`` does not silently become device 0 or 1.
        raise TopazDeviceValueError(
            f"topaz_device must be an int or string, got bool {value!r}"
        )
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise TopazDeviceValueError(
            f"topaz_device float must be integral, got {value!r}"
        )
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.lower() == "cpu":
            return None
        try:
            return int(stripped)
        except ValueError as exc:
            raise TopazDeviceValueError(
                f"topaz_device string must be 'cpu' or an int, got {value!r}"
            ) from exc
    raise TopazDeviceValueError(
        f"topaz_device must be int, None, or string; got {type(value).__name__}"
    )


def _resolve_micrograph_paths_from_star(star_path: Path) -> List[str]:
    """Extract micrograph file paths from a STAR file."""
    text = star_path.read_text(encoding="utf-8", errors="ignore")
    out: List[str] = []
    for _data_block, cols, row_lines in iter_star_loop_blocks(text):
        idx = cols.index("_rlnMicrographName") if "_rlnMicrographName" in cols else 0
        for raw in row_lines:
            parts = raw.split()
            if len(parts) <= idx:
                continue
            candidate = parts[idx].strip()
            if not candidate:
                continue
            p = Path(candidate)
            if not p.is_absolute():
                p = (star_path.parent / p).resolve()
            out.append(str(p))
    # Deduplicate while preserving order.
    seen: set[str] = set()
    deduped: List[str] = []
    for item in out:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _compute_topaz_radius(params: Dict[str, Any], config: Dict[str, Any]) -> float:
    dia_px = params.get("particle_diameter_px")
    if dia_px is not None:
        try:
            dp = float(dia_px)
            if dp > 0:
                return max(1.0, dp / 2.0)
        except (TypeError, ValueError):
            pass
    dia = params.get("particle_diameter_angstrom")
    apix = params.get("micrograph_pixel_size_angstrom") or config.get("micrograph_pixel_size_angstrom")
    if apix is not None and dia is not None:
        try:
            ap = float(apix)
            d_ang = float(dia)
            if ap > 0 and d_ang > 0:
                return max(1.0, (d_ang / ap) / 2.0)
        except (TypeError, ValueError):
            pass
    if dia is not None:
        try:
            d_any = float(dia)
            if d_any > 0:
                return max(1.0, d_any / 2.0)
        except (TypeError, ValueError):
            pass
    return 16.0


def build_topaz_extract_argv(
    params: Dict[str, Any],
    config: Dict[str, Any],
    preflight: Dict[str, Any],
) -> Tuple[List[str], str, Dict[str, Any]]:
    """Build argv, primary output STAR path, and metadata."""
    exe = str(preflight["executable_path"])
    mg_path, _mg_key = effective_micrographs_star(params)
    if not mg_path:
        raise ValueError("micrographs star path missing for Topaz argv build")
    mg_star = Path(mg_path)
    if not mg_star.is_file():
        raise ValueError(f"micrographs star file not found: {mg_star}")
    micrograph_paths = _resolve_micrograph_paths_from_star(mg_star)
    if not micrograph_paths:
        raise ValueError(f"no micrograph paths resolved from star: {mg_star}")

    model = params.get("topaz_model_path")
    model_source = "explicit" if model else "default_pretrained"
    out_raw = params.get("picked_particles_star") or params.get("topaz_output_coords") or params.get("topaz_output_path")
    if out_raw and str(out_raw).strip():
        out_coords = str(Path(str(out_raw).strip()))
    else:
        out_coords = str(Path(str(params["output_dir"])) / "picks.star")

    radius = int(round(_compute_topaz_radius(params, config)))
    argv: List[str] = [exe, "extract"]
    if model:
        argv.extend(["-m", str(model)])
    argv.extend(["-r", str(max(1, radius))])

    thr = params.get("topaz_threshold")
    if thr is None and config.get("default_threshold") is not None:
        thr = config["default_threshold"]
    if thr is not None:
        argv.extend(["-t", str(thr)])

    dev_raw = params.get("topaz_device")
    if dev_raw is None or (isinstance(dev_raw, str) and not dev_raw.strip()):
        dev_raw = config.get("default_device")
    dev = normalize_topaz_device(dev_raw)
    if dev is not None:
        argv.extend(["-d", str(dev)])
    argv.extend(["--format", "star", "-o", out_coords])
    argv.extend(micrograph_paths)
    return argv, out_coords, {
        "command_mode": "extract",
        "model_source": model_source,
        "topaz_radius_used": float(max(1, radius)),
        "micrograph_count": len(micrograph_paths),
    }


def run_topaz_pick(
    *,
    params: Dict[str, Any],
    project_dir: Path,
    topaz_config_path: Optional[Path],
    preflight: Dict[str, Any],
    dry_run: bool = True,
    timeout_sec: int = 7200,
    env: Optional[Mapping[str, str]] = None,
) -> ExecutionResult:
    """Run (or dry-run) Topaz extract; verify output file when not dry_run."""
    config = load_topaz_config(topaz_config_path)
    started = _now_iso()
    meta: Dict[str, Any] = {
        "picking_backend": "topaz",
        "execution_mode": "real" if not dry_run else "dry_run",
        "topaz_executable": preflight.get("executable_path"),
        "command_mode": preflight.get("command_mode") or "extract",
        "model_source": preflight.get("model_source") or "default_pretrained",
    }

    try:
        argv, out_coords, build_meta = build_topaz_extract_argv(params, config, preflight)
        meta.update(build_meta)
    except Exception as exc:  # noqa: BLE001
        return ExecutionResult(
            action_name="run_autopick",
            dry_run=dry_run,
            status="failed",
            error_message=f"failed to build topaz argv: {exc}",
            started_at=started,
            finished_at=_now_iso(),
            metadata=meta,
        )

    command = _argv_to_command(argv)
    meta["topaz_output_path"] = out_coords
    meta["topaz_command"] = command

    if dry_run:
        return ExecutionResult(
            action_name="run_autopick",
            command=command,
            command_argv=[str(a) for a in argv],
            dry_run=True,
            status="dry_run",
            return_code=0,
            stdout=f"[DRY-RUN] Would execute: {command}",
            stderr="",
            output_dir=str(params.get("output_dir")),
            started_at=started,
            finished_at=_now_iso(),
            metadata=meta,
        )

    run_env = dict(os.environ) if env is None else {**os.environ, **dict(env)}
    try:
        completed = subprocess.run(
            [str(a) for a in argv],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=run_env,
            check=False,
        )
    except FileNotFoundError as exc:
        return ExecutionResult(
            action_name="run_autopick",
            command=command,
            command_argv=[str(a) for a in argv],
            dry_run=False,
            status="failed",
            return_code=127,
            stdout="",
            stderr=str(exc),
            error_message=f"command not found: {argv[0]}",
            started_at=started,
            finished_at=_now_iso(),
            metadata=meta,
        )
    except subprocess.TimeoutExpired as exc:
        return ExecutionResult(
            action_name="run_autopick",
            command=command,
            command_argv=[str(a) for a in argv],
            dry_run=False,
            status="failed",
            return_code=None,
            stdout=exc.stdout or "",
            stderr=exc.stderr or str(exc),
            error_message=f"topaz timeout after {timeout_sec}s",
            started_at=started,
            finished_at=_now_iso(),
            metadata=meta,
        )

    out_path = Path(out_coords)
    output_ok = out_path.is_file() and out_path.stat().st_size > 0
    success = completed.returncode == 0 and output_ok
    if completed.returncode == 0 and not output_ok:
        meta["warnings"] = ["topaz exited 0 but expected output file missing or empty"]

    status = "success" if success else "failed"
    err = None if success else (
        completed.stderr.strip()
        or completed.stdout.strip()
        or ("missing or empty output file" if not output_ok else f"exit {completed.returncode}")
    )
    return ExecutionResult(
        action_name="run_autopick",
        command=command,
        command_argv=[str(a) for a in argv],
        dry_run=False,
        status=status,
        return_code=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        output_dir=str(params.get("output_dir")),
        generated_paths=[out_coords] if output_ok else [],
        error_message=err,
        started_at=started,
        finished_at=_now_iso(),
        metadata=meta,
    )
