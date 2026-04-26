"""Picking benchmark suite v1.

Case definitions, threshold-sweep expansion, a runner that calls
Scheduler.run_next per case, and summary writers (CSV / JSON / Markdown).
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

BENCHMARK_VERSION = "1.0.0"

_CSV_COLUMNS: List[str] = [
    "case_id",
    "dataset_id",
    "backend",
    "threshold",
    "status",
    "return_code",
    "runtime_sec",
    "n_picks",
    "n_micrographs_with_picks",
    "n_micrographs_total",
    "zero_pick_micrograph_ratio",
    "precision",
    "recall",
    "f1",
    "qc_status",
    "qc_flags",
    "recommended_next_action",
    "topaz_output_path",
    "error_message",
]


@dataclass
class PickingBenchmarkCase:
    """Configuration for a single picking benchmark case."""

    case_id: str
    dataset_id: str = ""
    project_path: str = ""
    micrographs_star_path: str = ""
    gt_coords_dir: str = ""
    picking_backend: str = "topaz"
    topaz_threshold: Optional[float] = None
    topaz_thresholds: List[float] = field(default_factory=list)
    topaz_device: Optional[str] = None
    coordinate_scale: float = 1.0
    matching_tolerance_px: float = 80.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PickingBenchmarkCase":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class PickingBenchmarkCaseResult:
    """Per-case benchmark result record."""

    case_id: str
    dataset_id: str = ""
    backend: str = ""
    threshold: Optional[float] = None
    status: str = ""
    return_code: Optional[int] = None
    runtime_sec: float = 0.0
    n_picks: Optional[int] = None
    n_micrographs_with_picks: Optional[int] = None
    n_micrographs_total: Optional[int] = None
    zero_pick_micrograph_ratio: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    qc_status: str = ""
    qc_flags: List[str] = field(default_factory=list)
    recommended_next_action: str = ""
    topaz_output_path: str = ""
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Flat dict for CSV — qc_flags is pipe-joined."""
        d = asdict(self)
        d["qc_flags"] = "|".join(self.qc_flags)
        return d

    def to_full_dict(self) -> Dict[str, Any]:
        """Full dict for JSON — qc_flags is a list."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Threshold sweep helpers
# ---------------------------------------------------------------------------


def _threshold_label(t: float) -> str:
    """Format a threshold as a safe identifier segment, e.g. -8.0 → 'minus8'."""
    if t == int(t):
        i = int(t)
        return f"minus{abs(i)}" if i < 0 else str(i)
    return str(t).replace("-", "minus").replace(".", "p")


def expand_threshold_sweep(case: PickingBenchmarkCase) -> List[PickingBenchmarkCase]:
    """Expand a case that has topaz_thresholds into one case per threshold.

    When topaz_thresholds is empty the original case is returned unchanged.
    Each expanded case gets ``topaz_threshold`` set and a ``_thresh_{label}``
    suffix appended to its case_id.
    """
    thresholds = list(case.topaz_thresholds or [])
    if not thresholds:
        return [case]
    expanded: List[PickingBenchmarkCase] = []
    for t in thresholds:
        label = _threshold_label(float(t))
        expanded.append(
            PickingBenchmarkCase(
                case_id=f"{case.case_id}_thresh_{label}",
                dataset_id=case.dataset_id,
                project_path=case.project_path,
                micrographs_star_path=case.micrographs_star_path,
                gt_coords_dir=case.gt_coords_dir,
                picking_backend=case.picking_backend,
                topaz_threshold=float(t),
                topaz_thresholds=[],
                topaz_device=case.topaz_device,
                coordinate_scale=case.coordinate_scale,
                matching_tolerance_px=case.matching_tolerance_px,
                notes=case.notes,
            )
        )
    return expanded


# ---------------------------------------------------------------------------
# Case loader
# ---------------------------------------------------------------------------


def load_cases_from_json(path: Path) -> List[PickingBenchmarkCase]:
    """Load benchmark cases from a JSON file.

    Accepts a JSON list, a dict with a ``"cases"`` key, or a single case dict.
    Unknown fields in each case dict are silently ignored.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        case_dicts = data
    elif isinstance(data, dict) and "cases" in data:
        case_dicts = list(data["cases"])
    elif isinstance(data, dict):
        case_dicts = [data]
    else:
        raise ValueError(f"Unexpected JSON structure in {path}")
    return [PickingBenchmarkCase.from_dict(d) for d in case_dicts]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class PickingBenchmarkRunner:
    """Runs picking benchmark cases by calling Scheduler.run_next per case."""

    def __init__(self, *, scheduler_factory: Optional[Callable[..., Any]] = None) -> None:
        self._scheduler_factory = scheduler_factory

    def _make_scheduler(self, project_root: Path) -> Any:
        if self._scheduler_factory is not None:
            return self._scheduler_factory(project_root=project_root)
        from app.scheduler import Scheduler  # deferred to avoid circular at import time
        return Scheduler(project_root=project_root)

    def run_case(
        self, case: PickingBenchmarkCase, *, dry_run: bool = False
    ) -> PickingBenchmarkCaseResult:
        start = time.time()
        picking_overrides: Dict[str, Any] = {}
        if case.micrographs_star_path:
            picking_overrides["micrographs_star_path"] = case.micrographs_star_path
        if case.topaz_threshold is not None:
            picking_overrides["topaz_threshold"] = case.topaz_threshold
        if case.topaz_device is not None:
            picking_overrides["topaz_device"] = case.topaz_device
        picking_qc_params: Dict[str, Any] = {
            "coordinate_scale": case.coordinate_scale,
            "matching_tolerance_px": case.matching_tolerance_px,
        }
        if case.gt_coords_dir:
            picking_qc_params["gt_coords_dir"] = case.gt_coords_dir

        try:
            sched = self._make_scheduler(Path(case.project_path))
            _action, result = sched.run_next(
                dry_run=dry_run,
                preferred_picking_backend=case.picking_backend,
                picking_overrides=picking_overrides or None,
                picking_qc_params=picking_qc_params,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - start
            return PickingBenchmarkCaseResult(
                case_id=case.case_id,
                dataset_id=case.dataset_id,
                backend=case.picking_backend,
                threshold=case.topaz_threshold,
                status="error",
                runtime_sec=round(elapsed, 4),
                error_message=str(exc),
            )

        elapsed = time.time() - start
        qc: Dict[str, Any] = result.get("picking_qc") or {}
        gt_eval: Dict[str, Any] = qc.get("ground_truth_evaluation") or {}
        return PickingBenchmarkCaseResult(
            case_id=case.case_id,
            dataset_id=case.dataset_id,
            backend=case.picking_backend,
            threshold=case.topaz_threshold,
            status=str(result.get("status") or ""),
            return_code=result.get("return_code"),
            runtime_sec=round(elapsed, 4),
            n_picks=qc.get("n_picks"),
            n_micrographs_with_picks=qc.get("n_micrographs_with_picks"),
            n_micrographs_total=qc.get("n_micrographs_total"),
            zero_pick_micrograph_ratio=qc.get("zero_pick_micrograph_ratio"),
            precision=gt_eval.get("precision"),
            recall=gt_eval.get("recall"),
            f1=gt_eval.get("f1"),
            qc_status=str(qc.get("qc_status") or ""),
            qc_flags=list(qc.get("qc_flags") or []),
            recommended_next_action=str(qc.get("recommended_next_action") or ""),
            topaz_output_path=str(result.get("topaz_output_path") or ""),
            error_message=str(result.get("error_message") or ""),
        )

    def run_cases(
        self, cases: List[PickingBenchmarkCase], *, dry_run: bool = False
    ) -> Dict[str, Any]:
        results = [self.run_case(c, dry_run=dry_run) for c in cases]
        return {
            "benchmark_version": BENCHMARK_VERSION,
            "case_count": len(results),
            "qc_passed": sum(1 for r in results if r.qc_status == "passed"),
            "qc_failed": sum(1 for r in results if r.qc_status == "failed_qc"),
            "errors": sum(
                1 for r in results if r.status in {"error", "failed", "preflight_blocked"}
            ),
            "results": [r.to_full_dict() for r in results],
        }


# ---------------------------------------------------------------------------
# Summary writers
# ---------------------------------------------------------------------------


def write_csv(results: List[Dict[str, Any]], path: Path) -> None:
    """Write result rows as CSV with canonical column order."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            flat: Dict[str, Any] = {}
            for col in _CSV_COLUMNS:
                v = row.get(col)
                if isinstance(v, list):
                    flat[col] = "|".join(str(x) for x in v)
                elif v is None:
                    flat[col] = ""
                else:
                    flat[col] = v
            writer.writerow(flat)


def write_json(summary: Dict[str, Any], path: Path) -> None:
    """Write full summary as pretty-printed JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_markdown(summary: Dict[str, Any], path: Path) -> None:
    """Write a human-readable markdown summary with a results table."""
    lines = [
        "# Picking Benchmark Summary",
        "",
        f"- Version: {summary.get('benchmark_version', BENCHMARK_VERSION)}",
        f"- Cases: {summary.get('case_count', 0)}",
        f"- QC Passed: {summary.get('qc_passed', 0)}",
        f"- QC Failed: {summary.get('qc_failed', 0)}",
        f"- Errors: {summary.get('errors', 0)}",
        "",
        "## Results",
        "",
    ]
    table_cols = [
        "case_id", "backend", "threshold", "status",
        "n_picks", "f1", "qc_status", "error_message",
    ]
    lines.append("| " + " | ".join(table_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(table_cols)) + " |")
    for r in summary.get("results", []):
        row_vals: List[str] = []
        for col in table_cols:
            v = r.get(col)
            if v is None:
                row_vals.append("")
            elif isinstance(v, float):
                row_vals.append(f"{v:.4f}")
            else:
                row_vals.append(str(v))
        lines.append("| " + " | ".join(row_vals) + " |")
    lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_outputs(
    summary: Dict[str, Any],
    out_dir: Path,
    *,
    stem: str = "benchmark_summary",
) -> Dict[str, Path]:
    """Write benchmark_summary.{csv,json,md} and latest.{json,md}; return paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    latest_json = out_dir / "latest.json"
    latest_md = out_dir / "latest.md"
    write_csv(summary.get("results", []), csv_path)
    write_json(summary, json_path)
    write_markdown(summary, md_path)
    write_json(summary, latest_json)
    write_markdown(summary, latest_md)
    return {
        "csv": csv_path,
        "json": json_path,
        "md": md_path,
        "latest_json": latest_json,
        "latest_md": latest_md,
    }
