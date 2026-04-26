"""CLI entrypoint for cryoem_relion_agent."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import typer

from agent.structure_router import ModelingContext
from app.scheduler import Scheduler
from memory.case_store import CaseStore

app = typer.Typer(help="RELION-based cryo-EM orchestration agent CLI.")


def setup_logging(verbose: bool = False) -> None:
    """Configure root logger."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def validate_sequence_path(sequence: Optional[Path]) -> Optional[Path]:
    """If ``--sequence`` is set, ensure the path exists and is a file."""
    if sequence is None:
        return None
    if not sequence.exists():
        typer.secho(f"Error: sequence file does not exist: {sequence}", err=True)
        raise typer.Exit(code=1)
    if not sequence.is_file():
        typer.secho(f"Error: --sequence must point to a regular file: {sequence}", err=True)
        raise typer.Exit(code=1)
    return sequence


@app.command("scan-project")
def scan_project(
    project: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    sequence: Optional[Path] = typer.Option(
        None,
        "--sequence",
        help="Path to FASTA/sequence file; stored on SampleState.sequence_path for this run.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Scan RELION project and print state + QC."""
    setup_logging(verbose)
    validate_sequence_path(sequence)
    scheduler = Scheduler(project_root=project)
    state, qc = scheduler.scan_project(sequence_override=sequence)
    typer.echo(json.dumps({"state": state.model_dump(mode="json"), "qc": qc.model_dump(mode="json")}, indent=2))


@app.command("decide-next")
def decide_next(
    project: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    sequence: Optional[Path] = typer.Option(
        None,
        "--sequence",
        help="Path to FASTA/sequence file; overrides persisted sequence for this run.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Decide next action from current project status."""
    setup_logging(verbose)
    validate_sequence_path(sequence)
    scheduler = Scheduler(project_root=project)
    action = scheduler.decide_next(sequence_override=sequence)
    typer.echo(json.dumps(action.model_dump(mode="json"), indent=2))


@app.command("run-next")
def run_next(
    project: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print planned commands and paths without executing subprocess.",
    ),
    sequence: Optional[Path] = typer.Option(
        None,
        "--sequence",
        help="Path to FASTA/sequence file for this run.",
    ),
    picking_backend: Optional[str] = typer.Option(
        None,
        "--picking-backend",
        help="Picking backend override: relion_log | topaz | cryolo.",
    ),
    micrographs_star: Optional[Path] = typer.Option(
        None,
        "--micrographs-star",
        help="Explicit micrographs.star input for picking (Topaz etc.).",
    ),
    particle_diameter_angstrom: Optional[float] = typer.Option(
        None,
        "--particle-diameter-angstrom",
        help="Expected particle diameter in angstroms.",
    ),
    topaz_threshold: Optional[float] = typer.Option(
        None,
        "--topaz-threshold",
        help="Topaz picking score threshold (e.g. -6).",
    ),
    topaz_device: Optional[str] = typer.Option(
        None,
        "--topaz-device",
        help='Topaz device spec (e.g. "cpu", "0", "1").',
    ),
    topaz_output_dir: Optional[Path] = typer.Option(
        None,
        "--topaz-output-dir",
        help="Override output directory for the Topaz run.",
    ),
    gt_coords_dir: Optional[Path] = typer.Option(
        None,
        "--gt-coords-dir",
        help="Directory of CryoPPP-style ground-truth coordinate CSVs (one per micrograph). Enables GT evaluation in picking QC.",
    ),
    coordinate_scale: Optional[float] = typer.Option(
        None,
        "--coordinate-scale",
        help="Scale predicted picks to GT coordinate space (e.g. 4.0 for ds4 Topaz). Default 1.0.",
    ),
    matching_tolerance_px: Optional[float] = typer.Option(
        None,
        "--matching-tolerance-px",
        help="Tolerance for greedy GT matching, in pixels. Default 80.0.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Decide and run next action."""
    setup_logging(verbose)
    validate_sequence_path(sequence)
    picking_overrides: Dict[str, Any] = {}
    if micrographs_star is not None:
        picking_overrides["micrographs_star_path"] = str(micrographs_star)
    if particle_diameter_angstrom is not None:
        picking_overrides["particle_diameter_angstrom"] = float(particle_diameter_angstrom)
    if topaz_threshold is not None:
        picking_overrides["topaz_threshold"] = float(topaz_threshold)
    if topaz_device is not None:
        picking_overrides["topaz_device"] = str(topaz_device)
    if topaz_output_dir is not None:
        picking_overrides["output_dir"] = str(topaz_output_dir)
    picking_qc_params: Dict[str, Any] = {}
    if gt_coords_dir is not None:
        picking_qc_params["gt_coords_dir"] = str(gt_coords_dir)
    if coordinate_scale is not None:
        picking_qc_params["coordinate_scale"] = float(coordinate_scale)
    if matching_tolerance_px is not None:
        picking_qc_params["matching_tolerance_px"] = float(matching_tolerance_px)
    scheduler = Scheduler(project_root=project)
    action, result = scheduler.run_next(
        dry_run=dry_run,
        sequence_override=sequence,
        preferred_picking_backend=(str(picking_backend).strip() if picking_backend else None),
        picking_overrides=picking_overrides or None,
        picking_qc_params=picking_qc_params or None,
    )
    typer.echo(json.dumps({"action": action.model_dump(mode="json"), "execution": result}, indent=2))


@app.command("route-structure")
def route_structure(
    project: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    known_sequence: bool = typer.Option(..., help="Whether sequence is known."),
    chain_count: Optional[int] = typer.Option(None, help="Estimated chain count."),
    af3_available: bool = typer.Option(False, help="Whether AF3 is available."),
    complex_type: Optional[str] = typer.Option(None, help="Complex type hint."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Route structure building tool based on context and QC."""
    setup_logging(verbose)
    scheduler = Scheduler(project_root=project)
    context = ModelingContext(
        known_sequence=known_sequence,
        chain_count=chain_count,
        af3_available=af3_available,
        complex_type=complex_type,
    )
    action = scheduler.route_structure(context=context)
    typer.echo(json.dumps(action.model_dump(mode="json"), indent=2))


@app.command("generate-report")
def generate_report(
    project: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    out: Path = typer.Option(..., help="Output markdown path."),
    out_json: Optional[Path] = typer.Option(None, help="Optional output JSON path."),
    sequence: Optional[Path] = typer.Option(
        None,
        "--sequence",
        help="Path to FASTA/sequence file; used for structure routing in the report.",
    ),
    refresh_state: bool = typer.Option(
        False,
        "--refresh-state",
        help="Write scanned sample state to sample_state.json (default: read-only report).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Generate markdown/json report."""
    setup_logging(verbose)
    validate_sequence_path(sequence)
    scheduler = Scheduler(project_root=project)
    report = scheduler.generate_report(
        out_json=out_json,
        out_md=out,
        sequence_override=sequence,
        refresh_state=refresh_state,
    )
    typer.echo(f"Report generated. stage={report.state.stage}, next_action={report.next_action.name.value if report.next_action else 'N/A'}")


@app.command("run-picking-benchmark")
def run_picking_benchmark(
    cases_json: Path = typer.Option(
        ...,
        "--cases-json",
        help="Path to benchmark cases JSON (list or {\"cases\": [...]}).",
    ),
    out_dir: Path = typer.Option(..., "--out-dir", help="Output directory for benchmark results."),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Dry-run each case without executing subprocesses.",
    ),
    threshold_sweep: Optional[str] = typer.Option(
        None,
        "--threshold-sweep",
        help="Comma-separated Topaz thresholds to sweep, e.g. '-8,-7,-6,-5,-4'. Overrides topaz_thresholds in cases JSON.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run picking benchmark suite from a cases JSON file."""
    setup_logging(verbose)
    from agent.picking_benchmark import (
        PickingBenchmarkRunner,
        expand_threshold_sweep,
        load_cases_from_json,
        write_outputs,
    )

    cases = load_cases_from_json(cases_json)
    if threshold_sweep:
        sweep_thresholds = [float(t.strip()) for t in threshold_sweep.split(",") if t.strip()]
        if sweep_thresholds:
            for c in cases:
                if c.picking_backend == "topaz":
                    c.topaz_thresholds = sweep_thresholds

    expanded: list = []
    for c in cases:
        expanded.extend(expand_threshold_sweep(c))

    runner = PickingBenchmarkRunner()
    summary = runner.run_cases(expanded, dry_run=dry_run)
    paths = write_outputs(summary, out_dir)
    typer.echo(
        json.dumps(
            {
                "case_count": summary["case_count"],
                "qc_passed": summary["qc_passed"],
                "qc_failed": summary["qc_failed"],
                "errors": summary["errors"],
                "outputs": {k: str(v) for k, v in paths.items()},
            },
            indent=2,
        )
    )


@app.command("init-db")
def init_db(
    db_path: Path = typer.Option(Path("cases.db"), help="Path to sqlite database."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Initialize sqlite case memory database."""
    setup_logging(verbose)
    store = CaseStore(db_path)
    ok = store.init_db()
    if ok:
        typer.echo(f"Initialized DB: {db_path}")
    else:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
