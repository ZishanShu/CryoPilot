import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_cli_plan_only(tmp_path):
    fixture = REPO / "examples/evidence_cases/empiar10017_full84"
    if not fixture.exists():
        return
    out_root = tmp_path / "cl_out"
    cmd = [
        sys.executable, "-m", "scripts.run_closed_loop_rerun_selection",
        "--baseline-root", str(fixture),
        "--out-root", str(out_root),
        "--dataset-id", "EMPIAR-10017",
        "--map2pdb-mode", "dry-run",
        "--run-llm", "false",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (out_root / "retrieval/candidate_strategies.json").exists()
    assert (out_root / "selection/rerun_plan.json").exists()
    assert (out_root / "comparison/final_selection.json").exists()
    assert (out_root / "closed_loop_report.md").exists()
