"""Tests for the Picking Benchmark Suite v1.

Covers: case parsing, threshold expansion, QC collection, CSV/JSON/MD output,
failed execution, and no-GT cases.
"""

from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

from agent.picking_benchmark import (
    BENCHMARK_VERSION,
    PickingBenchmarkCase,
    PickingBenchmarkRunner,
    _CSV_COLUMNS,
    expand_threshold_sweep,
    load_cases_from_json,
    write_csv,
    write_json,
    write_markdown,
    write_outputs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_case(**kwargs) -> PickingBenchmarkCase:
    defaults: dict = {
        "case_id": "test_case",
        "dataset_id": "ds1",
        "project_path": "/tmp/p",
        "picking_backend": "topaz",
    }
    defaults.update(kwargs)
    return PickingBenchmarkCase(**defaults)


def _mock_run_next_result(*, with_qc: bool = True, with_gt: bool = True) -> dict:
    result: dict = {
        "status": "success",
        "return_code": 0,
        "topaz_output_path": "/tmp/picks.star",
        "error_message": None,
    }
    if with_qc:
        qc: dict = {
            "qc_status": "passed",
            "qc_flags": [],
            "n_picks": 1200,
            "n_micrographs_with_picks": 48,
            "n_micrographs_total": 50,
            "zero_pick_micrograph_ratio": 0.04,
            "recommended_next_action": "proceed",
        }
        if with_gt:
            qc["ground_truth_evaluation"] = {"precision": 0.85, "recall": 0.72, "f1": 0.78}
        result["picking_qc"] = qc
    return result


def _make_mock_scheduler(result: dict) -> MagicMock:
    action_mock = MagicMock()
    action_mock.name.value = "run_autopick"
    sched = MagicMock()
    sched.run_next.return_value = (action_mock, result)
    return sched


def _two_result_summary() -> dict:
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "case_count": 2,
        "qc_passed": 1,
        "qc_failed": 1,
        "errors": 0,
        "results": [
            {
                "case_id": "c1",
                "dataset_id": "ds1",
                "backend": "topaz",
                "threshold": -6.0,
                "status": "success",
                "return_code": 0,
                "runtime_sec": 1.23,
                "n_picks": 1200,
                "n_micrographs_with_picks": 48,
                "n_micrographs_total": 50,
                "zero_pick_micrograph_ratio": 0.04,
                "precision": 0.85,
                "recall": 0.72,
                "f1": 0.78,
                "qc_status": "passed",
                "qc_flags": [],
                "recommended_next_action": "proceed",
                "topaz_output_path": "/tmp/picks.star",
                "error_message": "",
            },
            {
                "case_id": "c2",
                "dataset_id": "ds1",
                "backend": "topaz",
                "threshold": -8.0,
                "status": "success",
                "return_code": 0,
                "runtime_sec": 2.34,
                "n_picks": 43,
                "n_micrographs_with_picks": 13,
                "n_micrographs_total": 29,
                "zero_pick_micrograph_ratio": 0.55,
                "precision": 0.07,
                "recall": 0.001,
                "f1": 0.001,
                "qc_status": "failed_qc",
                "qc_flags": ["severe_under_picking"],
                "recommended_next_action": "threshold_sweep_or_train_topaz",
                "topaz_output_path": "/tmp/picks2.star",
                "error_message": "",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Case parsing
# ---------------------------------------------------------------------------


class TestLoadCasesFromJson(unittest.TestCase):
    @staticmethod
    def _write(path: Path, data) -> None:
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_list_format(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "cases.json"
            self._write(p, [
                {"case_id": "c1", "dataset_id": "ds1", "project_path": "/p1"},
                {"case_id": "c2", "dataset_id": "ds2", "project_path": "/p2"},
            ])
            cases = load_cases_from_json(p)
            self.assertEqual(len(cases), 2)
            self.assertEqual(cases[0].case_id, "c1")
            self.assertEqual(cases[1].case_id, "c2")

    def test_dict_with_cases_key(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "cases.json"
            self._write(p, {"version": "1", "cases": [{"case_id": "c1", "project_path": "/p1"}]})
            cases = load_cases_from_json(p)
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].case_id, "c1")

    def test_single_case_dict(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "case.json"
            self._write(p, {"case_id": "c1", "project_path": "/p1"})
            cases = load_cases_from_json(p)
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].case_id, "c1")

    def test_unknown_fields_are_ignored(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "cases.json"
            self._write(p, [{"case_id": "c1", "project_path": "/p1", "extra_key": "ignored"}])
            cases = load_cases_from_json(p)
            self.assertEqual(cases[0].case_id, "c1")

    def test_topaz_thresholds_are_parsed(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "cases.json"
            self._write(p, [{"case_id": "c1", "project_path": "/p", "topaz_thresholds": [-8, -6]}])
            cases = load_cases_from_json(p)
            self.assertEqual(cases[0].topaz_thresholds, [-8, -6])


# ---------------------------------------------------------------------------
# Threshold sweep expansion
# ---------------------------------------------------------------------------


class TestExpandThresholdSweep(unittest.TestCase):
    def test_no_thresholds_returns_original_unchanged(self) -> None:
        case = _minimal_case(case_id="base")
        expanded = expand_threshold_sweep(case)
        self.assertEqual(len(expanded), 1)
        self.assertIs(expanded[0], case)

    def test_expands_into_one_case_per_threshold(self) -> None:
        case = _minimal_case(case_id="base", topaz_thresholds=[-8.0, -7.0, -6.0])
        expanded = expand_threshold_sweep(case)
        self.assertEqual(len(expanded), 3)
        self.assertEqual([e.topaz_threshold for e in expanded], [-8.0, -7.0, -6.0])

    def test_expanded_case_ids_are_unique(self) -> None:
        case = _minimal_case(case_id="base", topaz_thresholds=[-8.0, -7.0, -6.0])
        ids = [e.case_id for e in expand_threshold_sweep(case)]
        self.assertEqual(len(ids), len(set(ids)), "expanded case_ids must be unique")

    def test_expanded_cases_clear_topaz_thresholds(self) -> None:
        case = _minimal_case(case_id="base", topaz_thresholds=[-8.0, -6.0])
        for e in expand_threshold_sweep(case):
            self.assertEqual(e.topaz_thresholds, [])

    def test_other_fields_are_preserved(self) -> None:
        case = _minimal_case(
            case_id="base",
            dataset_id="ds42",
            coordinate_scale=4.0,
            matching_tolerance_px=100.0,
            topaz_thresholds=[-6.0],
        )
        e = expand_threshold_sweep(case)[0]
        self.assertEqual(e.dataset_id, "ds42")
        self.assertEqual(e.coordinate_scale, 4.0)
        self.assertEqual(e.matching_tolerance_px, 100.0)

    def test_case_id_contains_threshold_label(self) -> None:
        case = _minimal_case(case_id="base", topaz_thresholds=[-8.0])
        e = expand_threshold_sweep(case)[0]
        self.assertIn("minus8", e.case_id)
        self.assertTrue(e.case_id.startswith("base_"))


# ---------------------------------------------------------------------------
# Runner: QC collection
# ---------------------------------------------------------------------------


class TestPickingBenchmarkRunnerQCCollection(unittest.TestCase):
    def test_qc_fields_extracted_with_gt(self) -> None:
        case = _minimal_case(gt_coords_dir="/tmp/gt")
        sched = _make_mock_scheduler(_mock_run_next_result(with_qc=True, with_gt=True))
        runner = PickingBenchmarkRunner(scheduler_factory=lambda **_: sched)
        r = runner.run_case(case, dry_run=True)
        self.assertEqual(r.status, "success")
        self.assertEqual(r.n_picks, 1200)
        self.assertEqual(r.n_micrographs_with_picks, 48)
        self.assertEqual(r.n_micrographs_total, 50)
        self.assertAlmostEqual(r.zero_pick_micrograph_ratio, 0.04)
        self.assertAlmostEqual(r.precision, 0.85)
        self.assertAlmostEqual(r.recall, 0.72)
        self.assertAlmostEqual(r.f1, 0.78)
        self.assertEqual(r.qc_status, "passed")
        self.assertEqual(r.topaz_output_path, "/tmp/picks.star")
        self.assertEqual(r.return_code, 0)

    def test_no_gt_case_has_none_precision_recall_f1(self) -> None:
        case = _minimal_case()
        sched = _make_mock_scheduler(_mock_run_next_result(with_qc=True, with_gt=False))
        runner = PickingBenchmarkRunner(scheduler_factory=lambda **_: sched)
        r = runner.run_case(case, dry_run=True)
        self.assertIsNone(r.precision)
        self.assertIsNone(r.recall)
        self.assertIsNone(r.f1)
        self.assertEqual(r.n_picks, 1200)

    def test_no_qc_result_gives_empty_qc_fields(self) -> None:
        case = _minimal_case()
        sched = _make_mock_scheduler(_mock_run_next_result(with_qc=False, with_gt=False))
        runner = PickingBenchmarkRunner(scheduler_factory=lambda **_: sched)
        r = runner.run_case(case, dry_run=True)
        self.assertIsNone(r.n_picks)
        self.assertIsNone(r.f1)
        self.assertEqual(r.qc_status, "")

    def test_failed_execution_returns_error_status(self) -> None:
        case = _minimal_case()
        sched = MagicMock()
        sched.run_next.side_effect = RuntimeError("scheduler exploded")
        runner = PickingBenchmarkRunner(scheduler_factory=lambda **_: sched)
        r = runner.run_case(case, dry_run=True)
        self.assertEqual(r.status, "error")
        self.assertIn("scheduler exploded", r.error_message)

    def test_threshold_is_propagated_to_result(self) -> None:
        case = _minimal_case(topaz_threshold=-6.0)
        sched = _make_mock_scheduler(_mock_run_next_result())
        runner = PickingBenchmarkRunner(scheduler_factory=lambda **_: sched)
        r = runner.run_case(case, dry_run=True)
        self.assertEqual(r.threshold, -6.0)

    def test_run_cases_aggregates_counts(self) -> None:
        action_mock = MagicMock()
        action_mock.name.value = "run_autopick"

        def _factory(**_):
            s = MagicMock()
            s.run_next.return_value = (action_mock, _mock_run_next_result())
            return s

        runner = PickingBenchmarkRunner(scheduler_factory=_factory)
        cases = [_minimal_case(case_id=f"c{i}") for i in range(3)]
        summary = runner.run_cases(cases, dry_run=True)
        self.assertEqual(summary["case_count"], 3)
        self.assertEqual(summary["qc_passed"], 3)
        self.assertEqual(summary["qc_failed"], 0)
        self.assertEqual(summary["errors"], 0)
        self.assertEqual(summary["benchmark_version"], BENCHMARK_VERSION)
        self.assertEqual(len(summary["results"]), 3)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


class TestPickingBenchmarkOutputs(unittest.TestCase):
    def test_write_outputs_creates_all_five_files(self) -> None:
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "results"
            write_outputs(_two_result_summary(), out_dir)
            self.assertTrue((out_dir / "benchmark_summary.csv").is_file())
            self.assertTrue((out_dir / "benchmark_summary.json").is_file())
            self.assertTrue((out_dir / "benchmark_summary.md").is_file())
            self.assertTrue((out_dir / "latest.json").is_file())
            self.assertTrue((out_dir / "latest.md").is_file())

    def test_csv_has_canonical_column_order(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "out.csv"
            write_csv(_two_result_summary()["results"], p)
            with open(p, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, _CSV_COLUMNS)

    def test_csv_contains_case_data_and_flattens_flags(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "out.csv"
            write_csv(_two_result_summary()["results"], p)
            with open(p, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["case_id"], "c1")
            self.assertEqual(rows[1]["case_id"], "c2")
            self.assertEqual(rows[1]["qc_flags"], "severe_under_picking")
            self.assertEqual(rows[0]["qc_flags"], "")

    def test_json_output_contains_all_fields(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "out.json"
            summary = _two_result_summary()
            write_json(summary, p)
            loaded = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(loaded["case_count"], 2)
            self.assertEqual(len(loaded["results"]), 2)
            self.assertEqual(loaded["benchmark_version"], BENCHMARK_VERSION)
            # qc_flags preserved as list in JSON
            self.assertIsInstance(loaded["results"][1]["qc_flags"], list)

    def test_markdown_contains_summary_header_and_table(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "out.md"
            write_markdown(_two_result_summary(), p)
            text = p.read_text(encoding="utf-8")
            self.assertIn("# Picking Benchmark Summary", text)
            self.assertIn("QC Passed", text)
            self.assertIn("c1", text)
            self.assertIn("c2", text)
            self.assertIn("failed_qc", text)

    def test_latest_json_matches_summary_json(self) -> None:
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            summary = _two_result_summary()
            write_outputs(summary, out_dir)
            latest = json.loads((out_dir / "latest.json").read_text(encoding="utf-8"))
            full = json.loads((out_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(latest, full)

    def test_write_outputs_creates_output_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "nested" / "deep" / "results"
            self.assertFalse(out_dir.exists())
            write_outputs(_two_result_summary(), out_dir)
            self.assertTrue(out_dir.is_dir())


# ---------------------------------------------------------------------------
# CLI: run-picking-benchmark --threshold-sweep
# ---------------------------------------------------------------------------


class TestRunPickingBenchmarkCLI(unittest.TestCase):
    """The CLI flag --threshold-sweep is a boolean (no argument required)."""

    def _write_cases(self, path: Path, cases: list) -> None:
        path.write_text(json.dumps({"cases": cases}), encoding="utf-8")

    def test_threshold_sweep_flag_is_boolean_in_help(self) -> None:
        from typer.testing import CliRunner
        from app.main import app

        runner = CliRunner()
        res = runner.invoke(app, ["run-picking-benchmark", "--help"])
        self.assertEqual(res.exit_code, 0, msg=res.output)
        # Boolean flag: pair with --no-threshold-sweep, no value placeholder.
        self.assertIn("--threshold-sweep", res.output)
        self.assertIn("--no-threshold-sweep", res.output)

    def test_threshold_sweep_does_not_require_argument(self) -> None:
        """--threshold-sweep must be usable as a bare flag with no value."""
        from typer.testing import CliRunner
        from unittest.mock import patch

        from app.main import app
        from agent.picking_benchmark import PickingBenchmarkCaseResult

        def _fake_run_case(self, case, *, dry_run: bool = False):
            return PickingBenchmarkCaseResult(
                case_id=case.case_id,
                dataset_id=case.dataset_id,
                backend=case.picking_backend,
                threshold=case.topaz_threshold,
                status="dry_run",
            )

        with TemporaryDirectory() as tmp:
            cases_json = Path(tmp) / "cases.json"
            self._write_cases(cases_json, [
                {
                    "case_id": "smoke",
                    "dataset_id": "ds1",
                    "project_path": tmp,
                    "picking_backend": "topaz",
                    "topaz_thresholds": [-6, -5, -4],
                }
            ])
            out_dir = Path(tmp) / "out"
            cli_runner = CliRunner()
            with patch(
                "agent.picking_benchmark.PickingBenchmarkRunner.run_case",
                _fake_run_case,
            ):
                res = cli_runner.invoke(
                    app,
                    [
                        "run-picking-benchmark",
                        "--cases-json", str(cases_json),
                        "--out-dir", str(out_dir),
                        "--dry-run",
                        "--threshold-sweep",
                    ],
                )
            self.assertEqual(res.exit_code, 0, msg=res.output)
            payload = json.loads(res.output)
            self.assertEqual(payload["case_count"], 3)

    def test_threshold_sweep_expands_three_thresholds_to_three_cases(self) -> None:
        from typer.testing import CliRunner
        from unittest.mock import patch

        from app.main import app
        from agent.picking_benchmark import PickingBenchmarkCaseResult

        observed_case_ids: list = []

        def _fake_run_case(self, case, *, dry_run: bool = False):
            observed_case_ids.append(case.case_id)
            return PickingBenchmarkCaseResult(
                case_id=case.case_id,
                dataset_id=case.dataset_id,
                backend=case.picking_backend,
                threshold=case.topaz_threshold,
                status="dry_run",
            )

        with TemporaryDirectory() as tmp:
            cases_json = Path(tmp) / "cases.json"
            self._write_cases(cases_json, [
                {
                    "case_id": "smoke",
                    "dataset_id": "ds1",
                    "project_path": tmp,
                    "picking_backend": "topaz",
                    "topaz_thresholds": [-6, -5, -4],
                }
            ])
            out_dir = Path(tmp) / "out"
            cli_runner = CliRunner()
            with patch(
                "agent.picking_benchmark.PickingBenchmarkRunner.run_case",
                _fake_run_case,
            ):
                res = cli_runner.invoke(
                    app,
                    [
                        "run-picking-benchmark",
                        "--cases-json", str(cases_json),
                        "--out-dir", str(out_dir),
                        "--dry-run",
                        "--threshold-sweep",
                    ],
                )
            self.assertEqual(res.exit_code, 0, msg=res.output)
            self.assertEqual(len(observed_case_ids), 3)
            self.assertEqual(len(set(observed_case_ids)), 3)
            for cid in observed_case_ids:
                self.assertTrue(cid.startswith("smoke_thresh_"), msg=cid)

            csv_path = out_dir / "benchmark_summary.csv"
            self.assertTrue(csv_path.is_file())
            with open(csv_path, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 3, msg="CSV should contain 3 data rows")

    def test_no_threshold_sweep_does_not_expand(self) -> None:
        """Without --threshold-sweep, a single case stays a single case
        even if topaz_thresholds is present in the JSON."""
        from typer.testing import CliRunner
        from unittest.mock import patch

        from app.main import app
        from agent.picking_benchmark import PickingBenchmarkCaseResult

        def _fake_run_case(self, case, *, dry_run: bool = False):
            return PickingBenchmarkCaseResult(
                case_id=case.case_id,
                dataset_id=case.dataset_id,
                backend=case.picking_backend,
                threshold=case.topaz_threshold,
                status="dry_run",
            )

        with TemporaryDirectory() as tmp:
            cases_json = Path(tmp) / "cases.json"
            self._write_cases(cases_json, [
                {
                    "case_id": "single",
                    "dataset_id": "ds1",
                    "project_path": tmp,
                    "picking_backend": "topaz",
                    "topaz_thresholds": [-6, -5, -4],
                }
            ])
            out_dir = Path(tmp) / "out"
            cli_runner = CliRunner()
            with patch(
                "agent.picking_benchmark.PickingBenchmarkRunner.run_case",
                _fake_run_case,
            ):
                res = cli_runner.invoke(
                    app,
                    [
                        "run-picking-benchmark",
                        "--cases-json", str(cases_json),
                        "--out-dir", str(out_dir),
                        "--dry-run",
                    ],
                )
            self.assertEqual(res.exit_code, 0, msg=res.output)
            payload = json.loads(res.output)
            self.assertEqual(payload["case_count"], 1)


if __name__ == "__main__":
    unittest.main()
