"""Tests for the LLM-supervised candidate-map routing v0."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import mrcfile
import numpy as np

from agent.llm_supervisor import (
    build_llm_decision_prompt,
    build_llm_observation,
    llm_supervised_route,
    parse_llm_decision,
    validate_llm_decision,
)
from agent.reporter import Reporter
from app.scheduler import Scheduler
from schemas.action import ActionName
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState


def _candidate_dict(path: Path, idx: int) -> dict:
    return {
        "source_backend": "cryodrgn_ai",
        "source_run_dir": str(path.parent),
        "map_path": str(path),
        "latent_source": str(path.parent / "representative_z.txt"),
        "weights_path": str(path.parent / "abinit" / "weights.pkl"),
        "qc_status": "not_evaluated",
        "recommended_next_action": "route_to_model_building",
    }


def _make_real_state(tmp: Path, n_maps: int = 2) -> SampleState:
    """Create an on-disk SampleState with real synthetic MRC candidate maps."""

    seq = tmp / "seq.fasta"
    seq.write_text(">a\nA\n", encoding="utf-8")
    base_map = tmp / "base_map.mrc"
    base_map.write_text("map", encoding="utf-8")
    map_dir = tmp / "sampled_volumes"
    map_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict] = []
    rng = np.random.default_rng(7)
    for idx in range(n_maps):
        map_path = map_dir / f"vol_{idx:03d}.mrc"
        with mrcfile.new(str(map_path), overwrite=True) as mrc:
            scale = 1.0 if idx == n_maps - 1 else 0.05
            mrc.set_data(rng.normal(0.0, scale, size=(12, 12, 12)).astype(np.float32))
        candidates.append(_candidate_dict(map_path, idx))
    return SampleState(
        project_id="p",
        project_path=str(tmp),
        stage="PostProcess",
        status="running",
        map_path=str(base_map),
        sequence_path=str(seq),
        last_execution={"candidate_maps": candidates},
    )


def _good_qc() -> QCMetrics:
    return QCMetrics(
        motioncorr_completed=True,
        ctf_completed=True,
        picking_completed=True,
        class2d_completed=True,
        init3d_completed=True,
        refine3d_completed=True,
        postprocess_completed=True,
        current_resolution=3.5,
    )


class TestBuildLLMObservation(unittest.TestCase):
    def test_extracts_compact_candidate_map_fields(self) -> None:
        with TemporaryDirectory() as tmp:
            state = _make_real_state(Path(tmp), n_maps=2)
            obs = build_llm_observation(state)

            self.assertEqual(obs["stage"], "PostProcess")
            self.assertTrue(obs["sequence_path_present"])
            self.assertFalse(obs["af3_dir_present"])
            self.assertEqual(obs["candidate_count"], 2)
            self.assertEqual(len(obs["candidate_maps"]), 2)

            required_keys = {
                "candidate_index",
                "map_path",
                "qc_score",
                "qc_status",
                "qc_reason",
                "finite_fraction",
                "density_std",
                "edge_artifact_score",
                "latent_source",
                "weights_path",
                "recommended_backend",
                "recommended_action",
                "preflight",
            }
            for cand in obs["candidate_maps"]:
                self.assertTrue(required_keys.issubset(cand.keys()))
                self.assertIn("available", cand["preflight"])
                self.assertIn("blocked_reason", cand["preflight"])

            # Compact view does not include the large raw candidate_map_qc dict.
            self.assertNotIn("candidate_map_qc", obs["candidate_maps"][0])

            # Rule-based selection summary is exposed.
            self.assertEqual(obs["rule_based_selected_action"], "run_modelangelo")
            self.assertEqual(obs["rule_based_selected_backend"], "modelangelo")
            self.assertEqual(obs["candidate_map_selection_reason"], "highest_qc_score")


class TestMockLLMSelectsHighestQCScore(unittest.TestCase):
    def test_mock_route_picks_top_ranked_candidate(self) -> None:
        with TemporaryDirectory() as tmp:
            state = _make_real_state(Path(tmp), n_maps=2)
            result = llm_supervised_route(state, provider="mock")

            self.assertEqual(result["mode"], "llm_supervised")
            self.assertFalse(result["fallback_used"])
            # The synthetic high-variance map sits at idx=1 -> top of QC ranking.
            self.assertEqual(result["selected_candidate_index"], 1)
            self.assertEqual(result["selected_backend"], "modelangelo")
            self.assertEqual(result["selected_action"], "run_modelangelo")
            self.assertTrue(result["validation"]["valid"])
            self.assertIsNone(result["fallback_reason"])


class TestParseLLMDecision(unittest.TestCase):
    def test_parses_valid_json(self) -> None:
        text = (
            "  "
            + json.dumps(
                {
                    "decision_type": "candidate_map_model_building",
                    "selected_candidate_index": 0,
                    "selected_backend": "modelangelo",
                    "selected_action": "run_modelangelo",
                    "confidence": 0.9,
                    "decision_reason": "highest qc score candidate",
                    "risk_flags": [],
                    "fallback_allowed": True,
                }
            )
            + "  \n"
        )
        decoded = parse_llm_decision(text)
        self.assertEqual(decoded["selected_backend"], "modelangelo")
        self.assertEqual(decoded["selected_candidate_index"], 0)

    def test_rejects_markdown_code_fences(self) -> None:
        text = "```json\n{\"selected_backend\": \"modelangelo\"}\n```"
        with self.assertRaises(ValueError):
            parse_llm_decision(text)


class TestValidateLLMDecision(unittest.TestCase):
    def _observation_with_two_candidates(self) -> dict:
        return {
            "stage": "PostProcess",
            "candidate_count": 2,
            "candidate_maps": [
                {
                    "candidate_index": 0,
                    "map_path": "/tmp/vol_000.mrc",
                    "qc_score": 60.0,
                    "qc_status": "passed",
                    "recommended_backend": "modelangelo",
                    "recommended_action": "run_modelangelo",
                    "preflight": {"available": True, "blocked_reason": None},
                },
                {
                    "candidate_index": 1,
                    "map_path": "/tmp/vol_001.mrc",
                    "qc_score": 30.0,
                    "qc_status": "warning",
                    "recommended_backend": "emodelx",
                    "recommended_action": "run_emodelx",
                    "preflight": {"available": False, "blocked_reason": "missing_inputs"},
                },
            ],
            "ranked_candidate_maps": [],
            "candidate_map_routing_records": [],
            "selected_candidate_index": 0,
            "candidate_map_selection_reason": "highest_qc_score",
            "sequence_path_present": True,
            "af3_dir_present": False,
            "rule_based_selected_backend": "modelangelo",
            "rule_based_selected_action": "run_modelangelo",
        }

    def test_rejects_nonexistent_candidate_index(self) -> None:
        obs = self._observation_with_two_candidates()
        decision = {
            "decision_type": "candidate_map_model_building",
            "selected_candidate_index": 7,
            "selected_backend": "modelangelo",
            "selected_action": "run_modelangelo",
            "confidence": 0.8,
            "decision_reason": "made up",
        }
        result = validate_llm_decision(decision, obs)
        self.assertFalse(result["valid"])
        self.assertTrue(any("not present in candidate_maps" in e for e in result["errors"]))

    def test_rejects_unavailable_backend_preflight(self) -> None:
        obs = self._observation_with_two_candidates()
        decision = {
            "decision_type": "candidate_map_model_building",
            "selected_candidate_index": 1,
            "selected_backend": "emodelx",
            "selected_action": "run_emodelx",
            "confidence": 0.8,
            "decision_reason": "second candidate",
        }
        result = validate_llm_decision(decision, obs)
        self.assertFalse(result["valid"])
        self.assertTrue(any("preflight" in e and "unavailable" in e for e in result["errors"]))

    def test_allows_human_review_even_when_preflight_missing(self) -> None:
        obs = self._observation_with_two_candidates()
        # Mark all candidate preflights unavailable to make sure human_review is exempt.
        for c in obs["candidate_maps"]:
            c["preflight"] = {"available": False, "blocked_reason": "low qc"}
        decision = {
            "decision_type": "candidate_map_model_building",
            "selected_candidate_index": 0,
            "selected_backend": "human_review",
            "selected_action": "human_review",
            "confidence": 0.5,
            "decision_reason": "candidate quality too low",
        }
        result = validate_llm_decision(decision, obs)
        self.assertTrue(result["valid"], msg=str(result["errors"]))
        validated = result["validated_decision"]
        self.assertEqual(validated["selected_backend"], "human_review")
        self.assertEqual(validated["selected_action"], "human_review")


class TestInvalidLLMFallsBackToRuleBased(unittest.TestCase):
    def test_invalid_decision_falls_back_to_rule_based(self) -> None:
        with TemporaryDirectory() as tmp:
            state = _make_real_state(Path(tmp), n_maps=2)
            bogus = json.dumps(
                {
                    "decision_type": "candidate_map_model_building",
                    "selected_candidate_index": 999,
                    "selected_backend": "modelangelo",
                    "selected_action": "run_modelangelo",
                    "confidence": 0.9,
                    "decision_reason": "fabricated index",
                }
            )
            result = llm_supervised_route(state, llm_response_text=bogus, provider="mock")
            self.assertTrue(result["fallback_used"])
            self.assertIsNotNone(result["fallback_reason"])
            self.assertFalse(result["validation"]["valid"])
            # Rule-based fallback chooses index=1 (top of QC ranking) as modelangelo.
            self.assertEqual(result["selected_candidate_index"], 1)
            self.assertEqual(result["selected_backend"], "modelangelo")
            self.assertEqual(result["selected_action"], "run_modelangelo")


class TestSchedulerLLMSupervisedAgentMode(unittest.TestCase):
    def test_returns_candidate_map_model_building_action(self) -> None:
        with TemporaryDirectory() as tmp:
            state = _make_real_state(Path(tmp), n_maps=2)
            qc = _good_qc()
            scheduler = Scheduler(project_root=Path(tmp))
            action = scheduler._resolve_next_action(
                state,
                qc,
                agent_mode="llm_supervised",
            )
            self.assertEqual(action.name, ActionName.RUN_MODELANGELO)
            self.assertEqual(
                action.parameters.get("branch_name"),
                "candidate_map_model_building",
            )
            self.assertEqual(action.parameters.get("agent_mode"), "llm_supervised")
            self.assertIsNotNone(action.parameters.get("llm_supervised_result"))
            self.assertEqual(action.parameters.get("llm_selected_action"), "run_modelangelo")
            self.assertEqual(action.parameters.get("llm_selected_backend"), "modelangelo")
            self.assertFalse(action.parameters.get("llm_fallback_used"))
            self.assertIsInstance(action.parameters.get("llm_decision_reason"), str)


class TestReporterIncludesLLMSupervisedSection(unittest.TestCase):
    def test_report_renders_llm_supervised_routing_section(self) -> None:
        with TemporaryDirectory() as tmp:
            state = _make_real_state(Path(tmp), n_maps=2)
            qc = _good_qc()
            scheduler = Scheduler(project_root=Path(tmp))
            action = scheduler._resolve_next_action(
                state,
                qc,
                agent_mode="llm_supervised",
            )
            report = Reporter().build_report(
                state=state,
                qc=qc,
                recent_action=None,
                next_action=action,
            )
            md = report.to_markdown()
            self.assertIn("## LLM-Supervised Routing", md)
            self.assertIn("Mode: llm_supervised", md)
            self.assertIn("Selected backend: modelangelo", md)
            self.assertIn("Selected action: run_modelangelo", md)
            self.assertIn("Fallback used: False", md)
            self.assertIn("Selected candidate index: 1", md)
            self.assertIn("Validation errors: none", md)
            self.assertIn("Rule-based selected candidate", md)
            self.assertIn("LLM agrees with rule-based selection: True", md)


class TestPromptSafety(unittest.TestCase):
    def test_prompt_instructs_no_markdown_and_includes_observation(self) -> None:
        with TemporaryDirectory() as tmp:
            state = _make_real_state(Path(tmp), n_maps=1)
            obs = build_llm_observation(state)
            prompt = build_llm_decision_prompt(obs)
            self.assertIn("JSON", prompt)
            self.assertIn("Markdown", prompt)
            self.assertIn("candidate_map_model_building", prompt)
            # Must not embed Markdown code fences itself.
            self.assertNotIn("```", prompt)


if __name__ == "__main__":
    unittest.main()
