from __future__ import annotations

import unittest

from policies.reconstruction_policy import ReconstructionPolicy
from schemas.qc_metrics import QCMetrics
from schemas.sample_state import SampleState


class TestReconstructionPolicy(unittest.TestCase):
    def setUp(self) -> None:
        self.state = SampleState(project_id="p", project_path="/tmp/p", stage="Class2D", status="running")

    def test_no_initial_model_selects_cryodrgn_ai(self) -> None:
        decision = ReconstructionPolicy().select(
            self.state,
            QCMetrics(init3d_completed=False),
            context={"no_initial_model": True},
        )
        self.assertEqual(decision["selected_backend"], "cryodrgn_ai")
        self.assertIn("no_reliable_initial_model", decision["selection_reason"])

    def test_heterogeneity_score_above_threshold_selects_cryodrgn_ai(self) -> None:
        decision = ReconstructionPolicy({"reconstruction": {"heterogeneity_threshold": 0.5}}).select(
            self.state,
            QCMetrics(init3d_completed=True),
            context={"heterogeneity_score": 0.8},
        )
        self.assertEqual(decision["selected_backend"], "cryodrgn_ai")
        self.assertIn("heterogeneity_score_above_threshold", decision["selection_reason"])

    def test_relion_preferred_when_no_neural_trigger(self) -> None:
        decision = ReconstructionPolicy().select(
            self.state,
            QCMetrics(init3d_completed=True),
            context={"heterogeneity_score": 0.1},
        )
        self.assertEqual(decision["selected_backend"], "relion")


if __name__ == "__main__":
    unittest.main()
