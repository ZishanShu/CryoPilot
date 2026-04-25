"""Routing policy for optional reconstruction backends."""

from __future__ import annotations

from typing import Any, Dict, Optional


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "neural", "cryodrgn_ai"}


class ReconstructionPolicy:
    """Select RELION or cryodrgn_ai without replacing RELION as the default."""

    def __init__(self, rules: Optional[Dict[str, Any]] = None) -> None:
        self.rules = rules or {}

    def threshold(self) -> float:
        thresholds = self.rules.get("thresholds") if isinstance(self.rules, dict) else {}
        if isinstance(thresholds, dict) and thresholds.get("heterogeneity_score_for_cryodrgn_ai") is not None:
            return float(thresholds["heterogeneity_score_for_cryodrgn_ai"])
        reconstruction = self.rules.get("reconstruction") if isinstance(self.rules, dict) else {}
        if isinstance(reconstruction, dict) and reconstruction.get("heterogeneity_threshold") is not None:
            return float(reconstruction["heterogeneity_threshold"])
        return 0.7

    def select(self, state: Any, qc: Any, *, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ctx = dict(context or {})
        threshold = self.threshold()
        score = ctx.get("heterogeneity_score")
        if score is None:
            score = getattr(qc, "heterogeneity_score", None)
        if score is None and bool(getattr(qc, "heterogeneity_warning", False)):
            score = 1.0

        no_initial_model = _truthy(ctx.get("no_initial_model")) or _truthy(ctx.get("no_reliable_initial_model"))
        if "no_initial_model" not in ctx and "no_reliable_initial_model" not in ctx:
            no_initial_model = False
        unstable_class3d = _truthy(ctx.get("relion_3d_classification_unstable")) or _truthy(
            ctx.get("class3d_unstable")
        )
        explicit_neural = _truthy(ctx.get("requested_backend")) or _truthy(
            ctx.get("user_requested_neural_heterogeneity")
        )

        reasons: list[str] = []
        if explicit_neural:
            reasons.append("user_requested_neural_heterogeneity")
        if no_initial_model:
            reasons.append("no_reliable_initial_model")
        if score is not None and float(score) >= threshold:
            reasons.append(f"heterogeneity_score_above_threshold:{float(score):.3f}>={threshold:.3f}")
        if unstable_class3d:
            reasons.append("relion_3d_classification_unstable")

        if reasons:
            return {
                "category": "reconstruction",
                "selected_backend": "cryodrgn_ai",
                "selection_reason": "; ".join(reasons),
                "heterogeneity_score": score,
                "threshold": threshold,
                "candidate_backends": ["relion", "cryodrgn_ai"],
                "fallback_candidates": ["relion_initial_model", "relion_3d_classification", "human_review"],
            }

        return {
            "category": "reconstruction",
            "selected_backend": "relion",
            "selection_reason": "prefer_existing_relion_reconstruction_path",
            "heterogeneity_score": score,
            "threshold": threshold,
            "candidate_backends": ["relion", "cryodrgn_ai"],
            "fallback_candidates": ["cryodrgn_ai", "human_review"],
        }
