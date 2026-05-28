from __future__ import annotations

import math


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


class MultiObjectiveRanker:
    """Deployable ranker with explicit features and inspectable weights.

    The training scripts show how this can be replaced by an XGBoost, Wide &
    Deep, or DLRM-style ranker while preserving the serving contract.
    """

    def __init__(self) -> None:
        self.ctr_weights = {
            "bias": -0.65,
            "similarity": 2.20,
            "category_affinity": 1.25,
            "creator_affinity": 0.35,
            "freshness": 0.35,
            "quality": 0.85,
            "prior_ctr": 2.40,
            "seen_penalty": -2.30,
            "negative_affinity": -1.60,
            "device_match": 0.10,
        }
        self.watch_weights = {
            "bias": -0.35,
            "similarity": 1.40,
            "category_affinity": 0.90,
            "freshness": 0.15,
            "quality": 1.10,
            "seen_penalty": -1.35,
            "negative_affinity": -1.20,
        }
        self.retention_weights = {
            "bias": -0.20,
            "similarity": 0.80,
            "category_affinity": 0.45,
            "freshness": 0.65,
            "quality": 0.55,
            "seen_penalty": -0.90,
            "negative_affinity": -0.95,
        }

    def score(self, features: dict[str, float]) -> tuple[float, dict[str, float]]:
        ctr = sigmoid(self._linear(features, self.ctr_weights))
        expected_watch = sigmoid(self._linear(features, self.watch_weights))
        retention = sigmoid(self._linear(features, self.retention_weights))
        objective_score = (0.52 * ctr) + (0.30 * expected_watch) + (0.18 * retention)
        return objective_score, {
            "p_ctr": ctr,
            "p_long_watch": expected_watch,
            "p_retention": retention,
            "objective_score": objective_score,
        }

    @staticmethod
    def _linear(features: dict[str, float], weights: dict[str, float]) -> float:
        total = weights.get("bias", 0.0)
        for name, weight in weights.items():
            if name == "bias":
                continue
            total += features.get(name, 0.0) * weight
        return total

