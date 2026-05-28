from __future__ import annotations

import argparse
import json

from realtime_recs.config import get_settings
from realtime_recs.service import RecommendationService


def evaluate(k: int = 10) -> dict[str, float]:
    service = RecommendationService.create_demo(get_settings())
    users = service.catalog.users()
    preference_precisions: list[float] = []
    diversities: list[float] = []

    for user in users:
        response = service.recommend(user.user_id, k=k, use_cache=False)
        preferred_categories = {
            category
            for category, weight in user.preferred_categories.items()
            if weight >= 0.6
        }
        if preferred_categories:
            hits = sum(
                1
                for item in response.recommendations
                if item.item.category in preferred_categories
            )
            preference_precisions.append(hits / min(k, len(response.recommendations)))
        categories = [item.item.category for item in response.recommendations]
        diversities.append(len(set(categories)) / max(len(categories), 1))

    return {
        "preference_precision_at_k": round(
            sum(preference_precisions) / max(len(preference_precisions), 1),
            4,
        ),
        "diversity_at_k": round(sum(diversities) / max(len(diversities), 1), 4),
        "users": float(len(users)),
        "k": float(k),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(evaluate(k=args.k), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
