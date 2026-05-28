from __future__ import annotations

import random

from realtime_recs.domain import RankedItem


class DiversityReranker:
    def __init__(self, exploration_rate: float = 0.08) -> None:
        self.exploration_rate = max(0.0, min(exploration_rate, 0.5))

    def rerank(self, ranked: list[RankedItem], k: int, request_seed: str) -> list[RankedItem]:
        rng = random.Random(request_seed)
        pool = list(ranked)
        selected: list[RankedItem] = []
        category_counts: dict[str, int] = {}
        creator_counts: dict[str, int] = {}

        while pool and len(selected) < k:
            if rng.random() < self.exploration_rate:
                exploration_slice = pool[min(len(pool) - 1, max(0, len(pool) // 3)) :]
                choice = rng.choice(exploration_slice or pool)
                explore = True
            else:
                choice = max(
                    pool,
                    key=lambda item: self._rerank_score(item, category_counts, creator_counts),
                )
                explore = False

            pool.remove(choice)
            category_counts[choice.item.category] = category_counts.get(choice.item.category, 0) + 1
            creator_counts[choice.item.creator_id] = creator_counts.get(choice.item.creator_id, 0) + 1
            selected.append(self._with_updated_scores(choice, category_counts, creator_counts, explore))

        return selected

    def _rerank_score(
        self,
        ranked: RankedItem,
        category_counts: dict[str, int],
        creator_counts: dict[str, int],
    ) -> float:
        category_penalty = 0.075 * category_counts.get(ranked.item.category, 0)
        creator_penalty = 0.055 * creator_counts.get(ranked.item.creator_id, 0)
        freshness_boost = 0.04 * ranked.features.get("freshness", 0.0)
        return ranked.score - category_penalty - creator_penalty + freshness_boost

    def _with_updated_scores(
        self,
        ranked: RankedItem,
        category_counts: dict[str, int],
        creator_counts: dict[str, int],
        explore: bool,
    ) -> RankedItem:
        diversity_score = max(
            0.0,
            1.0
            - 0.15 * max(0, category_counts.get(ranked.item.category, 0) - 1)
            - 0.10 * max(0, creator_counts.get(ranked.item.creator_id, 0) - 1),
        )
        reasons = list(ranked.reasons)
        if explore:
            reasons.append("exploration slot")
        if diversity_score < 1.0:
            reasons.append("diversity adjusted")
        return RankedItem(
            item=ranked.item,
            score=ranked.score,
            retrieval_score=ranked.retrieval_score,
            ranker_score=ranked.ranker_score,
            diversity_score=diversity_score,
            freshness_score=ranked.freshness_score,
            explore=explore,
            reasons=reasons,
            features=ranked.features,
        )

