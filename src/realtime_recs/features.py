from __future__ import annotations

import math
from time import time

from realtime_recs.domain import Interaction, Item, UserState, Vector
from realtime_recs.storage.catalog import InMemoryCatalog
from realtime_recs.vector import blend, dot, hashed_text_embedding


EVENT_WEIGHTS = {
    "impression": 0.05,
    "click": 0.40,
    "watch": 0.65,
    "complete": 0.85,
    "like": 1.00,
    "share": 1.15,
    "hide": -0.85,
    "dislike": -1.00,
}


class FeatureBuilder:
    def __init__(self, catalog: InMemoryCatalog, dim: int) -> None:
        self.catalog = catalog
        self.dim = dim

    def user_embedding(self, user: UserState) -> Vector:
        vectors: list[tuple[Vector, float]] = [(user.long_term_embedding, 1.0)]
        now = time()
        for event in user.recent_events[:25]:
            item = self.catalog.item(event.item_id)
            if item is None:
                continue
            event_weight = EVENT_WEIGHTS.get(event.event_type, 0.0) * max(event.value, 0.05)
            if event_weight <= 0:
                continue
            age_hours = max((now - event.timestamp) / 3600.0, 0.0)
            recency = math.exp(-age_hours / 18.0)
            vectors.append((item.embedding, 1.4 * event_weight * recency))

        if user.onboarding_topics:
            vectors.append(
                (
                    hashed_text_embedding(" ".join(user.onboarding_topics), dim=self.dim),
                    0.35,
                )
            )
        return blend(vectors, dim=self.dim)

    def ranking_features(
        self,
        user: UserState,
        item: Item,
        query_embedding: Vector,
        retrieval_score: float,
        context: dict[str, str] | None = None,
    ) -> dict[str, float]:
        context = context or {}
        seen_item_ids = {event.item_id for event in user.recent_events if event.event_type != "impression"}
        category_affinity = user.preferred_categories.get(item.category, 0.0)
        creator_affinity = self._creator_affinity(user, item.creator_id)
        negative_affinity = self._negative_category_affinity(user, item.category)
        freshness_score = math.exp(-max(item.age_hours, 0.0) / 96.0)
        device_match = 1.0 if context.get("device", user.device) == user.device else 0.6

        return {
            "similarity": max(dot(query_embedding, item.embedding), -1.0),
            "retrieval_score": retrieval_score,
            "category_affinity": category_affinity,
            "creator_affinity": creator_affinity,
            "freshness": freshness_score,
            "quality": item.quality_score,
            "prior_ctr": item.prior_ctr,
            "seen_penalty": 1.0 if item.item_id in seen_item_ids else 0.0,
            "negative_affinity": negative_affinity,
            "device_match": device_match,
        }

    def _creator_affinity(self, user: UserState, creator_id: str) -> float:
        positive = 0
        total = 0
        for event in user.recent_events:
            item = self.catalog.item(event.item_id)
            if item is None or item.creator_id != creator_id:
                continue
            total += 1
            if EVENT_WEIGHTS.get(event.event_type, 0.0) > 0.3:
                positive += 1
        return 0.0 if total == 0 else positive / total

    def _negative_category_affinity(self, user: UserState, category: str) -> float:
        negative = 0
        total = 0
        for event in user.recent_events:
            item = self.catalog.item(event.item_id)
            if item is None or item.category != category:
                continue
            total += 1
            if EVENT_WEIGHTS.get(event.event_type, 0.0) < 0:
                negative += 1
        return 0.0 if total == 0 else negative / total


def build_interaction(user_id: str, item_id: str, event_type: str, value: float = 1.0) -> Interaction:
    return Interaction(user_id=user_id, item_id=item_id, event_type=event_type, value=value)

