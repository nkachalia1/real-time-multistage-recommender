from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any


Vector = list[float]


@dataclass(frozen=True)
class Item:
    item_id: str
    title: str
    creator_id: str
    category: str
    tags: list[str]
    description: str
    age_hours: float
    quality_score: float
    prior_ctr: float
    embedding: Vector


@dataclass(frozen=True)
class Interaction:
    user_id: str
    item_id: str
    event_type: str
    value: float
    timestamp: float = field(default_factory=time)


@dataclass
class UserState:
    user_id: str
    display_name: str
    onboarding_topics: list[str]
    preferred_categories: dict[str, float]
    long_term_embedding: Vector
    recent_events: list[Interaction] = field(default_factory=list)
    device: str = "web"
    region: str = "US"

    def clone_with_event(self, event: Interaction, max_events: int = 50) -> "UserState":
        events = [event, *self.recent_events][:max_events]
        return UserState(
            user_id=self.user_id,
            display_name=self.display_name,
            onboarding_topics=list(self.onboarding_topics),
            preferred_categories=dict(self.preferred_categories),
            long_term_embedding=list(self.long_term_embedding),
            recent_events=events,
            device=self.device,
            region=self.region,
        )


@dataclass(frozen=True)
class Candidate:
    item: Item
    retrieval_score: float
    source: str


@dataclass(frozen=True)
class RankedItem:
    item: Item
    score: float
    retrieval_score: float
    ranker_score: float
    diversity_score: float
    freshness_score: float
    explore: bool
    reasons: list[str]
    features: dict[str, float]


@dataclass(frozen=True)
class RecommendationResponse:
    user_id: str
    request_id: str
    model_version: str
    latency_ms: float
    cache_hit: bool
    candidates_considered: int
    recommendations: list[RankedItem]
    diagnostics: dict[str, Any]

