from __future__ import annotations

import uuid
from dataclasses import replace

from realtime_recs.config import Settings, get_settings
from realtime_recs.data.bootstrap import bootstrap_catalog
from realtime_recs.domain import Interaction, RankedItem, RecommendationResponse, UserState
from realtime_recs.features import FeatureBuilder
from realtime_recs.observability import ServiceMetrics, Timer
from realtime_recs.ranker import MultiObjectiveRanker
from realtime_recs.reranker import DiversityReranker
from realtime_recs.retrieval import BruteForceAnnIndex, RetrievalIndex
from realtime_recs.storage.cache import TTLCache
from realtime_recs.storage.catalog import InMemoryCatalog
from realtime_recs.vector import hashed_text_embedding


class RecommendationService:
    def __init__(
        self,
        settings: Settings,
        catalog: InMemoryCatalog,
        retrieval: RetrievalIndex,
        cache: TTLCache,
    ) -> None:
        self.settings = settings
        self.catalog = catalog
        self.retrieval = retrieval
        self.cache = cache
        self.features = FeatureBuilder(catalog, settings.embedding_dim)
        self.ranker = MultiObjectiveRanker()
        self.reranker = DiversityReranker(settings.exploration_rate)
        self.metrics = ServiceMetrics()

    @classmethod
    def create_demo(cls, settings: Settings | None = None) -> "RecommendationService":
        resolved_settings = settings or get_settings()
        items, users = bootstrap_catalog(dim=resolved_settings.embedding_dim)
        catalog = InMemoryCatalog(items=items, users=users)
        retrieval = BruteForceAnnIndex(items)
        cache = TTLCache(ttl_seconds=resolved_settings.cache_ttl_seconds)
        return cls(resolved_settings, catalog, retrieval, cache)

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        candidate_pool: int | None = None,
        context: dict[str, str] | None = None,
        use_cache: bool = True,
    ) -> RecommendationResponse:
        k = max(1, min(k, 25))
        candidate_pool = min(
            candidate_pool or self.settings.max_candidate_pool,
            self.settings.max_candidate_pool,
        )
        context = context or {}
        cache_key = self._cache_key(user_id, k, candidate_pool, context)
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return replace(cached, cache_hit=True, latency_ms=0.0)

        with Timer() as timer:
            user = self._resolve_user(user_id, context)
            query_embedding = self.features.user_embedding(user)
            seen_item_ids = {
                event.item_id
                for event in user.recent_events[:10]
                if event.event_type in {"like", "share", "complete", "hide", "dislike"}
            }
            candidates = self.retrieval.search(
                query_embedding,
                k=max(k * 3, candidate_pool),
                exclude_item_ids=seen_item_ids,
            )
            ranked = [
                self._rank_candidate(user, candidate, query_embedding, context)
                for candidate in candidates
            ]
            ranked.sort(key=lambda item: item.ranker_score, reverse=True)
            request_id = uuid.uuid4().hex[:12]
            recommendations = self.reranker.rerank(ranked, k=k, request_seed=request_id)

        response = RecommendationResponse(
            user_id=user.user_id,
            request_id=request_id,
            model_version=self.settings.model_version,
            latency_ms=round(timer.elapsed_ms, 2),
            cache_hit=False,
            candidates_considered=len(candidates),
            recommendations=recommendations,
            diagnostics={
                "retrieval_stage": "two_tower_embedding_ann",
                "ranking_stage": "multi_objective_ranker",
                "reranking_stage": "diversity_freshness_exploration",
                "latency_target_ms": 100,
            },
        )
        self.metrics.observe_request(timer.elapsed_ms, len(candidates))
        if use_cache:
            self.cache.set(cache_key, response)
        return response

    def ingest_event(
        self,
        user_id: str,
        item_id: str,
        event_type: str,
        value: float = 1.0,
    ) -> UserState:
        if self.catalog.item(item_id) is None:
            raise KeyError(f"Unknown item_id: {item_id}")
        event = Interaction(
            user_id=user_id,
            item_id=item_id,
            event_type=event_type,
            value=max(0.0, min(value, 1.5)),
        )
        updated_user = self.catalog.append_event(event)
        self.cache.invalidate_prefix(f"recommend:{user_id}:")
        self.metrics.observe_event()
        return updated_user

    def telemetry(self) -> dict[str, object]:
        snapshot = self.catalog.snapshot()
        return {
            "service": self.settings.service_name,
            "env": self.settings.env,
            "model_version": self.settings.model_version,
            "catalog": snapshot.__dict__,
            "cache": {
                "entries": self.cache.size(),
                "hits": self.cache.stats.hits,
                "misses": self.cache.stats.misses,
                "hit_rate": round(self.cache.stats.hit_rate, 3),
            },
            "metrics": self.metrics.as_dict(),
        }

    def _rank_candidate(
        self,
        user: UserState,
        candidate,
        query_embedding,
        context: dict[str, str],
    ) -> RankedItem:
        features = self.features.ranking_features(
            user=user,
            item=candidate.item,
            query_embedding=query_embedding,
            retrieval_score=candidate.retrieval_score,
            context=context,
        )
        ranker_score, objectives = self.ranker.score(features)
        merged_features = {**features, **objectives}
        reasons = self._reasons(candidate.item.category, merged_features)
        return RankedItem(
            item=candidate.item,
            score=ranker_score,
            retrieval_score=candidate.retrieval_score,
            ranker_score=ranker_score,
            diversity_score=1.0,
            freshness_score=features["freshness"],
            explore=False,
            reasons=reasons,
            features=merged_features,
        )

    def _resolve_user(self, user_id: str, context: dict[str, str]) -> UserState:
        existing = self.catalog.user(user_id)
        if existing is not None:
            return existing

        topics = [
            topic.strip().lower()
            for topic in context.get("onboarding_topics", "ai,design,education").split(",")
            if topic.strip()
        ]
        preferred_categories = {topic: 0.6 for topic in topics[:4]}
        embedding = hashed_text_embedding(" ".join(topics) or "new user discovery")
        user = UserState(
            user_id=user_id,
            display_name=f"Guest {user_id[-4:]}",
            onboarding_topics=topics,
            preferred_categories=preferred_categories,
            long_term_embedding=embedding,
            device=context.get("device", "web"),
            region=context.get("region", "US"),
        )
        self.catalog.upsert_user(user)
        return user

    @staticmethod
    def _reasons(category: str, features: dict[str, float]) -> list[str]:
        reasons: list[str] = []
        if features.get("category_affinity", 0.0) > 0.5:
            reasons.append(f"strong {category} affinity")
        if features.get("similarity", 0.0) > 0.25:
            reasons.append("semantic match")
        if features.get("freshness", 0.0) > 0.65:
            reasons.append("fresh content")
        if features.get("quality", 0.0) > 0.82:
            reasons.append("high quality prior")
        if not reasons:
            reasons.append("balanced objective fit")
        return reasons[:3]

    @staticmethod
    def _cache_key(
        user_id: str,
        k: int,
        candidate_pool: int,
        context: dict[str, str],
    ) -> str:
        context_bits = "&".join(f"{key}={context[key]}" for key in sorted(context))
        return f"recommend:{user_id}:k={k}:pool={candidate_pool}:{context_bits}"
