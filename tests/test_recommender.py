from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from realtime_recs.config import Settings
from realtime_recs.embeddings import HashingEmbeddingProvider
from realtime_recs.service import RecommendationService


class RecommendationServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = RecommendationService.create_demo(
            Settings(cache_ttl_seconds=10, exploration_rate=0.0)
        )

    def test_recommendations_return_requested_shape(self) -> None:
        response = self.service.recommend("u_alex", k=8, use_cache=False)
        self.assertEqual(response.user_id, "u_alex")
        self.assertEqual(len(response.recommendations), 8)
        self.assertGreater(response.candidates_considered, 8)
        self.assertIn("retrieval_stage", response.diagnostics)

    def test_feedback_updates_recent_features_and_invalidates_cache(self) -> None:
        first = self.service.recommend("u_alex", k=5, use_cache=True)
        cached = self.service.recommend("u_alex", k=5, use_cache=True)
        self.assertTrue(cached.cache_hit)
        self.service.ingest_event("u_alex", first.recommendations[0].item.item_id, "hide", value=1.0)
        second = self.service.recommend("u_alex", k=5, use_cache=True)
        self.assertFalse(second.cache_hit)
        self.assertNotEqual(
            first.recommendations[0].item.item_id,
            second.recommendations[0].item.item_id,
        )

    def test_cold_start_user_uses_onboarding_topics(self) -> None:
        response = self.service.recommend(
            "u_new_test",
            k=6,
            context={"onboarding_topics": "music,gaming"},
            use_cache=False,
        )
        categories = {item.item.category for item in response.recommendations}
        self.assertTrue({"music", "gaming"} & categories)

    def test_reranking_promotes_category_diversity(self) -> None:
        response = self.service.recommend("u_maya", k=10, use_cache=False)
        categories = [item.item.category for item in response.recommendations]
        self.assertGreaterEqual(len(set(categories)), 3)

    def test_hashing_embedding_provider_is_deterministic(self) -> None:
        provider = HashingEmbeddingProvider(dim=16)
        self.assertEqual(provider.embed("ai agents"), provider.embed("ai agents"))
        self.assertEqual(len(provider.embed("ai agents")), 16)


if __name__ == "__main__":
    unittest.main()
