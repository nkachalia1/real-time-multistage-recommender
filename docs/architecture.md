# Architecture

This project is a compact version of a production real-time recommender. It keeps the core interfaces realistic while staying small enough to deploy and explain in an interview.

## Request Path

1. The client calls `POST /api/recommendations` with `user_id`, `k`, context, and cache preference.
2. `RecommendationService` resolves the user profile. Unknown users are initialized from onboarding topics.
3. `FeatureBuilder` builds a user embedding from long-term preferences plus recent positive events with time decay.
4. `RetrievalIndex` searches item embeddings and returns a candidate pool.
5. `MultiObjectiveRanker` scores each candidate for CTR, long watch probability, and retention.
6. `DiversityReranker` applies category diversity, creator de-duplication, freshness, and exploration.
7. The response includes ranked items, objective scores, reasons, latency, model version, and diagnostics.

## Production Mapping

| Demo component | Production component |
| --- | --- |
| `InMemoryCatalog` | Postgres or DynamoDB metadata store |
| `TTLCache` | Redis |
| `BruteForceAnnIndex` | FAISS IVF/HNSW, ScaNN, Pinecone, Milvus |
| `FeatureBuilder` | Feature store online/offline transformations |
| `MultiObjectiveRanker` | XGBoost, DLRM, Wide & Deep, or transformer ranker |
| `backfill_features.py` | Airflow/Dagster/Spark feature backfill |
| `/api/events` | Kafka/Kinesis event producer |

## Failure Handling

- If cache is empty, the service computes recommendations directly.
- If a user is unknown, cold-start onboarding topics create a semantic profile.
- If a feedback event references an unknown item, the API returns 404 instead of corrupting the profile.
- If optional training dependencies are absent, training scripts fail with an install hint while serving remains unaffected.

## Latency Budget

| Stage | Target | Notes |
| --- | ---: | --- |
| User feature read | 5-10 ms | Redis/online feature store in production |
| ANN retrieval | 10-30 ms | Approximate search over millions of vectors |
| Ranking | 20-40 ms | Batch model inference over hundreds of candidates |
| Re-ranking | 1-5 ms | Constraint optimization over final slate |
| Serialization/network | 10-20 ms | Keep payload small and cache common requests |

The demo exposes observed latency in each recommendation response and aggregate metrics at `/api/metrics`.

