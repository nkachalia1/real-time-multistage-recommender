# MLOps Playbook

## Data Contracts

User events:

- `user_id`: stable user identifier.
- `item_id`: stable item identifier.
- `event_type`: one of impression, click, watch, complete, like, share, hide, dislike.
- `value`: normalized event strength.
- `timestamp`: event time, not ingestion time.

Item metadata:

- `item_id`, `creator_id`, `category`, `tags`, `title`, `description`.
- `embedding`: item tower output or semantic embedding.
- `age_hours`, `quality_score`, `prior_ctr`.

## Training Pipeline

1. Validate raw event logs and item metadata.
2. Generate offline feature snapshots for user, item, and context features.
3. Train retrieval model with pairwise or sampled-softmax loss.
4. Build ANN index from item embeddings.
5. Train ranker with labels for click, watch time, conversion, and retention.
6. Evaluate offline metrics and calibration.
7. Push model and feature schemas to registry.
8. Shadow deploy before an A/B test.

## Monitoring

Operational:

- p50/p95/p99 latency.
- error rate and timeout rate.
- cache hit rate.
- candidate pool size.

Data:

- missing feature rate.
- feature distribution drift.
- event volume by type.
- delayed or duplicate event counts.

Model:

- embedding norm drift.
- score distribution drift.
- calibration by user segment.
- online metric deltas for CTR, watch time, retention, and hides.

Product:

- creator/category concentration.
- cold-start activation.
- exploration bucket performance.
- long-term retention guardrails.

## Rollback Plan

- Keep previous model artifacts and ANN index available.
- Version every response with `model_version`.
- Support feature flags for exploration, diversity, and freshness boosts.
- Fall back to popular/fresh content if user features or vector index are unavailable.

