# Interview Guide

## One-Minute Pitch

This is a real-time multi-stage recommender. It uses a two-tower retrieval stage to cheaply find semantically relevant candidates, a multi-objective ranker to estimate engagement and retention, and a re-ranking layer to control diversity, freshness, and exploration. The API ingests feedback events and updates recent user features immediately, which demonstrates the core production loop.

## System Design Walkthrough

Start with the scale problem: there may be millions of items, so ranking every item is impossible. Retrieval narrows the search to hundreds of candidates using vector similarity. Ranking then spends more compute on richer features. Re-ranking optimizes the final slate for product constraints that pure CTR ranking misses.

Important phrases to use:

- "Candidate generation trades accuracy for recall and latency."
- "The ranker owns precision, calibration, and multi-objective utility."
- "The re-ranker is where product constraints live."
- "Feature parity between training and serving is non-negotiable."
- "Cold-start and exploration are not edge cases; they are part of the core loop."

## ML Modeling

Retrieval:

- User tower encodes recent events, long-term preferences, context, and onboarding interests.
- Item tower encodes title, tags, category, creator, and semantic metadata.
- Dot product similarity is used for ANN search.

Ranking:

- Features include semantic similarity, category affinity, creator affinity, freshness, quality prior, item CTR prior, seen penalty, and negative affinity.
- Objectives include CTR, long-watch probability, and retention probability.

Re-ranking:

- Penalizes repeated categories and repeated creators.
- Boosts fresh content.
- Reserves exploration slots with epsilon-greedy sampling.

## MLOps Story

Batch:

- Daily retraining on event logs.
- Feature backfills for historical windows.
- Offline evaluation with AUC, precision@k, recall@k, NDCG, and diversity@k.

Streaming:

- User events update recent features within seconds.
- Online feature store prevents stale profiles.
- Event quality checks protect the feedback loop.

Monitoring:

- Latency, errors, cache hit rate, candidate counts.
- Feature drift and embedding norm drift.
- Model drift through calibration and objective deltas.
- Business metrics through A/B tests.

## Tradeoffs

- More candidates improve recall but increase ranker cost.
- Aggressive caching lowers latency but can slow personalization after feedback.
- Optimizing CTR alone can reduce diversity and long-term retention.
- Semantic embeddings improve cold start but need monitoring for domain drift.
- Approximate vector search improves latency but can miss exact nearest neighbors.

