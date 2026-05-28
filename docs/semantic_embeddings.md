# Semantic Embeddings

The deployed demo uses deterministic local embeddings so the Render service has no external API dependency. For an interview, the production path is to materialize item embeddings offline with a stronger semantic model, then serve only the resulting vectors.

## Providers

- `HashingEmbeddingProvider`: dependency-free local provider used by default.
- `OpenAIEmbeddingProvider`: optional provider for offline semantic item-vector generation.

## OpenAI Offline Flow

```bash
set RECSYS_EMBEDDING_PROVIDER=openai
set OPENAI_API_KEY=...
set OPENAI_EMBEDDING_MODEL=text-embedding-3-small
python -m realtime_recs.pipelines.materialize_item_embeddings --output data/artifacts/item_embeddings.json
```

The provider sends batches to the embeddings endpoint, requests `encoding_format=float`, and uses `dimensions` for `text-embedding-3` models so the output vector size matches the demo ANN interface.

## Production Notes

- Run embedding generation in a batch pipeline, not inside the request path.
- Version item vectors with the item tower or embedding model version.
- Rebuild the ANN index after embedding backfills.
- Monitor embedding norm drift and category-level similarity drift.
- Keep a fallback local/provider cache so new content can still be indexed during provider outages.

