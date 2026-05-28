from __future__ import annotations

import argparse
import json
from pathlib import Path

from realtime_recs.config import get_settings
from realtime_recs.data.bootstrap import generate_items
from realtime_recs.embeddings import provider_from_env


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize item semantic embeddings with the configured provider."
    )
    parser.add_argument("--output", type=Path, default=Path("data/artifacts/item_embeddings.json"))
    args = parser.parse_args()

    settings = get_settings()
    provider = provider_from_env(dim=settings.embedding_dim)
    items = generate_items(dim=settings.embedding_dim)
    texts = [
        f"{item.title}. {item.description}. Category: {item.category}. Tags: {', '.join(item.tags)}"
        for item in items
    ]
    embeddings = provider.embed_batch(texts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "model_version": settings.model_version,
                "embedding_dim": settings.embedding_dim,
                "provider": provider.__class__.__name__,
                "items": [
                    {"item_id": item.item_id, "embedding": embedding}
                    for item, embedding in zip(items, embeddings)
                ],
            },
            indent=2,
        )
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

