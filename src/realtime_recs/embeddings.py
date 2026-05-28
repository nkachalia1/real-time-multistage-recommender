from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from realtime_recs.domain import Vector
from realtime_recs.vector import hashed_text_embedding, normalize


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> Vector:
        ...

    def embed_batch(self, texts: list[str]) -> list[Vector]:
        ...


@dataclass(frozen=True)
class HashingEmbeddingProvider:
    dim: int = 64

    def embed(self, text: str) -> Vector:
        return hashed_text_embedding(text, dim=self.dim)

    def embed_batch(self, texts: list[str]) -> list[Vector]:
        return [self.embed(text) for text in texts]


@dataclass(frozen=True)
class OpenAIEmbeddingProvider:
    """Optional semantic embedding provider for offline item-vector materialization."""

    api_key: str
    model: str = "text-embedding-3-small"
    dim: int = 64
    endpoint: str = "https://api.openai.com/v1/embeddings"

    def embed(self, text: str) -> Vector:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []
        payload = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float",
        }
        if self.model.startswith("text-embedding-3"):
            payload["dimensions"] = self.dim

        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))

        ordered = sorted(body["data"], key=lambda row: row["index"])
        return [normalize([float(value) for value in row["embedding"]]) for row in ordered]


def provider_from_env(dim: int = 64) -> EmbeddingProvider:
    provider = os.getenv("RECSYS_EMBEDDING_PROVIDER", "hashing").lower()
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required when RECSYS_EMBEDDING_PROVIDER=openai")
        return OpenAIEmbeddingProvider(
            api_key=api_key,
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            dim=dim,
        )
    return HashingEmbeddingProvider(dim=dim)

