from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from realtime_recs.domain import Candidate, Item, Vector
from realtime_recs.vector import dot


class RetrievalIndex(Protocol):
    def search(self, query: Vector, k: int, exclude_item_ids: set[str] | None = None) -> list[Candidate]:
        ...


@dataclass
class BruteForceAnnIndex:
    """Small dependency-free ANN stand-in for Render.

    For a production scale interview answer, this interface maps directly to a
    FAISS IVF/HNSW index or a managed vector database.
    """

    items: list[Item]

    def search(self, query: Vector, k: int, exclude_item_ids: set[str] | None = None) -> list[Candidate]:
        exclude_item_ids = exclude_item_ids or set()
        scored = [
            Candidate(item=item, retrieval_score=dot(query, item.embedding), source="ann_vector")
            for item in self.items
            if item.item_id not in exclude_item_ids
        ]
        scored.sort(key=lambda candidate: candidate.retrieval_score, reverse=True)
        return scored[:k]


class FaissAnnIndex:
    """Optional FAISS adapter used by the training/offline path when installed."""

    def __init__(self, items: list[Item]) -> None:
        try:
            import faiss  # type: ignore
            import numpy as np  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "FAISS index requires optional training dependencies. "
                "Install with: pip install -r requirements-train.txt"
            ) from exc

        self._faiss = faiss
        self._np = np
        self.items = items
        vectors = np.array([item.embedding for item in items], dtype="float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query: Vector, k: int, exclude_item_ids: set[str] | None = None) -> list[Candidate]:
        exclude_item_ids = exclude_item_ids or set()
        query_np = self._np.array([query], dtype="float32")
        scores, indexes = self.index.search(query_np, min(k + len(exclude_item_ids), len(self.items)))
        candidates: list[Candidate] = []
        for score, index in zip(scores[0], indexes[0]):
            item = self.items[int(index)]
            if item.item_id in exclude_item_ids:
                continue
            candidates.append(Candidate(item=item, retrieval_score=float(score), source="faiss_flat_ip"))
            if len(candidates) >= k:
                break
        return candidates

