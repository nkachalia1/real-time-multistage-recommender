from __future__ import annotations

import hashlib
import math

from realtime_recs.domain import Vector


def dot(left: Vector, right: Vector) -> float:
    return sum(a * b for a, b in zip(left, right))


def norm(vec: Vector) -> float:
    return math.sqrt(max(dot(vec, vec), 1e-12))


def normalize(vec: Vector) -> Vector:
    length = norm(vec)
    return [v / length for v in vec]


def add_scaled(base: Vector, other: Vector, scale: float) -> Vector:
    return [a + scale * b for a, b in zip(base, other)]


def zeros(dim: int) -> Vector:
    return [0.0] * dim


def hashed_text_embedding(text: str, dim: int = 64) -> Vector:
    """Small deterministic embedding used for the deployable demo.

    The training pipeline can replace this with OpenAI/BERT/PyTorch-produced
    embeddings, but this keeps the Render service dependency-light.
    """

    vec = zeros(dim)
    tokens = [token.strip(".,:;!?()[]{}").lower() for token in text.split()]
    for token in [t for t in tokens if t]:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for offset in range(0, len(digest), 4):
            bucket = int.from_bytes(digest[offset : offset + 2], "big") % dim
            sign = 1.0 if digest[offset + 2] % 2 == 0 else -1.0
            weight = 0.5 + (digest[offset + 3] / 255.0)
            vec[bucket] += sign * weight
    return normalize(vec)


def blend(vectors: list[tuple[Vector, float]], dim: int) -> Vector:
    merged = zeros(dim)
    total_weight = 0.0
    for vec, weight in vectors:
        if weight <= 0:
            continue
        merged = add_scaled(merged, vec, weight)
        total_weight += weight
    if total_weight == 0:
        return normalize([1.0 if i == 0 else 0.0 for i in range(dim)])
    return normalize([v / total_weight for v in merged])

