from __future__ import annotations

from dataclasses import dataclass
import os


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    service_name: str = "Real-Time Multi-Stage Recommender"
    env: str = os.getenv("RECSYS_ENV", "local")
    model_version: str = os.getenv("RECSYS_MODEL_VERSION", "demo-two-tower-ranker-v1")
    embedding_dim: int = _int_env("RECSYS_EMBEDDING_DIM", 64)
    cache_ttl_seconds: int = _int_env("RECSYS_CACHE_TTL_SECONDS", 45)
    exploration_rate: float = _float_env("RECSYS_EXPLORATION_RATE", 0.08)
    max_candidate_pool: int = _int_env("RECSYS_MAX_CANDIDATE_POOL", 80)


def get_settings() -> Settings:
    return Settings()

