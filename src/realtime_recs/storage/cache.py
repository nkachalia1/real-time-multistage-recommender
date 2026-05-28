from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Any


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return 0.0 if total == 0 else self.hits / total


class TTLCache:
    def __init__(self, ttl_seconds: int = 45) -> None:
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, tuple[float, Any]] = {}
        self.stats = CacheStats()

    def get(self, key: str) -> Any | None:
        now = time()
        entry = self._store.get(key)
        if entry is None:
            self.stats.misses += 1
            return None
        expires_at, value = entry
        if expires_at < now:
            self._store.pop(key, None)
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time() + self.ttl_seconds, value)

    def invalidate_prefix(self, prefix: str) -> None:
        for key in list(self._store):
            if key.startswith(prefix):
                self._store.pop(key, None)

    def size(self) -> int:
        return len(self._store)

