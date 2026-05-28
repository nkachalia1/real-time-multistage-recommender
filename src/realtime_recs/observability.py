from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class ServiceMetrics:
    requests: int = 0
    events_ingested: int = 0
    total_latency_ms: float = 0.0
    last_latency_ms: float = 0.0
    errors: int = 0
    stage_counts: dict[str, int] = field(default_factory=dict)

    @property
    def avg_latency_ms(self) -> float:
        return 0.0 if self.requests == 0 else self.total_latency_ms / self.requests

    def observe_request(self, latency_ms: float, candidates: int) -> None:
        self.requests += 1
        self.total_latency_ms += latency_ms
        self.last_latency_ms = latency_ms
        self.stage_counts["retrieval_candidates"] = candidates

    def observe_event(self) -> None:
        self.events_ingested += 1

    def as_dict(self) -> dict[str, float | int | dict[str, int]]:
        return {
            "requests": self.requests,
            "events_ingested": self.events_ingested,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "last_latency_ms": round(self.last_latency_ms, 2),
            "errors": self.errors,
            "stage_counts": dict(self.stage_counts),
        }


class Timer:
    def __enter__(self) -> "Timer":
        self.started = perf_counter()
        self.elapsed_ms = 0.0
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed_ms = (perf_counter() - self.started) * 1000.0

