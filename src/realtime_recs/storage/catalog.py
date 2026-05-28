from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from realtime_recs.domain import Interaction, Item, UserState


@dataclass
class CatalogSnapshot:
    item_count: int
    user_count: int
    event_count: int
    categories: list[str]


class InMemoryCatalog:
    """Thread-safe demo metadata store.

    Production swap: Postgres for item/user metadata and Kafka/S3 for immutable
    event logs. The API surface is intentionally small so the adapter can change.
    """

    def __init__(self, items: list[Item], users: dict[str, UserState]) -> None:
        self._items = {item.item_id: item for item in items}
        self._users = dict(users)
        self._events: list[Interaction] = []
        self._lock = RLock()

    def all_items(self) -> list[Item]:
        with self._lock:
            return list(self._items.values())

    def item(self, item_id: str) -> Item | None:
        with self._lock:
            return self._items.get(item_id)

    def users(self) -> list[UserState]:
        with self._lock:
            return list(self._users.values())

    def user(self, user_id: str) -> UserState | None:
        with self._lock:
            return self._users.get(user_id)

    def upsert_user(self, user: UserState) -> None:
        with self._lock:
            self._users[user.user_id] = user

    def append_event(self, event: Interaction) -> UserState:
        with self._lock:
            user = self._users.get(event.user_id)
            if user is None:
                raise KeyError(f"Unknown user_id: {event.user_id}")
            updated = user.clone_with_event(event)
            self._users[event.user_id] = updated
            self._events.append(event)
            return updated

    def event_count(self) -> int:
        with self._lock:
            return len(self._events)

    def snapshot(self) -> CatalogSnapshot:
        with self._lock:
            categories = sorted({item.category for item in self._items.values()})
            return CatalogSnapshot(
                item_count=len(self._items),
                user_count=len(self._users),
                event_count=len(self._events),
                categories=categories,
            )

