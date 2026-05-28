from __future__ import annotations

import random

from realtime_recs.domain import Interaction, Item, UserState
from realtime_recs.vector import blend, hashed_text_embedding


CATEGORY_BLUEPRINTS: dict[str, list[str]] = {
    "ai": ["agents", "prompting", "evals", "embeddings", "automation"],
    "fitness": ["mobility", "strength", "recovery", "nutrition", "running"],
    "cooking": ["meal prep", "ramen", "spices", "baking", "quick dinner"],
    "finance": ["index funds", "budget", "founders", "markets", "real estate"],
    "travel": ["tokyo", "lisbon", "hidden gems", "budget travel", "rail"],
    "music": ["production", "jazz", "synth", "songwriting", "mixing"],
    "gaming": ["speedrun", "indie", "strategy", "esports", "retro"],
    "education": ["math", "history", "language", "science", "career"],
    "design": ["ux", "systems", "typography", "figma", "accessibility"],
    "home": ["plants", "organization", "renovation", "lighting", "workspace"],
}

TITLE_PATTERNS = [
    "{tag} in 60 seconds",
    "The hidden playbook for {tag}",
    "I tried {tag} for a week",
    "{tag}: beginner to advanced",
    "Why experts care about {tag}",
    "A practical guide to {tag}",
    "The mistake everyone makes with {tag}",
    "{tag} explained with examples",
]


def generate_items(count_per_category: int = 8, dim: int = 64) -> list[Item]:
    random.seed(42)
    items: list[Item] = []
    for category, tags in CATEGORY_BLUEPRINTS.items():
        for index in range(count_per_category):
            tag = tags[index % len(tags)]
            title = TITLE_PATTERNS[(index + len(category)) % len(TITLE_PATTERNS)].format(tag=tag)
            extended_tags = [tag, category, tags[(index + 2) % len(tags)]]
            description = (
                f"{title}. Short-form {category} content for people interested in "
                f"{', '.join(extended_tags)}."
            )
            embedding = hashed_text_embedding(
                f"{title} {category} {' '.join(extended_tags)} {description}",
                dim=dim,
            )
            item_number = len(items) + 1
            items.append(
                Item(
                    item_id=f"vid_{item_number:03d}",
                    title=title,
                    creator_id=f"creator_{(item_number % 17) + 1:02d}",
                    category=category,
                    tags=extended_tags,
                    description=description,
                    age_hours=float((index * 9 + len(category) * 3) % 168),
                    quality_score=round(0.62 + random.random() * 0.35, 3),
                    prior_ctr=round(0.035 + random.random() * 0.11, 3),
                    embedding=embedding,
                )
            )
    return items


def generate_users(items: list[Item], dim: int = 64) -> dict[str, UserState]:
    users: dict[str, tuple[str, list[str], dict[str, float]]] = {
        "u_alex": ("Alex", ["ai", "design", "automation"], {"ai": 1.0, "design": 0.7}),
        "u_maya": ("Maya", ["fitness", "cooking", "travel"], {"fitness": 0.9, "cooking": 0.8}),
        "u_jules": ("Jules", ["music", "gaming", "education"], {"music": 0.9, "gaming": 0.6}),
        "u_sam": ("Sam", ["finance", "home", "ai"], {"finance": 0.8, "home": 0.6}),
    }

    item_by_category = _group_items(items)
    result: dict[str, UserState] = {}
    for user_id, (name, topics, category_weights) in users.items():
        vectors = [
            (hashed_text_embedding(f"{topic} personalized preference", dim=dim), 1.0)
            for topic in topics
        ]
        for category, weight in category_weights.items():
            category_items = item_by_category.get(category, [])[:3]
            vectors.extend((item.embedding, weight) for item in category_items)

        recent_events: list[Interaction] = []
        for category, weight in category_weights.items():
            for item in item_by_category.get(category, [])[:2]:
                recent_events.append(
                    Interaction(
                        user_id=user_id,
                        item_id=item.item_id,
                        event_type="like" if weight >= 0.8 else "click",
                        value=weight,
                    )
                )

        result[user_id] = UserState(
            user_id=user_id,
            display_name=name,
            onboarding_topics=topics,
            preferred_categories=category_weights,
            long_term_embedding=blend(vectors, dim=dim),
            recent_events=recent_events,
        )
    return result


def _group_items(items: list[Item]) -> dict[str, list[Item]]:
    grouped: dict[str, list[Item]] = {}
    for item in items:
        grouped.setdefault(item.category, []).append(item)
    return grouped


def bootstrap_catalog(dim: int = 64) -> tuple[list[Item], dict[str, UserState]]:
    items = generate_items(dim=dim)
    users = generate_users(items, dim=dim)
    return items, users

