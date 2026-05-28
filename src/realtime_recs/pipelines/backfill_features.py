from __future__ import annotations

import argparse
import json
from pathlib import Path

from realtime_recs.config import get_settings
from realtime_recs.service import RecommendationService


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill demo feature-store snapshots.")
    parser.add_argument("--output", type=Path, default=Path("data/processed/feature_snapshot.json"))
    args = parser.parse_args()

    service = RecommendationService.create_demo(get_settings())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "model_version": service.settings.model_version,
        "users": [
            {
                "user_id": user.user_id,
                "display_name": user.display_name,
                "preferred_categories": user.preferred_categories,
                "recent_event_count": len(user.recent_events),
            }
            for user in service.catalog.users()
        ],
        "items": [
            {
                "item_id": item.item_id,
                "category": item.category,
                "quality_score": item.quality_score,
                "prior_ctr": item.prior_ctr,
                "age_hours": item.age_hours,
            }
            for item in service.catalog.all_items()
        ],
    }
    args.output.write_text(json.dumps(snapshot, indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

