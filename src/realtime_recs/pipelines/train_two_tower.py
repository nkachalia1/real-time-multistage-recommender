from __future__ import annotations

import argparse
import json
from pathlib import Path

from realtime_recs.config import get_settings
from realtime_recs.data.bootstrap import bootstrap_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a compact PyTorch two-tower retrieval model.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("data/artifacts"))
    args = parser.parse_args()

    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is not installed. Install training dependencies with "
            "`pip install -r requirements-train.txt`."
        ) from exc

    settings = get_settings()
    items, users = bootstrap_catalog(dim=settings.embedding_dim)
    item_index = {item.item_id: idx for idx, item in enumerate(items)}
    user_index = {user.user_id: idx for idx, user in enumerate(users.values())}

    positives: list[tuple[int, int]] = []
    for user in users.values():
        for event in user.recent_events:
            if event.event_type in {"click", "watch", "complete", "like", "share"}:
                positives.append((user_index[user.user_id], item_index[event.item_id]))

    class TwoTower(nn.Module):
        def __init__(self, user_count: int, item_vectors: list[list[float]], dim: int) -> None:
            super().__init__()
            self.user_embedding = nn.Embedding(user_count, dim)
            self.item_projection = nn.Linear(dim, dim)
            self.register_buffer("item_vectors", torch.tensor(item_vectors, dtype=torch.float32))

        def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
            user_vec = F.normalize(self.user_embedding(user_ids), dim=1)
            item_vec = F.normalize(self.item_projection(self.item_vectors[item_ids]), dim=1)
            return (user_vec * item_vec).sum(dim=1)

    model = TwoTower(
        user_count=len(users),
        item_vectors=[item.embedding for item in items],
        dim=settings.embedding_dim,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.01)
    pairs = torch.tensor(positives, dtype=torch.long)

    for epoch in range(args.epochs):
        order = torch.randperm(len(pairs))
        epoch_loss = 0.0
        for idx in order:
            user_id, positive_item_id = pairs[idx]
            negative_item_id = torch.randint(low=0, high=len(items), size=(1,)).squeeze(0)
            positive_score = model(user_id.view(1), positive_item_id.view(1))
            negative_score = model(user_id.view(1), negative_item_id.view(1))
            loss = -F.logsigmoid(positive_score - negative_score).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
        print(f"epoch={epoch + 1} loss={epoch_loss / len(pairs):.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "two_tower.pt")
    metadata = {
        "model_version": settings.model_version,
        "embedding_dim": settings.embedding_dim,
        "users": list(user_index),
        "items": list(item_index),
        "objective": "BPR pairwise retrieval loss",
    }
    (args.output_dir / "two_tower_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"saved artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()

