from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from realtime_recs.service import RecommendationService


service = RecommendationService.create_demo()

app = FastAPI(
    title="Real-Time Multi-Stage Recommendation System",
    description=(
        "TikTok/YouTube-style recommendation demo: retrieval, ranking, re-ranking, "
        "real-time feedback, monitoring, and Render-ready serving."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendationRequest(BaseModel):
    user_id: str = Field(..., examples=["u_alex"])
    k: int = Field(default=10, ge=1, le=25)
    candidate_pool: int = Field(default=60, ge=10, le=100)
    context: dict[str, str] = Field(
        default_factory=dict,
        examples=[{"device": "web", "region": "US", "onboarding_topics": "ai,design"}],
    )
    use_cache: bool = True


class EventRequest(BaseModel):
    user_id: str = Field(..., examples=["u_alex"])
    item_id: str = Field(..., examples=["vid_001"])
    event_type: str = Field(..., examples=["like"])
    value: float = Field(default=1.0, ge=0.0, le=1.5)


def serialize_ranked_item(item) -> dict[str, Any]:
    return {
        "item_id": item.item.item_id,
        "title": item.item.title,
        "creator_id": item.item.creator_id,
        "category": item.item.category,
        "tags": item.item.tags,
        "score": round(item.score, 4),
        "retrieval_score": round(item.retrieval_score, 4),
        "ranker_score": round(item.ranker_score, 4),
        "diversity_score": round(item.diversity_score, 4),
        "freshness_score": round(item.freshness_score, 4),
        "explore": item.explore,
        "reasons": item.reasons,
        "objectives": {
            "p_ctr": round(item.features.get("p_ctr", 0.0), 4),
            "p_long_watch": round(item.features.get("p_long_watch", 0.0), 4),
            "p_retention": round(item.features.get("p_retention", 0.0), 4),
        },
    }


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    return DASHBOARD_HTML


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "model_version": service.settings.model_version}


@app.get("/readyz")
def readyz() -> dict[str, Any]:
    snapshot = service.catalog.snapshot()
    return {
        "status": "ready",
        "items": snapshot.item_count,
        "users": snapshot.user_count,
        "categories": snapshot.categories,
    }


@app.get("/api/users")
def list_users() -> list[dict[str, Any]]:
    return [
        {
            "user_id": user.user_id,
            "display_name": user.display_name,
            "topics": user.onboarding_topics,
            "preferred_categories": user.preferred_categories,
            "recent_events": len(user.recent_events),
        }
        for user in service.catalog.users()
    ]


@app.get("/api/catalog")
def catalog(category: str | None = None, limit: int = Query(default=80, ge=1, le=120)) -> list[dict[str, Any]]:
    items = service.catalog.all_items()
    if category:
        items = [item for item in items if item.category == category]
    return [
        {
            "item_id": item.item_id,
            "title": item.title,
            "creator_id": item.creator_id,
            "category": item.category,
            "tags": item.tags,
            "quality_score": item.quality_score,
            "prior_ctr": item.prior_ctr,
            "age_hours": item.age_hours,
        }
        for item in items[:limit]
    ]


@app.post("/api/recommendations")
def recommend(payload: RecommendationRequest) -> dict[str, Any]:
    response = service.recommend(
        user_id=payload.user_id,
        k=payload.k,
        candidate_pool=payload.candidate_pool,
        context=payload.context,
        use_cache=payload.use_cache,
    )
    return {
        "user_id": response.user_id,
        "request_id": response.request_id,
        "model_version": response.model_version,
        "latency_ms": response.latency_ms,
        "cache_hit": response.cache_hit,
        "candidates_considered": response.candidates_considered,
        "diagnostics": response.diagnostics,
        "recommendations": [serialize_ranked_item(item) for item in response.recommendations],
    }


@app.post("/api/events")
def ingest_event(payload: EventRequest) -> dict[str, Any]:
    allowed_events = {"impression", "click", "watch", "complete", "like", "share", "hide", "dislike"}
    if payload.event_type not in allowed_events:
        raise HTTPException(status_code=422, detail=f"event_type must be one of {sorted(allowed_events)}")
    try:
        user = service.ingest_event(
            user_id=payload.user_id,
            item_id=payload.item_id,
            event_type=payload.event_type,
            value=payload.value,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "status": "accepted",
        "user_id": user.user_id,
        "recent_events": [
            {
                "item_id": event.item_id,
                "event_type": event.event_type,
                "value": event.value,
            }
            for event in user.recent_events[:8]
        ],
    }


@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    return service.telemetry()


DASHBOARD_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Real-Time Recommender</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #14213d;
      --muted: #5f6c7b;
      --line: #d8dee8;
      --bg: #f7f9fc;
      --panel: #ffffff;
      --accent: #0f766e;
      --accent-2: #b45309;
      --accent-3: #334155;
      --good: #15803d;
      --bad: #b91c1c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    header {
      padding: 22px clamp(18px, 4vw, 56px);
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      flex-wrap: wrap;
    }
    h1 { margin: 0; font-size: clamp(1.25rem, 2.4vw, 2rem); letter-spacing: 0; }
    .subhead { margin-top: 5px; color: var(--muted); font-size: 0.95rem; }
    main {
      padding: 24px clamp(18px, 4vw, 56px) 42px;
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      gap: 22px;
    }
    aside, section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    aside { padding: 18px; align-self: start; }
    section { overflow: hidden; }
    label { display: block; font-weight: 700; font-size: 0.82rem; margin-bottom: 7px; }
    select, input {
      width: 100%;
      min-height: 40px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      font: inherit;
      color: var(--ink);
      background: white;
    }
    .field { margin-bottom: 16px; }
    .button-row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    button {
      min-height: 40px;
      border: 1px solid #0f766e;
      background: var(--accent);
      color: white;
      border-radius: 6px;
      font-weight: 750;
      cursor: pointer;
    }
    button.secondary {
      border-color: var(--line);
      background: white;
      color: var(--accent-3);
    }
    .pipeline {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      background: #eef6f4;
    }
    .stage {
      border: 1px solid #bedbd5;
      background: white;
      border-radius: 8px;
      padding: 12px;
      min-height: 78px;
    }
    .stage strong { display: block; font-size: 0.86rem; }
    .stage span { color: var(--muted); font-size: 0.8rem; }
    .stats {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      border-bottom: 1px solid var(--line);
    }
    .stat { padding: 14px 18px; border-right: 1px solid var(--line); }
    .stat:last-child { border-right: 0; }
    .stat b { display: block; font-size: 1.15rem; }
    .stat span { color: var(--muted); font-size: 0.78rem; }
    .results {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 14px;
      padding: 18px;
    }
    .item {
      border: 1px solid var(--line);
      border-radius: 8px;
      min-height: 230px;
      padding: 14px;
      display: grid;
      grid-template-rows: auto auto 1fr auto;
      gap: 10px;
      background: white;
    }
    .item h2 { margin: 0; font-size: 1rem; line-height: 1.25; letter-spacing: 0; }
    .meta { display: flex; gap: 8px; flex-wrap: wrap; }
    .chip {
      background: #eef2f7;
      color: #334155;
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 0.74rem;
      font-weight: 700;
    }
    .reason { color: var(--muted); font-size: 0.84rem; line-height: 1.35; }
    .bar {
      height: 8px;
      background: #e5e7eb;
      border-radius: 999px;
      overflow: hidden;
      margin-top: 5px;
    }
    .bar > span { display: block; height: 100%; background: var(--accent-2); width: 0%; }
    .actions { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
    .actions button { min-height: 34px; font-size: 0.78rem; }
    .like { background: var(--good); border-color: var(--good); }
    .hide { background: var(--bad); border-color: var(--bad); }
    @media (max-width: 860px) {
      main { grid-template-columns: 1fr; }
      .pipeline, .stats { grid-template-columns: 1fr; }
      .stat { border-right: 0; border-bottom: 1px solid var(--line); }
      .stat:last-child { border-bottom: 0; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Real-Time Multi-Stage Recommender</h1>
      <div class="subhead">Retrieval to ranking to re-ranking, with live feedback updates.</div>
    </div>
    <a href="/docs">API docs</a>
  </header>
  <main>
    <aside>
      <div class="field">
        <label for="user">User</label>
        <select id="user"></select>
      </div>
      <div class="field">
        <label for="topics">Cold-start topics</label>
        <input id="topics" value="ai,design,education" />
      </div>
      <div class="field">
        <label for="limit">Result count</label>
        <select id="limit">
          <option>8</option>
          <option selected>10</option>
          <option>12</option>
        </select>
      </div>
      <div class="button-row">
        <button id="refresh">Refresh</button>
        <button id="new-user" class="secondary">New user</button>
      </div>
    </aside>
    <section>
      <div class="pipeline">
        <div class="stage"><strong>Retrieval</strong><span>Two-tower user and item embeddings, ANN candidate search.</span></div>
        <div class="stage"><strong>Ranking</strong><span>Multi-objective CTR, watch-time, and retention scoring.</span></div>
        <div class="stage"><strong>Re-ranking</strong><span>Diversity, freshness, and exploration controls.</span></div>
      </div>
      <div class="stats" id="stats"></div>
      <div class="results" id="results"></div>
    </section>
  </main>
  <script>
    const userSelect = document.querySelector("#user");
    const resultBox = document.querySelector("#results");
    const statsBox = document.querySelector("#stats");
    const topicsInput = document.querySelector("#topics");
    const limitSelect = document.querySelector("#limit");

    async function loadUsers() {
      const users = await fetch("/api/users").then(r => r.json());
      userSelect.innerHTML = users.map(u =>
        `<option value="${u.user_id}">${u.display_name} (${u.user_id})</option>`
      ).join("");
    }

    function stat(label, value) {
      return `<div class="stat"><b>${value}</b><span>${label}</span></div>`;
    }

    async function recommend(useCache = true) {
      const payload = {
        user_id: userSelect.value || "u_guest",
        k: Number(limitSelect.value),
        candidate_pool: 60,
        context: { device: "web", region: "US", onboarding_topics: topicsInput.value },
        use_cache: useCache
      };
      const data = await fetch("/api/recommendations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      }).then(r => r.json());

      statsBox.innerHTML = [
        stat("latency", `${data.latency_ms} ms`),
        stat("candidates", data.candidates_considered),
        stat("cache", data.cache_hit ? "hit" : "miss"),
        stat("model", data.model_version.replace("demo-", ""))
      ].join("");

      resultBox.innerHTML = data.recommendations.map((item, index) => `
        <article class="item">
          <div class="meta">
            <span class="chip">#${index + 1}</span>
            <span class="chip">${item.category}</span>
            ${item.explore ? '<span class="chip">explore</span>' : ""}
          </div>
          <h2>${item.title}</h2>
          <div class="reason">${item.reasons.join(" &middot; ")}</div>
          <div>
            <div class="reason">Score ${item.score} &middot; CTR ${item.objectives.p_ctr} &middot; retention ${item.objectives.p_retention}</div>
            <div class="bar"><span style="width:${Math.round(item.score * 100)}%"></span></div>
          </div>
          <div class="actions">
            <button class="like" data-event="like" data-id="${item.item_id}">Like</button>
            <button data-event="watch" data-id="${item.item_id}">Watch</button>
            <button class="hide" data-event="hide" data-id="${item.item_id}">Hide</button>
          </div>
        </article>
      `).join("");
    }

    async function sendEvent(itemId, eventType) {
      await fetch("/api/events", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: userSelect.value,
          item_id: itemId,
          event_type: eventType,
          value: eventType === "hide" ? 0.8 : 1
        })
      });
      await recommend(false);
    }

    document.querySelector("#refresh").addEventListener("click", () => recommend(false));
    document.querySelector("#new-user").addEventListener("click", async () => {
      const id = `u_guest_${Math.floor(Math.random() * 9000 + 1000)}`;
      const option = document.createElement("option");
      option.value = id;
      option.textContent = `Guest (${id})`;
      userSelect.appendChild(option);
      userSelect.value = id;
      await recommend(false);
    });
    resultBox.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-event]");
      if (!button) return;
      sendEvent(button.dataset.id, button.dataset.event);
    });

    loadUsers().then(() => recommend(false));
  </script>
</body>
</html>
"""
