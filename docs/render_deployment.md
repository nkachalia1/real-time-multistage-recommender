# Render Deployment

This repository is ready for Render as a Python web service.

## Included Files

- `render.yaml`: Render Blueprint for a web service.
- `.python-version`: pins Python 3.12.13.
- `requirements.txt`: lightweight serving dependencies.
- `app.py`: ASGI entrypoint exposing `app`.
- `/healthz`: health check endpoint.

## Manual Web Service Settings

If you do not use the Blueprint:

| Setting | Value |
| --- | --- |
| Runtime | Python |
| Build command | `pip install -r requirements.txt && pip install -e .` |
| Start command | `uvicorn app:app --host 0.0.0.0 --port $PORT` |
| Health check path | `/healthz` |
| Python version | `3.12.13` |

## Why Serving Dependencies Are Small

The live API avoids PyTorch and FAISS in `requirements.txt` because those packages make cold builds slower and can create wheel compatibility issues on small instances. The project still includes the PyTorch/FAISS training path in `requirements-train.txt`, and the serving code has a FAISS-compatible retrieval interface.

## Post-Deploy Checks

After Render finishes deploying:

1. Open the service URL.
2. Open `/docs` to confirm FastAPI generated the OpenAPI docs.
3. Open `/healthz` and confirm `{"status":"ok"}`.
4. Run a `POST /api/recommendations` request from Swagger.
5. Send a `POST /api/events` feedback event and verify recommendations refresh.
