from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class ApiContractTest(unittest.TestCase):
    def test_health_and_recommendation_contracts(self) -> None:
        try:
            from fastapi.testclient import TestClient
            from realtime_recs.api import app
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"FastAPI test dependencies unavailable: {exc}")

        client = TestClient(app)
        health = client.get("/healthz")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")

        response = client.post(
            "/api/recommendations",
            json={"user_id": "u_alex", "k": 5, "candidate_pool": 20, "context": {}, "use_cache": False},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["recommendations"]), 5)
        self.assertIn("latency_ms", payload)
        self.assertIn("diagnostics", payload)


if __name__ == "__main__":
    unittest.main()
