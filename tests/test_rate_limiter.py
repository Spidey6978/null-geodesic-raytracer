"""
Module: tests.test_rate_limiter
Automated unit tests for sliding-window rate limiting middleware.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.rate_limiter import RateLimitMiddleware

app = FastAPI()
app.add_middleware(RateLimitMiddleware, max_requests=2, window_seconds=60)


@app.post("/api/v1/renders/image")
def sample_render_endpoint():
    return {"status": "ok"}


client = TestClient(app)


def test_rate_limiting_enforcement():
    # Requests 1 and 2 should succeed (200 OK)
    r1 = client.post("/api/v1/renders/image")
    r2 = client.post("/api/v1/renders/image")
    assert r1.status_code == 200
    assert r2.status_code == 200

    # Request 3 within window should be rate-limited (429 Too Many Requests)
    r3 = client.post("/api/v1/renders/image")
    assert r3.status_code == 429
    data = r3.json()
    assert data["error_code"] == "RATE_LIMIT_EXCEEDED"
