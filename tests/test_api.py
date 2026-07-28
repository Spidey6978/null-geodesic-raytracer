"""
Module: tests.test_api
Automated tests for FastAPI endpoints, Pydantic schemas, and engine API.
"""

from fastapi.testclient import TestClient
from api.main import app
from core.config import RenderConfig, BlackHoleConfig, CameraConfig, RenderMode

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Null Geodesic Raytracer API"
    assert data["status"] == "online"


def test_presets_endpoint():
    response = client.get("/api/v1/presets")
    assert response.status_code == 200
    data = response.json()
    assert "hero" in data["presets"]
    assert "luminet_1979" in data["presets"]
    assert "preview" in data["render_modes"]


def test_config_model_validation():
    config = RenderConfig(
        black_hole=BlackHoleConfig(mass=1.0, spin=0.998),
        camera=CameraConfig(preset="hero"),
        mode=RenderMode.PREVIEW,
        frame_time=1.5
    )
    assert config.black_hole.spin == 0.998
    assert config.frame_time == 1.5


def test_submit_render_job_schema():
    payload = {
        "config": {
            "black_hole": {"mass": 1.0, "spin": 0.998},
            "camera": {"preset": "hero"},
            "mode": "preview"
        }
    }
    response = client.post("/api/v1/renders/image", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] in ["QUEUED", "PROCESSING"]
