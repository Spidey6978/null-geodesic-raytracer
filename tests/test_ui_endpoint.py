"""
Module: tests.test_ui_endpoint
Automated unit tests for KERR-TRACE web application HTML/CSS static endpoints.
"""

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_serve_ui_endpoint():
    response = client.get("/ui")
    assert response.status_code == 200
    assert "KERR-TRACE" in response.text
    assert "id=\"viewport-container\"" in response.text


def test_serve_static_css_and_js():
    css_res = client.get("/static/styles.css")
    assert css_res.status_code == 200
    assert "--bg-color: #0F1722;" in css_res.text

    js_res = client.get("/static/app.js")
    assert js_res.status_code == 200
    assert "DOMContentLoaded" in js_res.text
