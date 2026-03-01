"""Integration tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_list_bins_returns_dict():
    response = client.get("/bins")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_get_unknown_bin_returns_404():
    response = client.get("/bins/UNKNOWN-BIN")
    assert response.status_code == 404

