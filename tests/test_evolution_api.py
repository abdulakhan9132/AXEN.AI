"""
tests/test_evolution_api.py
===========================
Integration tests for FastAPI evolution endpoints.
"""

from fastapi.testclient import TestClient
import pytest

from app.ui.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_evolution_api_status(client):
    response = client.get("/api/evolution/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "total_experiments" in data
    assert "pending_approvals" in data


def test_evolution_api_experiments(client):
    response = client.get("/api/evolution/experiments")
    assert response.status_code == 200
    data = response.json()
    assert "experiments" in data
    assert isinstance(data["experiments"], list)


def test_evolution_api_approvals(client):
    response = client.get("/api/evolution/approvals")
    assert response.status_code == 200
    data = response.json()
    assert "approvals" in data
    assert isinstance(data["approvals"], list)
