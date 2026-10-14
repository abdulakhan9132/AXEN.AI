"""
tests/test_model_manager.py
===========================
Tests for Unified AI Model Management and Cloud Provider Privacy.
"""

import os
from unittest.mock import MagicMock, patch
import pytest
from starlette.testclient import TestClient

from app.ai.model_manager import ModelManager
from app.ai.cloud_provider import CloudModelProvider
from app.context.models import ModelRequest, AssembledContext
from app.ui.app import app


def test_model_manager_catalog():
    mgr = ModelManager()
    catalog = mgr.list_models()
    assert isinstance(catalog, list)
    assert len(catalog) >= 5

    ids = [m["id"] for m in catalog]
    assert "router" in ids
    assert "own_brain" in ids
    assert "qwen2.5:1.5b" in ids
    assert "tinyllama" in ids
    assert "cloud_openrouter" in ids

    # Router must be active by default
    router_item = next(m for m in catalog if m["id"] == "router")
    assert router_item["status"] == "active"


def test_model_manager_switch():
    mgr = ModelManager()
    assert mgr.active_model_name == "router"
    mgr.switch_model("tinyllama")
    assert mgr.active_model_name == "tinyllama"

    catalog = mgr.list_models()
    tl_item = next(m for m in catalog if m["id"] == "tinyllama")
    assert tl_item["status"] == "active"


def test_cloud_provider_privacy_default():
    # Guarantee local-first privacy: cloud provider is inactive without key
    with patch.dict(os.environ, {}, clear=True):
        cloud = CloudModelProvider(provider="openrouter", api_key=None)
        assert cloud.health_check() is False
        req = ModelRequest(
            context=AssembledContext(messages_for_model=[{"role": "user", "content": "Hello cloud"}]),
            generation_settings={},
            session_metadata={},
        )
        with pytest.raises(RuntimeError) as exc_info:
            cloud.generate(req)
        assert "unavailable" in str(exc_info.value).lower()


def test_cloud_provider_mocked_generate():
    cloud = CloudModelProvider(provider="openrouter", api_key="sk-test-fake-key-12345678")
    assert cloud.health_check() is True

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "Hello from mock cloud!"}}]
    }

    with patch("requests.post", return_value=fake_response) as mock_post:
        req = ModelRequest(
            context=AssembledContext(messages_for_model=[{"role": "user", "content": "Ping"}]),
            generation_settings={},
            session_metadata={},
        )
        reply = cloud.generate(req)
        assert reply == "Hello from mock cloud!"
        mock_post.assert_called_once()


def test_api_models_endpoints():
    client = TestClient(app)

    # 1. GET /api/models
    res = client.get("/api/models")
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    assert "active_model" in data
    assert any(m["id"] == "router" for m in data["models"])

    # 2. POST /api/models/select
    post_res = client.post("/api/models/select", json={"model_id": "tinyllama"})
    assert post_res.status_code == 200
    post_data = post_res.json()
    assert post_data["success"] is True
    assert post_data["active_model"] == "tinyllama"
