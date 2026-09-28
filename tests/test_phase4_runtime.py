from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.ai.local_model import LocalModel
from app.ai.ollama_doctor import inspect_ollama
from app.agent.task_router import TaskRouter
from reasoning.math.reasoning_controller import ReasoningController


def test_doctor_reports_ollama_unavailable():
    with patch("app.ai.ollama_doctor.shutil.which", return_value=None):
        report = inspect_ollama()

    assert report.ollama_installed is False
    assert report.model_available is False
    assert report.structured_output is False


def test_doctor_reports_installed_but_missing_model():
    response = Mock(status_code=200)
    response.raise_for_status.return_value = None
    response.json.return_value = {"models": [{"name": "other:1b"}]}
    config = SimpleNamespace(model="qwen2.5:1.5b", base_url="http://localhost:11434", timeout=1.0)
    with patch("app.ai.ollama_doctor.shutil.which", return_value="ollama"), patch("app.ai.ollama_doctor.requests.get", return_value=response):
        report = inspect_ollama(config, probe=False)

    assert report.ollama_installed is True
    assert report.service_running is True
    assert report.model_available is False
    assert report.selected_model == "other:1b"


def test_local_model_requests_json_format_and_limits():
    settings = SimpleNamespace(model="tiny", base_url="http://localhost:11434", temperature=0.1, context_length=128, max_output_tokens=32, timeout=3.0)
    tags = Mock(status_code=200)
    tags.raise_for_status.return_value = None
    generated = Mock(status_code=200)
    generated.raise_for_status.return_value = None
    generated.json.return_value = {"response": '{"ok": true}'}
    with patch("app.ai.local_model.requests.get", return_value=tags), patch("app.ai.local_model.requests.post", return_value=generated) as post:
        model = LocalModel(config=settings)
        assert model.structured_output("return JSON") == {"ok": True}

    payload = post.call_args.kwargs["json"]
    assert payload["format"] == "json"
    assert payload["options"]["num_ctx"] == 128
    assert payload["options"]["num_predict"] == 32


def test_task_router_classifies_math_and_unsupported_tasks():
    manager = Mock()
    manager.config = SimpleNamespace(max_strategy_attempts=5, max_formalization_attempts=3)
    manager.health_check.return_value = False
    manager.unavailable_message.return_value = "unavailable"
    router = TaskRouter(model_manager=manager)

    assert router.classify("What is 2 + 2?") == "mathematics"
    assert router.classify("Please delete this file") == "unsupported"
    assert router.route("What is 2 + 2?")["status"] == "llm_unavailable"
    assert router.route("Please delete this file")["status"] == "unsupported_task"
