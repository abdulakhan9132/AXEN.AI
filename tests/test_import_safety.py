"""
tests/test_import_safety.py
===========================
Regression tests verifying:
1. Importing axen.agent.loop does not eagerly instantiate AxenBrain(), ImageBrain(),
   or execute any network / GPU code.
2. AgentLoop functions properly even when code_brain is unavailable (CodeBrain is None).
3. An exception in the intent router (get_brain().predict) never crashes a user request.
"""

import importlib
import sys
from unittest.mock import MagicMock

import pytest

from axen.memory.conversation_memory import ConversationMemory
from axen.memory.long_term_memory import LongTermMemory
from axen.tools import build_default_registry


def test_import_safety_no_eager_brain_or_gpu_or_network(monkeypatch):
    """Importing axen.agent.loop must never call AxenBrain, ImageBrain, GPU checks, or network."""
    brain_init_mock = MagicMock()
    image_brain_init_mock = MagicMock()
    requests_post_mock = MagicMock()
    urllib_mock = MagicMock()

    import axen_brain
    monkeypatch.setattr(axen_brain.AxenBrain, "__init__", brain_init_mock)

    import image_brain
    monkeypatch.setattr(image_brain.ImageBrain, "__init__", image_brain_init_mock)

    import requests
    monkeypatch.setattr(requests, "post", requests_post_mock)

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", urllib_mock)

    if "axen.agent.loop" in sys.modules:
        import axen.agent.loop
        importlib.reload(axen.agent.loop)
    else:
        import axen.agent.loop

    assert brain_init_mock.call_count == 0, "AxenBrain should not be instantiated at import time"
    assert image_brain_init_mock.call_count == 0, "ImageBrain should not be instantiated at import time"
    assert requests_post_mock.call_count == 0, "Network should not be called at import time"
    assert urllib_mock.call_count == 0, "Network should not be called at import time"


def test_agent_loop_works_when_code_brain_is_none(monkeypatch):
    """AgentLoop should gracefully fall back when CodeBrain is None."""
    import axen.agent.loop as loop_module
    from axen.agent.loop import AgentLoop

    monkeypatch.setattr(loop_module, "CodeBrain", None)

    tools = build_default_registry()
    conv_mem = ConversationMemory()
    long_mem = LongTermMemory(path=None)

    class DummyModel:
        is_loaded = True
        model_name = "mock-llm"
        def chat(self, messages):
            return "def add(a, b): return a + b"
        def chat_with_stats(self, messages):
            return "def add(a, b): return a + b", {"input_tokens": 10, "generated_tokens": 10}

    mock_brain = MagicMock()
    mock_brain.predict.return_value = {"intent": "code", "confidence": 0.95}
    monkeypatch.setattr(loop_module, "get_brain", lambda: mock_brain)

    loop = AgentLoop(
        tools=tools,
        memory=conv_mem,
        long_memory=long_mem,
        model=DummyModel(),
    )

    ans, state = loop.run("write a python function to add two numbers")
    assert ans is not None
    assert isinstance(ans, str)
    assert state.status in ("completed", "error", "executing")


def test_router_exception_does_not_crash_request(monkeypatch):
    """If get_brain().predict raises an exception, the request must not crash."""
    import axen.agent.loop as loop_module
    from axen.agent.loop import AgentLoop

    mock_brain = MagicMock()
    mock_brain.predict.side_effect = RuntimeError("Simulated router hardware / network crash")
    monkeypatch.setattr(loop_module, "get_brain", lambda: mock_brain)

    tools = build_default_registry()
    conv_mem = ConversationMemory()
    long_mem = LongTermMemory(path=None)

    class DummyModel:
        is_loaded = True
        model_name = "mock-llm"
        def chat(self, messages):
            return "The answer is 1200"
        def chat_with_stats(self, messages):
            return "The answer is 1200", {"input_tokens": 10, "generated_tokens": 10}

    loop = AgentLoop(
        tools=tools,
        memory=conv_mem,
        long_memory=long_mem,
        model=DummyModel(),
    )

    ans, state = loop.run("What is 25 * 48?")
    assert ans is not None
    assert isinstance(ans, str)
    assert state.status != "unhandled_exception"
