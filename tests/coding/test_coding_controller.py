"""Tests for AXEN Coding Controller Pipeline."""

from __future__ import annotations

from unittest.mock import Mock
import pytest
from reasoning.coding.coding_controller import CodingController


def test_classify_coding_tasks():
    controller = CodingController()
    assert controller.classify_task("Write a function in Python to reverse a list") == "generate"
    assert controller.classify_task("Explain how this Dijkstra algorithm works") == "explain"
    assert controller.classify_task("Debug this segmentation fault in C++") == "debug"
    assert controller.classify_task("Review this Pull Request for security vulnerabilities") == "review"
    assert controller.classify_task("Refactor this class to reduce cyclomatic complexity") == "refactor"
    assert controller.classify_task("Optimize this matrix multiplication kernel") == "optimize"
    assert controller.classify_task("Generate unit tests with pytest for this file") == "test"
    assert controller.classify_task("Translate this Python code to Rust") == "translate"


def test_controller_static_review_without_llm():
    controller = CodingController(model_manager=None)
    sample_code = """
    def process_data(items):
        res = []
        for x in items:
            for y in x:
                res.append(y * 2)
        return res
    """
    res = controller.process(
        "Review this code for quality",
        language="python",
        code=sample_code,
    )
    assert res.status == "success"
    assert res.task_type == "review"
    assert res.language == "Python"
    assert res.analysis is not None
    assert "metrics" in res.analysis
    assert res.validation is not None
    assert res.validation["valid"] is True


def test_controller_security_blocks_malicious_input():
    controller = CodingController()
    malicious = "import os\nos.system('del /s /q C:\\Windows')"
    res = controller.process("Execute this script", language="python", code=malicious, execute_safe=True)
    assert res.status == "security_blocked"
    assert "destructive" in res.reason.lower() or "policy" in res.reason.lower()


def test_controller_mock_llm_generation():
    mock_model = Mock()
    mock_model.generate.return_value = "```python\ndef add(a, b):\n    return a + b\n```"
    mock_manager = Mock()
    mock_manager.health_check.return_value = True
    mock_manager.get_model.return_value = mock_model

    controller = CodingController(model_manager=mock_manager)
    res = controller.process("Create an add function in Python", language="python")
    assert res.status == "success"
    assert res.task_type == "generate"
    assert res.code == "def add(a, b):\n    return a + b"
    assert res.validation is not None
    assert res.validation["valid"] is True
