"""Tests for AXEN Code Translator."""

from __future__ import annotations

import pytest
from reasoning.coding.translator import prepare_translation


def test_prepare_translation_python_to_rust():
    py_code = "def add(a: int, b: int) -> int:\n    return a + b\n"
    res = prepare_translation("python", "rust", py_code)
    assert res.source_language == "Python"
    assert res.target_language == "Rust"
    assert "ownership" in " ".join(res.idiom_notes).lower()
    assert "```rust" in res.system_prompt
    assert "```python" in res.user_prompt


def test_prepare_translation_js_to_ts():
    js_code = "function greet(name) { return 'Hello ' + name; }"
    res = prepare_translation("javascript", "typescript", js_code)
    assert res.source_language == "JavaScript"
    assert res.target_language == "TypeScript"
    assert any("interface" in n.lower() or "type" in n.lower() for n in res.idiom_notes)


def test_prepare_translation_general_pair():
    code = "fn main() {}"
    res = prepare_translation("rust", "go", code)
    assert res.source_language == "Rust"
    assert res.target_language == "Go"
    assert len(res.idiom_notes) >= 1
