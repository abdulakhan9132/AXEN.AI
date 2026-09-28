"""Tests for AXEN Syntax Validator."""

from __future__ import annotations

import pytest
from reasoning.coding.validator import validate_code


def test_valid_python_syntax():
    valid = "def greet(name: str) -> str:\n    return f'Hello, {name}'\n"
    res = validate_code(valid, "python")
    assert res.valid is True
    assert res.method == "python_ast"
    assert len(res.errors) == 0


def test_invalid_python_syntax():
    invalid = "def broken(x\n    return x\n"
    res = validate_code(invalid, "python")
    assert res.valid is False
    assert res.method == "python_ast"
    assert len(res.errors) > 0
    assert res.line_number is not None


def test_balanced_delimiters_for_other_languages():
    cpp_code = """
    #include <iostream>
    int main() {
        std::cout << "Hello world" << std::endl;
        return 0;
    }
    """
    res = validate_code(cpp_code, "cpp")
    assert res.valid is True

    unbalanced_cpp = """
    #include <iostream>
    int main() {
        std::cout << "Hello world" << std::endl;
    // Missing closing brace
    """
    res_bad = validate_code(unbalanced_cpp, "cpp")
    assert res_bad.valid is False
    assert any("unclosed" in e.lower() or "delimiter" in e.lower() for e in res_bad.errors)


def test_unknown_language_handled_gracefully():
    res = validate_code("code", "klingon_dialect")
    assert res.valid is False
    assert "Unsupported" in res.errors[0]
