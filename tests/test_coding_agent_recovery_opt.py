"""
tests/test_coding_agent_recovery_opt.py
======================================
Tests verifying the Coding Agent & RecoveryManager optimization:
- Deterministic classification of Python syntax errors into CODE_SYNTAX_ERROR
- Deterministic classification of Python runtime exceptions into CODE_RUNTIME_ERROR
- Structured error diagnostic extraction (exception type, line number, message)
- Generation of actionable code repair prompts
- Safe retry boundary enforcement (retryable up to max_retries, bounded)
"""

from __future__ import annotations

import pytest

from axen.agent.recovery import FailureType, RecoveryManager


def test_classify_syntax_error():
    rm = RecoveryManager()
    error_msg = '  File "main.py", line 14\n    def broken_func(:\n                    ^\nSyntaxError: invalid syntax'
    ft = rm.classify_failure(error_msg)
    assert ft == FailureType.CODE_SYNTAX_ERROR


def test_classify_runtime_error():
    rm = RecoveryManager()
    error_msg = 'Traceback (most recent call last):\n  File "calc.py", line 22, in calculate\nNameError: name \'total_sum\' is not defined'
    ft = rm.classify_failure(error_msg)
    assert ft == FailureType.CODE_RUNTIME_ERROR


def test_code_diagnostic_extraction():
    rm = RecoveryManager()
    error_msg = '  File "service.py", line 42, in process\nTypeError: unsupported operand type(s) for +: \'int\' and \'str\''
    diag = rm.extract_code_diagnostic(error_msg)
    assert diag["exception_type"] == "TypeError"
    assert diag["line_number"] == 42
    assert "int" in diag["message"]
    assert "type mismatch" in diag["repair_guidance"].lower()


def test_recovery_action_suggests_code_repair():
    rm = RecoveryManager(max_retries_per_tool=2)
    error_msg = '  File "model.py", line 88\nIndentationError: expected an indented block'
    action = rm.get_recovery_action(
        tool_name="python",
        error=error_msg,
        attempts=0,
    )
    assert action["failure_type"] == "code_syntax_error"
    assert action["retryable"] is True
    assert action["recommended_strategy"] == "code_repair"
    assert "repair_prompt" in action
    assert "indentation" in action["repair_prompt"].lower()


def test_retry_boundary_enforced():
    rm = RecoveryManager(max_retries_per_tool=2)
    error_msg = 'SyntaxError: unexpected EOF while parsing (line 5)'
    # Attempt 0: retryable
    action0 = rm.get_recovery_action("python", error_msg, attempts=0)
    assert action0["retryable"] is True

    # Attempt 2: max retries reached, retryable is False
    action2 = rm.get_recovery_action("python", error_msg, attempts=2)
    assert action2["retryable"] is False
