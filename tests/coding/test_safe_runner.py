"""Tests for AXEN Safe Code Runner."""

from __future__ import annotations

import pytest
from reasoning.coding.runner import run_code


def test_safe_python_execution():
    code = "import sys\nprint('HELLO FROM SANDBOX')\n"
    res = run_code(code, "python")
    assert res.status == "success"
    assert "HELLO FROM SANDBOX" in res.stdout
    assert res.exit_code == 0
    assert res.execution_time_sec >= 0.0


def test_dangerous_code_blocked_from_execution():
    code = "import os\nos.system('del /s /q C:\\Windows')"
    res = run_code(code, "python")
    assert res.status == "security_blocked"
    assert "blocked" in res.stderr.lower()
    assert res.exit_code is None


def test_missing_runtime_reports_unavailable_honestly():
    # Attempting to run SAS or Mojo when not installed
    code = "data test; set input; run;"
    res = run_code(code, "sas")
    assert res.status == "runtime_unavailable"
    assert "not installed" in res.reason.lower() or "requires" in res.reason.lower()
    assert res.exit_code is None


def test_execution_timeout_enforced():
    infinite_loop = "import time\nwhile True:\n    time.sleep(0.1)\n"
    # Use small timeout for testing
    from reasoning.coding.runner import SafeRunner
    runner = SafeRunner(timeout_sec=0.5)
    res = runner.run(infinite_loop, "python")
    assert res.status == "timeout"
    assert "timed out" in res.stderr.lower()
