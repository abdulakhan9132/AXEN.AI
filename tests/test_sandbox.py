"""
tests/test_sandbox.py
=====================
Tests for SandboxPolicy deny-list enforcement and SandboxExecutor process isolation.
"""

import pytest
from axen.evolution.sandbox import SandboxExecutor
from axen.evolution.sandbox_policy import SandboxPolicy


def test_sandbox_policy_deny_list():
    policy = SandboxPolicy()

    # Immutable core files must be denied
    assert not policy.is_path_permitted("axen/evolution/sandbox_policy.py")
    assert not policy.is_path_permitted("axen/evolution/approval.py")
    assert not policy.is_path_permitted("app/security/permissions.py")
    assert not policy.is_path_permitted("checkpoints/foundation/best.pt")

    # Safe experiment targets permitted
    assert policy.is_path_permitted("axen/utils/math_helper.py")
    assert policy.is_path_permitted("plugins/custom_tool.py")

    with pytest.raises(PermissionError):
        policy.validate_execution_request("axen/evolution/approval.py")


def test_sandbox_executor_fails_closed_without_container(monkeypatch):
    monkeypatch.setattr("axen.evolution.sandbox.shutil.which", lambda executable: None)
    executor = SandboxExecutor()
    code = "import sys\nprint('HELLO_SANDBOX')\nsys.exit(0)\n"
    res = executor.run_code_snippet(code, timeout_seconds=5.0)

    assert res.is_success is False
    assert "containment unavailable" in res.error
    assert res.exit_code == -1
    assert not res.timed_out


def test_sandbox_executor_no_environment_or_host_fallback():
    executor = SandboxExecutor()
    # Code that attempts to run indefinitely
    code = "import time\ntime.sleep(10)\n"
    res = executor.run_code_snippet(code, timeout_seconds=1.0, env_overrides={"SECRET": "value"})

    assert "Environment overrides forbidden" in res.error
    assert res.is_success is False
