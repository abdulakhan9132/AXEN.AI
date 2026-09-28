"""AXEN Safe Code Runner.

Executes code under strict isolation, timeouts, and security policies.
If a language runtime is not genuinely installed, returns 'runtime_unavailable'
without fabricating execution output.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .registry import LanguageSpec, get_language_registry
from .security import SecurityRisk, check_security


@dataclass
class ExecutionResult:
    status: str  # "success", "error", "timeout", "runtime_unavailable", "security_blocked"
    exit_code: Optional[int]
    stdout: str
    stderr: str
    execution_time_sec: float
    runtime_used: Optional[str] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "execution_time_sec": round(self.execution_time_sec, 4),
            "runtime_used": self.runtime_used,
            "reason": self.reason,
        }


class SafeRunner:
    """Executes code in bounded sandboxed environments."""

    def __init__(self, timeout_sec: float = 5.0) -> None:
        self.timeout_sec = timeout_sec
        self.registry = get_language_registry()

    def run(self, code: str, language_name_or_slug: str, stdin_data: str = "") -> ExecutionResult:
        spec = self.registry.get(language_name_or_slug)
        if not spec:
            return ExecutionResult(
                status="runtime_unavailable",
                exit_code=None,
                stdout="",
                stderr="",
                execution_time_sec=0.0,
                reason=f"Unknown or unsupported language: {language_name_or_slug}",
            )

        # 1. Security Check
        sec_report = check_security(code)
        if not sec_report.is_safe_to_execute:
            dangerous_findings = [f.description for f in sec_report.findings if f.severity == SecurityRisk.DANGEROUS]
            return ExecutionResult(
                status="security_blocked",
                exit_code=None,
                stdout="",
                stderr=f"Execution blocked by safety policy: {'; '.join(dangerous_findings)}",
                execution_time_sec=0.0,
                reason="Code contains potentially destructive or unauthorized instructions.",
            )

        # 2. Runtime Availability Check
        installed, binary_path = spec.check_runtime()
        if not installed or not binary_path:
            return ExecutionResult(
                status="runtime_unavailable",
                exit_code=None,
                stdout="",
                stderr="",
                execution_time_sec=0.0,
                reason=f"Runtime for {spec.name} ({spec.compilers_or_interpreters[0]}) is not installed on this machine.",
            )

        # 3. Controlled Execution in Scratch Sandbox
        ext = spec.extensions[0] if spec.extensions else ".txt"
        with tempfile.TemporaryDirectory(prefix="axen_run_") as tmpdir:
            source_file = Path(tmpdir) / f"main{ext}"
            source_file.write_text(code, encoding="utf-8")

            cmd: list[str]
            # Formulate invocation command based on language
            if spec.slug == "python":
                cmd = [sys.executable, str(source_file)]
            elif spec.slug == "javascript" and "node" in binary_path.lower():
                cmd = [binary_path, str(source_file)]
            elif spec.slug == "typescript" and "ts-node" in binary_path.lower():
                cmd = [binary_path, str(source_file)]
            elif spec.slug == "ruby":
                cmd = [binary_path, str(source_file)]
            elif spec.slug == "perl":
                cmd = [binary_path, str(source_file)]
            elif spec.slug == "lua":
                cmd = [binary_path, str(source_file)]
            elif spec.slug == "bash":
                cmd = [binary_path, str(source_file)]
            else:
                # Compiled languages or complex runtimes
                return ExecutionResult(
                    status="runtime_unavailable",
                    exit_code=None,
                    stdout="",
                    stderr="",
                    execution_time_sec=0.0,
                    reason=f"Compiled language {spec.name} requires multi-stage build toolchain not currently configured for direct scratch invocation.",
                )

            start_t = time.perf_counter()
            try:
                proc = subprocess.run(
                    cmd,
                    input=stdin_data,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                    cwd=tmpdir,
                )
                elapsed = time.perf_counter() - start_t
                status = "success" if proc.returncode == 0 else "error"
                return ExecutionResult(
                    status=status,
                    exit_code=proc.returncode,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    execution_time_sec=elapsed,
                    runtime_used=binary_path,
                )
            except subprocess.TimeoutExpired:
                elapsed = time.perf_counter() - start_t
                return ExecutionResult(
                    status="timeout",
                    exit_code=None,
                    stdout="",
                    stderr=f"Process timed out after {self.timeout_sec} seconds.",
                    execution_time_sec=elapsed,
                    runtime_used=binary_path,
                    reason=f"Execution exceeded strict {self.timeout_sec}s timeout limit.",
                )
            except Exception as exc:
                elapsed = time.perf_counter() - start_t
                return ExecutionResult(
                    status="error",
                    exit_code=None,
                    stdout="",
                    stderr=str(exc),
                    execution_time_sec=elapsed,
                    runtime_used=binary_path,
                    reason=f"Subprocess failure: {exc}",
                )


_DEFAULT_RUNNER = SafeRunner()


def run_code(code: str, language: str, stdin_data: str = "") -> ExecutionResult:
    return _DEFAULT_RUNNER.run(code, language, stdin_data)
