"""
axen/coding/test_runner.py
===========================
Test runner and test output parser for the AXEN Coding Agent:
- Executes targeted or full test suites
- Parses pytest and npm test outputs into structured TestResult objects
- ErrorAnalyzer pinpoints failure locations, exception types, and line numbers
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
import sys
from typing import Any, Dict, List, Optional

from .command_runner import CommandResult, CommandRunner
from .project_inspector import ProjectInspector
from .workspace import Workspace

logger = logging.getLogger("axen.coding.testing")


@dataclass
class FailureInfo:
    test_name: str
    error_type: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    message: str = ""


@dataclass
class TestResult:
    __test__ = False
    command: str
    exit_code: int
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration_seconds: float = 0.0
    failures: List[FailureInfo] = field(default_factory=list)
    raw_stdout: str = ""
    raw_stderr: str = ""

    @property
    def all_passed(self) -> bool:
        return self.exit_code == 0 and self.failed == 0 and self.errors == 0


class ErrorAnalyzer:
    """Extracts structured diagnostic information from Python tracebacks and test logs."""

    # Traceback file line matcher: File "path/to/file.py", line 42, in test_func
    _TB_FILE_LINE = re.compile(r'File "(?P<file>[^"]+)", line (?P<line>\d+)(?:, in (?P<func>\w+))?')
    _EXCEPTION_PATTERN = re.compile(r"^(?:E\s+)?(?P<type>[a-zA-Z_]\w*(?:Error|Exception|Warning)):\s*(?P<msg>.*)")
    _ASSERTION_PATTERN = re.compile(r"^E\s+(?P<msg>.*)")

    @classmethod
    def analyze_output(cls, stdout: str, stderr: str) -> List[FailureInfo]:
        """Parse test execution logs to locate root cause errors."""
        failures: list[FailureInfo] = []
        combined = stdout + "\n" + stderr
        lines = combined.splitlines()

        current_file = None
        current_line = None
        current_test = "test"
        last_assertion = ""

        for l in lines:
            # Check for pytest failure headers: ___ test_name ___
            test_match = re.match(r"^_{3,}\s*(?P<name>[\w_]+)\s*_{3,}$", l.strip())
            if test_match:
                current_test = test_match.group("name")

            # Check for collection errors: ERROR collecting path/to/test.py
            err_coll = re.search(r"ERROR collecting\s+([^\s]+)", l)
            if err_coll:
                current_file = err_coll.group(1)
                current_test = current_file

            # Check for traceback file/line
            tb_m = cls._TB_FILE_LINE.search(l)
            if tb_m:
                current_file = tb_m.group("file")
                try:
                    current_line = int(tb_m.group("line"))
                except ValueError:
                    current_line = None

            # Check for pytest assertion line: E  assert x == y
            as_m = cls._ASSERTION_PATTERN.match(l)
            if as_m:
                last_assertion = as_m.group("msg").strip()

            # Check for standard Exception line (with or without 'E ' prefix)
            exc_m = cls._EXCEPTION_PATTERN.match(l.strip())
            if exc_m:
                err_type = exc_m.group("type")
                err_msg = exc_m.group("msg").strip()
                failures.append(
                    FailureInfo(
                        test_name=current_test,
                        error_type=err_type,
                        file_path=current_file,
                        line_number=current_line,
                        message=err_msg or last_assertion,
                    )
                )

        # Fallback if pytest failed without standard python traceback
        if not failures and ("FAILED " in stdout or "failed in" in stdout):
            for l in lines:
                if l.startswith("FAILED "):
                    test_id = l.replace("FAILED ", "").split(" - ")[0].strip()
                    failures.append(
                        FailureInfo(
                            test_name=test_id,
                            error_type="AssertionError",
                            message=last_assertion or "Test assertion failed",
                        )
                    )

        return failures


class TestRunner:
    """Executes tests and parses outcomes into structured results."""
    __test__ = False

    def __init__(self, workspace: Workspace, command_runner: CommandRunner) -> None:
        self.workspace = workspace
        self.runner = command_runner
        self.inspector = ProjectInspector(workspace)

    def run_tests(
        self,
        target_path: Optional[str] = None,
        custom_command: Optional[str] = None,
        timeout: float = 45.0,
    ) -> TestResult:
        """Run tests for the project or a targeted test file."""
        lang, default_cmd = self.inspector.detect_ecosystem()

        if custom_command:
            cmd = custom_command
        elif default_cmd == "pytest" or lang == "python":
            python_exe = sys.executable
            if target_path:
                cmd = f'"{python_exe}" -m pytest {target_path} -v'
            else:
                cmd = f'"{python_exe}" -m pytest -v'
        elif default_cmd:
            cmd = f"{default_cmd} {target_path or ''}".strip()
        else:
            return TestResult(
                command="unknown",
                exit_code=1,
                raw_stderr="No test runner detected for this project.",
            )

        cmd_res = self.runner.run(cmd, timeout=timeout)
        return self._parse_result(cmd_res)

    def _parse_result(self, cmd_res: CommandResult) -> TestResult:
        """Parse raw command execution output into a structured TestResult."""
        out = cmd_res.stdout
        passed = 0
        failed = 0
        skipped = 0
        errors = 0

        # Pytest summary line parser: "== 5 passed, 1 failed, 2 skipped in 1.23s =="
        summary_m = re.search(r"==+\s*(.*?)\s+in\s+[\d\.]+s\s*==+", out)
        if summary_m:
            summary_str = summary_m.group(1)
            for part in summary_str.split(","):
                part = part.strip()
                if "passed" in part:
                    p_num = re.search(r"(\d+)\s+passed", part)
                    if p_num:
                        passed = int(p_num.group(1))
                elif "failed" in part:
                    f_num = re.search(r"(\d+)\s+failed", part)
                    if f_num:
                        failed = int(f_num.group(1))
                elif "skipped" in part:
                    s_num = re.search(r"(\d+)\s+skipped", part)
                    if s_num:
                        skipped = int(s_num.group(1))
                elif "error" in part:
                    e_num = re.search(r"(\d+)\s+error", part)
                    if e_num:
                        errors = int(e_num.group(1))
        else:
            # Fallback heuristic: check if exit code was 0
            if cmd_res.exit_code == 0:
                passed = max(1, out.count("PASSED") or out.count("ok"))
            else:
                failed = max(1, out.count("FAILED") or out.count("FAIL"))

        failures = ErrorAnalyzer.analyze_output(cmd_res.stdout, cmd_res.stderr)

        return TestResult(
            command=cmd_res.command,
            exit_code=cmd_res.exit_code,
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            duration_seconds=cmd_res.duration_seconds,
            failures=failures,
            raw_stdout=cmd_res.stdout,
            raw_stderr=cmd_res.stderr,
        )
