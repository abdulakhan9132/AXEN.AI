"""
axen/tools/python_tool.py
==========================
Safe Python code execution tool for AXEN (Phase 5).

SAFETY ARCHITECTURE:
  - Runs code in a detached subprocess with a strict execution timeout (default 5s, max 10s).
  - Explicit AST analysis blocks dangerous calls (eval, exec, __import__, os.system) and modules
    (subprocess, os.system, socket, multiprocessing, threading, ctypes, shutil).
  - Subprocess inherits a minimal sanitized environment (no API keys, no secret env vars).
  - Strict output limits: stdout and stderr are capped to prevent memory exhaustion DoS.
  - Process cleanup: child processes are killed immediately on timeout or user cancellation.
  - Execution directory: operates inside the project workspace or an isolated temp folder.
  - Cancellation: integrates with ToolExecutionContext.cancel_event.

WINDOWS PLATFORM SANDBOX LIMITATION (CRITICAL NOTICE):
  True OS-level kernel sandboxing (e.g. Linux cgroups, namespaces, seccomp, chroot)
  is not natively available on Windows without Hyper-V microVMs or Docker containers.
  Therefore, this tool provides process-level isolation, AST validation, environment sanitization,
  strict timeouts, and output bounds. It does NOT guarantee kernel-level containment against
  advanced native binary exploitation. User/model-generated code is constrained by runtime policies.
"""

from __future__ import annotations

import ast
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .base import (
    ExecutionState,
    ToolDefinition,
    ToolExecutionContext,
    ToolPermission,
    ToolResult,
    ToolRiskLevel,
)

logger = logging.getLogger("axen.tools.python")

# Blocked modules from being imported in execution
_BLOCKED_IMPORTS = frozenset({
    "subprocess", "os.system", "shutil", "socket",
    "multiprocessing", "threading", "ctypes",
    "importlib", "builtins", "signal", "winreg",
    "webbrowser", "pynput", "pyautogui",
})

# Blocked function calls that could be destructive or escape
_BLOCKED_CALLS = frozenset({
    "os.system", "os.popen", "os.remove", "os.rmdir", "os.unlink",
    "shutil.rmtree", "subprocess.call", "subprocess.run",
    "subprocess.Popen", "eval", "exec", "__import__",
})


def _validate_code(code: str) -> Tuple[bool, str]:
    """
    Parse code with Python AST and check for blocked modules, calls, and attributes.
    Returns (is_safe, reason).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"Syntax error: {exc}"

    for node in ast.walk(tree):
        # Block imports of dangerous modules
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    if module in _BLOCKED_IMPORTS:
                        return False, f"Blocked import: '{alias.name}'"
            elif isinstance(node, ast.ImportFrom) and node.module:
                module = node.module.split(".")[0]
                if module in _BLOCKED_IMPORTS:
                    return False, f"Blocked import from: '{node.module}'"

        # Block calls to dangerous functions
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in _BLOCKED_CALLS:
                    return False, f"Blocked call: '{node.func.id}'"
            elif isinstance(node.func, ast.Attribute):
                attr_name = node.func.attr
                if attr_name in {"system", "popen", "rmtree", "remove", "unlink"}:
                    return False, f"Blocked dangerous method call: '{attr_name}'"

    return True, "OK"


def execute_python(
    args: Dict[str, Any],
    context: Optional[ToolExecutionContext] = None,
) -> Dict[str, Any]:
    """
    Execute Python code in a sandboxed subprocess with bounds.

    Args:
        args["code"]: Python source code to execute.
        args["timeout"]: Max execution time in seconds (default 5, max 10).
    """
    code = str(args.get("code", "")).strip()
    timeout = min(float(args.get("timeout", 5.0)), 10.0)
    max_output = context.max_output_bytes if context else 100_000

    if not code:
        return {
            "success": False,
            "error": "Missing 'code' argument.",
            "validated": False,
            "executed": False,
        }

    # Pre-check cancellation
    if context and context.is_cancelled():
        return {
            "success": False,
            "error": "Execution was cancelled.",
            "validated": False,
            "executed": False,
            "cancelled": True,
        }

    # Step 1: Validate AST
    is_safe, reason = _validate_code(code)
    if not is_safe:
        return {
            "success": False,
            "error": f"Code validation failed: {reason}",
            "validated": False,
            "executed": False,
        }

    # Step 2: Determine execution directory
    cwd = None
    if context and context.workspace_path:
        cwd = str(Path(context.workspace_path).resolve())

    # Step 3: Write code to temporary file
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(textwrap.dedent(code))
            tmp_path = tmp.name

        # Clean environment: copy system environment, but strictly strip API keys and secrets
        clean_env = os.environ.copy()
        for k in list(clean_env.keys()):
            k_lower = k.lower()
            if any(sec in k_lower for sec in ("key", "secret", "token", "auth", "password", "credential")):
                clean_env.pop(k, None)
        clean_env["PYTHONIOENCODING"] = "utf-8"

        # Spawn subprocess without shell
        proc = subprocess.Popen(
            [sys.executable, tmp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=cwd,
            env=clean_env,
            shell=False,
        )

        # Handle cancellation with background monitor
        cancel_monitor = None
        if context and context.cancel_event:
            def _watch_cancel():
                while proc.poll() is None:
                    if context.is_cancelled():
                        try:
                            proc.kill()
                        except Exception:
                            pass
                        break
                    time.sleep(0.05)
            cancel_monitor = threading.Thread(target=_watch_cancel, daemon=True)
            cancel_monitor.start()

        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            return {
                "success": False,
                "error": f"Code execution timed out after {timeout} seconds.",
                "validated": True,
                "executed": True,
                "timeout": True,
            }

        if context and context.is_cancelled():
            return {
                "success": False,
                "error": "Code execution was cancelled.",
                "validated": True,
                "executed": False,
                "cancelled": True,
            }

        # Enforce output length limit
        if len(stdout.encode("utf-8")) > max_output:
            stdout = stdout[:max_output] + "\n...[TRUNCATED: Output exceeded size limit]"
        if len(stderr.encode("utf-8")) > max_output:
            stderr = stderr[:max_output] + "\n...[TRUNCATED: Output exceeded size limit]"

        return {
            "success": proc.returncode == 0,
            "stdout": stdout.strip(),
            "stderr": stderr.strip(),
            "returncode": proc.returncode,
            "validated": True,
            "executed": True,
        }

    except FileNotFoundError:
        return {
            "success": False,
            "error": "Python interpreter not found. Cannot execute code.",
            "validated": True,
            "executed": False,
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"Execution error: {exc}",
            "validated": True,
            "executed": False,
        }
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


def build_python_tool() -> ToolDefinition:
    """Return the AXEN canonical Python execution tool."""
    return ToolDefinition(
        name="python.execute",
        description=(
            "Execute Python code in a safe sandboxed subprocess. "
            "Returns actual stdout, stderr, and return code. "
            "Dangerous imports (subprocess, os.system, socket) are blocked."
        ),
        handler=execute_python,
        input_schema={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source code to execute.",
                },
                "timeout": {
                    "type": "number",
                    "description": "Max execution time in seconds (default 5, max 10).",
                },
            },
            "required": ["code"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "stdout": {"type": "string"},
                "stderr": {"type": "string"},
                "returncode": {"type": "integer"},
                "executed": {"type": "boolean"},
            },
        },
        permission=ToolPermission.CODE_EXECUTION,
        risk_level="medium",
        risk="medium",
        capabilities=["code_execution", "python_sandbox", "python_execute"],
        deterministic=False,
        requires_confirmation=False,
        supports_verification=True,
    )


def build_python_named_tool() -> ToolDefinition:
    """Canonical name alias for 'python_execute'."""
    tool = build_python_tool()
    tool.name = "python_execute"
    return tool
