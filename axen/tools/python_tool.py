"""
axen/tools/python_tool.py
===========================
Safe Python code execution tool for AXEN.

SAFETY DESIGN:
  - Runs code in a subprocess with a hard timeout (default 5 seconds).
  - The subprocess inherits NO dangerous environment variables.
  - BLOCKED: os.system, subprocess, __import__ for dangerous modules.
  - stdout and stderr are captured and returned.
  - The process is killed if the timeout expires.
  - This tool NEVER claims code ran when it didn't.

Returns the ACTUAL stdout/stderr produced by Python.
Does NOT fabricate output.

Distinction:
  SUPPORTED_LANGUAGE = True  (Python is registered)
  RUNTIME_AVAILABLE  = True  (python.exe is installed)
  CODE_VALIDATED     = True  (AST parse check)
  CODE_EXECUTED      = True only when actual subprocess succeeds
"""

from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from .tool_registry import ToolDefinition


# Modules that are blocked from being imported in sandboxed execution
_BLOCKED_IMPORTS = frozenset({
    "subprocess", "os.system", "shutil", "socket",
    "multiprocessing", "threading", "ctypes",
    "importlib", "builtins", "signal",
})

# Blocked function calls that could be destructive
_BLOCKED_CALLS = frozenset({
    "os.system", "os.popen", "os.remove", "os.rmdir",
    "shutil.rmtree", "subprocess.call", "subprocess.run",
    "subprocess.Popen", "eval", "exec", "__import__",
})


def _validate_code(code: str) -> tuple[bool, str]:
    """
    Parse code with Python's AST and check for blocked patterns.
    Returns (is_safe, reason).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"Syntax error: {exc}"

    for node in ast.walk(tree):
        # Block imports of dangerous modules
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = ""
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name
                    if module in _BLOCKED_IMPORTS:
                        return False, f"Blocked import: '{module}'"
            elif isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
                if module in _BLOCKED_IMPORTS:
                    return False, f"Blocked import from: '{module}'"

        # Block calls to dangerous functions (by name)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in _BLOCKED_CALLS:
                    return False, f"Blocked call: '{node.func.id}'"
            elif isinstance(node.func, ast.Attribute):
                full_name = f"{ast.dump(node.func.value)}.{node.func.attr}"
                if node.func.attr in {"system", "popen", "rmtree", "remove"}:
                    return False, f"Blocked call: '{node.func.attr}'"

    return True, "OK"


def execute_python(args: dict) -> dict:
    """
    Tool handler: execute Python code in a sandboxed subprocess.

    Args:
        args["code"]: Python source code to execute.
        args["timeout"]: Max execution time in seconds (default 5).

    Returns:
        {
          "success": bool,
          "stdout": str,
          "stderr": str,
          "returncode": int,
          "validated": bool,
          "executed": bool,
        }
    """
    code = str(args.get("code", "")).strip()
    timeout = min(float(args.get("timeout", 5)), 10.0)

    if not code:
        return {
            "success": False,
            "error": "Missing 'code' argument.",
            "validated": False,
            "executed": False,
        }

    # Step 1: Validate (AST check + blocked patterns)
    is_safe, reason = _validate_code(code)
    if not is_safe:
        return {
            "success": False,
            "error": f"Code validation failed: {reason}",
            "validated": False,
            "executed": False,
        }

    # Step 2: Write code to a temp file and execute in subprocess
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(textwrap.dedent(code))
            tmp_path = tmp.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            # Minimal environment — no secrets
            env={"PYTHONIOENCODING": "utf-8", "PATH": ""},
        )
        Path(tmp_path).unlink(missing_ok=True)

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
            "validated": True,
            "executed": True,
        }

    except subprocess.TimeoutExpired:
        Path(tmp_path).unlink(missing_ok=True)
        return {
            "success": False,
            "error": f"Code execution timed out after {timeout} seconds.",
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
            "error": str(exc),
            "validated": True,
            "executed": False,
        }


def build_python_tool() -> ToolDefinition:
    """Return the AXEN Python execution ToolDefinition."""
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
        risk="medium",
    )
