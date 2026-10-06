"""
axen/tools/command_tool.py
==========================
Safe Allowlisted Command Execution Tool for AXEN (Phase 5).

SAFETY ARCHITECTURE:
  - NEVER uses `shell=True` or `os.system`.
  - ONLY allowlisted binaries: `python`, `node`, `git`.
  - Denies arbitrary shells: powershell, cmd, bash, sh, wscript, cscript.
  - Denies system administration tools: net, sc, reg, format, shutdown, wmic.
  - Denies network inspection tools: curl, wget, nc, telnet, ssh.
  - Rejects command injection and chaining: ;, &&, ||, |, >, <, `, $.
  - Rejects path traversal in arguments.
  - Enforces execution timeout, process cleanup, and cancellation.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import (
    ExecutionState,
    ToolDefinition,
    ToolExecutionContext,
    ToolPermission,
    ToolResult,
    ToolRiskLevel,
)

logger = logging.getLogger("axen.tools.command")

# Strict binary allowlist
ALLOWED_COMMANDS = frozenset({"python", "node", "git"})

# Blocked binary names (explicit denylist for clarity and early failure)
BLOCKED_BINARIES = frozenset({
    "powershell", "powershell.exe", "pwsh", "cmd", "cmd.exe",
    "bash", "sh", "zsh", "wscript", "cscript",
    "reg", "reg.exe", "net", "net.exe", "format", "format.exe",
    "shutdown", "shutdown.exe", "reboot", "wmic", "wmic.exe",
    "sc", "sc.exe", "taskkill", "taskkill.exe",
    "curl", "curl.exe", "wget", "wget.exe", "nc", "telnet",
    "del", "rm", "rmdir", "mklink", "icacls", "takeown",
})

# Dangerous shell metacharacters
_DANGEROUS_CHARS_PATTERN = re.compile(r"[;&|`$><]")


def _validate_command_tokens(command: str, args: List[str]) -> Tuple[str, List[str]]:
    """
    Validate command and arguments.
    Returns (resolved_command, sanitized_args_list) or raises ValueError/PermissionError.
    """
    cmd_clean = command.strip().lower()

    # Check for dangerous shell metacharacters in command
    if _DANGEROUS_CHARS_PATTERN.search(command):
        raise ValueError(f"Dangerous metacharacter detected in command: '{command}'")

    # Normalize command name (strip .exe if present)
    base_name = Path(cmd_clean).name.lower()
    if base_name.endswith(".exe"):
        base_name = base_name[:-4]

    if base_name in BLOCKED_BINARIES or cmd_clean in BLOCKED_BINARIES:
        raise PermissionError(f"Command '{command}' is explicitly blocked by AXEN security policy.")

    if base_name not in ALLOWED_COMMANDS:
        raise PermissionError(
            f"Command '{command}' is not in the allowlist. Allowed commands: {sorted(ALLOWED_COMMANDS)}."
        )

    # Validate each argument against injection and path traversal
    sanitized_args: List[str] = []
    for arg in args:
        arg_str = str(arg)
        if _DANGEROUS_CHARS_PATTERN.search(arg_str):
            raise ValueError(f"Dangerous metacharacter detected in command argument: '{arg_str}'")
        if ".." in arg_str or "/etc" in arg_str.lower() or "\\windows" in arg_str.lower():
            raise ValueError(f"Restricted path or traversal detected in argument: '{arg_str}'")
        sanitized_args.append(arg_str)

    # For python, resolve to sys.executable if running 'python'
    executable = sys.executable if base_name == "python" else base_name
    return executable, sanitized_args


def execute_allowlisted_command(
    args: Dict[str, Any],
    context: Optional[ToolExecutionContext] = None,
) -> Dict[str, Any]:
    """
    Execute an allowlisted command with strict bounds.

    Args:
        args["command"]: binary to execute ('python', 'node', 'git')
        args["args"]: list of arguments or space-delimited string
        args["timeout"]: max duration in seconds (default 10, max 30)
    """
    raw_cmd = str(args.get("command", "")).strip()
    raw_args = args.get("args", [])
    timeout = min(float(args.get("timeout", 10.0)), 30.0)

    if not raw_cmd:
        return {"success": False, "error": "Missing 'command' parameter."}

    # If the user passed full command string in "command" (e.g. "git status")
    if not raw_args and " " in raw_cmd:
        parts = shlex.split(raw_cmd)
        raw_cmd = parts[0]
        raw_args = parts[1:]
    elif isinstance(raw_args, str):
        raw_args = shlex.split(raw_args)
    elif not isinstance(raw_args, list):
        return {"success": False, "error": "'args' parameter must be a list or string."}

    # Security validation
    try:
        executable, sanitized_args = _validate_command_tokens(raw_cmd, raw_args)
    except (ValueError, PermissionError) as exc:
        return {"success": False, "error": str(exc)}

    # Determine execution working directory (must be workspace if specified)
    cwd = None
    if context and context.workspace_path:
        cwd = str(Path(context.workspace_path).resolve())

    # Environment: clean and stripped
    safe_env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONIOENCODING": "utf-8",
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:\\Windows"),
        "TEMP": os.environ.get("TEMP", "C:\\Temp"),
    }

    full_argv = [executable] + sanitized_args

    try:
        proc = subprocess.Popen(
            full_argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=cwd,
            env=safe_env,
            shell=False,  # STRICT: NEVER SHELL=TRUE
        )

        # Polling loop supporting cancellation & timeout
        t0 = time.time()
        stdout_chunks: List[str] = []
        stderr_chunks: List[str] = []

        while proc.poll() is None:
            if context and context.is_cancelled():
                proc.kill()
                return {
                    "success": False,
                    "error": "Command execution was cancelled.",
                    "cancelled": True,
                }
            if time.time() - t0 > timeout:
                proc.kill()
                return {
                    "success": False,
                    "error": f"Command timed out after {timeout} seconds.",
                    "timeout": True,
                }
            time.sleep(0.05)

        stdout, stderr = proc.communicate()
        return {
            "success": proc.returncode == 0,
            "stdout": stdout.strip(),
            "stderr": stderr.strip(),
            "returncode": proc.returncode,
            "command": raw_cmd,
        }

    except FileNotFoundError:
        return {
            "success": False,
            "error": f"Executable '{raw_cmd}' was not found on this system.",
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"Command execution failed: {exc}",
        }


def build_command_tool() -> ToolDefinition:
    """Return the AXEN canonical allowlisted command execution tool."""
    return ToolDefinition(
        name="command.execute",
        description=(
            "Execute an allowlisted command ('python', 'node', 'git') with strict argument validation. "
            "Never uses shell=True. Arbitrary system commands and shell chaining are strictly blocked."
        ),
        handler=execute_allowlisted_command,
        input_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Binary to execute: 'python', 'node', or 'git'.",
                },
                "args": {
                    "type": "array",
                    "description": "List of sanitized arguments.",
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds (default 10, max 30).",
                },
            },
            "required": ["command"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "stdout": {"type": "string"},
                "stderr": {"type": "string"},
                "returncode": {"type": "integer"},
            },
        },
        permission=ToolPermission.COMMAND_EXECUTION,
        risk_level="high",
        risk="high",
        capabilities=["command_execution", "git", "python_runner"],
        deterministic=False,
        requires_confirmation=True,
        requires_approval=True,
        supports_verification=True,
    )


def build_shell_compat_tool() -> ToolDefinition:
    """Backward compatibility alias for 'shell.execute' using the safe command runner."""
    tool = build_command_tool()
    tool.name = "shell.execute"
    tool.description = "Backward compatibility alias for command.execute. Arbitrary shells remain blocked."
    return tool
