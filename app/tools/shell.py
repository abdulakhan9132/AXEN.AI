"""
app/tools/shell.py
==================
Safe command execution tool delegate for app.tools (Phase 5).
Replaces raw shell=True with allowlisted execution (python, node, git).
"""

from __future__ import annotations

from app.agent.tools import ToolDefinition
from axen.tools.base import ToolPermission
from axen.tools.command_tool import execute_allowlisted_command


def run_shell_command(args: dict) -> dict:
    """Safely execute command using allowlisted command runner without shell=True."""
    return execute_allowlisted_command(args)


def build_shell_tool() -> ToolDefinition:
    return ToolDefinition(
        name="shell.execute",
        description="Run an allowlisted command (python, node, git) with strict argument validation.",
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "args": {"type": "array"},
                "reason": {"type": "string"},
                "risk": {"type": "string"},
            },
            "required": ["command"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "stdout": {"type": "string"},
                "stderr": {"type": "string"},
                "returncode": {"type": "integer"},
            },
        },
        permission=ToolPermission.COMMAND_EXECUTION,
        risk_level="high",
        risk="high",
        requires_approval=True,
        requires_confirmation=True,
        handler=run_shell_command,
    )
