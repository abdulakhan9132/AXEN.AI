from __future__ import annotations

import subprocess

from app.agent.tools import ToolDefinition


def _validate_command(command: str) -> None:
    blocked = [
        "rm -rf /",
        "shutdown",
        "reboot",
        "powershell -enc",
        "curl http",
        "wget http",
        "format c:",
        "del /s /q",
        "sc delete",
        "net user",
        "mklink",
        "&&",
        "||",
    ]
    lowered = command.lower()
    for token in blocked:
        if token in lowered:
            raise PermissionError(f"Command blocked by policy: {command}")


def run_shell_command(args: dict) -> dict:
    command = args.get("command", "")
    if not command:
        return {"success": False, "error": "Missing command"}
    try:
        _validate_command(command)
        completed = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=20)
        return {
            "success": completed.returncode == 0,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "returncode": completed.returncode,
        }
    except PermissionError as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def build_shell_tool() -> ToolDefinition:
    return ToolDefinition(
        name="shell.execute",
        description="Run a validated shell command when explicitly approved.",
        input_schema={
            "type": "object",
            "properties": {"command": {"type": "string"}, "reason": {"type": "string"}, "risk": {"type": "string"}},
            "required": ["command", "reason"],
        },
        output_schema={"type": "object", "properties": {"stdout": {"type": "string"}, "stderr": {"type": "string"}, "returncode": {"type": "integer"}}},
        risk="high",
        requires_approval=True,
        handler=run_shell_command,
    )
