from __future__ import annotations

from enum import Enum


class PermissionLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SecurityPolicy:
    """Enforce tool permissions and command validation."""

    def __init__(self):
        self.risk_map = {
            "filesystem.read": PermissionLevel.LOW,
            "filesystem.list_dir": PermissionLevel.LOW,
            "filesystem.search": PermissionLevel.LOW,
            "calculator.evaluate": PermissionLevel.LOW,
            "memory.read": PermissionLevel.LOW,
            "filesystem.write": PermissionLevel.MEDIUM,
            "shell.execute": PermissionLevel.HIGH,
            "browser.snapshot": PermissionLevel.MEDIUM,
        }

    def validate_tool_call(self, tool_name: str, arguments: dict) -> bool:
        if tool_name in {"shell.execute"}:
            command = str(arguments.get("command", ""))
            if self._is_dangerous_command(command):
                raise PermissionError(f"Command is denied by policy: {command}")
        if tool_name == "filesystem.write" and arguments.get("path", "").startswith("/"):
            raise PermissionError("Refusing to write outside the workspace root.")
        return True

    @staticmethod
    def _is_dangerous_command(command: str) -> bool:
        lowered = command.lower()
        blocked = [
            "rm -rf /",
            "shutdown",
            "reboot",
            "powershell -enc",
            "curl http",
            "wget http",
            "del /s /q",
            "format c:",
            "mklink",
            "sc delete",
            "net user",
            "start /b",
            "&&",
            "||",
            ";",
        ]
        return any(token in lowered for token in blocked)

    def permission_for(self, tool_name: str) -> PermissionLevel:
        return self.risk_map.get(tool_name, PermissionLevel.LOW)
