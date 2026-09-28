from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class PermissionLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ToolPolicy:
    name: str
    description: str
    risk: PermissionLevel = PermissionLevel.LOW
    requires_approval: bool = False
    input_schema: Dict[str, Any] | None = None
    output_schema: Dict[str, Any] | None = None


class PolicyEngine:
    """Project-level permissions and required approval rules."""

    def __init__(self):
        self.policies = {
            "filesystem.read": ToolPolicy("filesystem.read", "Read a file", PermissionLevel.LOW),
            "filesystem.list_dir": ToolPolicy("filesystem.list_dir", "List a folder", PermissionLevel.LOW),
            "filesystem.search": ToolPolicy("filesystem.search", "Search files", PermissionLevel.LOW),
            "filesystem.write": ToolPolicy("filesystem.write", "Write a file", PermissionLevel.MEDIUM, requires_approval=False),
            "calculator.evaluate": ToolPolicy("calculator.evaluate", "Evaluate arithmetic", PermissionLevel.LOW),
            "shell.execute": ToolPolicy("shell.execute", "Run shell commands", PermissionLevel.HIGH, requires_approval=True),
            "browser.snapshot": ToolPolicy("browser.snapshot", "Inspect browser state", PermissionLevel.MEDIUM),
        }

    def tool_for(self, tool_name: str) -> ToolPolicy:
        return self.policies.get(tool_name, ToolPolicy(tool_name, "custom tool", PermissionLevel.LOW))

    def requires_approval(self, tool_name: str) -> bool:
        return self.tool_for(tool_name).requires_approval
