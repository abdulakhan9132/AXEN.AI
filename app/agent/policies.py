from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

log = logging.getLogger("aexn.policy")

# Full access is ON by default. Set AEXN_FULL_ACCESS=0 to bring approvals back.
FULL_ACCESS = os.getenv("AEXN_FULL_ACCESS", "1") != "0"


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
    """Project-level permissions. In full-access mode nothing needs approval."""

    def __init__(self):
        self.policies = {
            "filesystem.read": ToolPolicy("filesystem.read", "Read a file", PermissionLevel.LOW),
            "filesystem.list_dir": ToolPolicy("filesystem.list_dir", "List a folder", PermissionLevel.LOW),
            "filesystem.search": ToolPolicy("filesystem.search", "Search files", PermissionLevel.LOW),
            "filesystem.write": ToolPolicy("filesystem.write", "Write a file", PermissionLevel.MEDIUM),
            "calculator.evaluate": ToolPolicy("calculator.evaluate", "Evaluate arithmetic", PermissionLevel.LOW),
            "shell.execute": ToolPolicy("shell.execute", "Run shell commands", PermissionLevel.HIGH, requires_approval=True),
            "browser.snapshot": ToolPolicy("browser.snapshot", "Inspect browser state", PermissionLevel.MEDIUM),
        }

    def tool_for(self, tool_name: str) -> ToolPolicy:
        return self.policies.get(tool_name, ToolPolicy(tool_name, "custom tool", PermissionLevel.LOW))

    def requires_approval(self, tool_name: str) -> bool:
        if FULL_ACCESS:
            log.info("auto-approved: %s", tool_name)
            return False
        return self.tool_for(tool_name).requires_approval
