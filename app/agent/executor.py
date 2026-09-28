from __future__ import annotations

from typing import Any, Dict, Optional

from app.agent.tools import ToolRegistry
from app.security.permissions import SecurityPolicy


class Executor:
    """Execute a planned step using the registered tool registry."""

    def __init__(self, tool_registry: ToolRegistry, security_policy: Optional[SecurityPolicy] = None):
        self.tool_registry = tool_registry
        self.security_policy = security_policy or SecurityPolicy()

    def execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        tool_name = step.get("tool")
        arguments = step.get("arguments", {})
        if not tool_name:
            return {"success": False, "error": "Step is missing a tool name."}

        try:
            self.security_policy.validate_tool_call(tool_name, arguments)
        except PermissionError as exc:
            return {"success": False, "error": str(exc), "tool": tool_name}

        result = self.tool_registry.execute(tool_name, arguments)
        if result.get("success") is True:
            return {"success": True, "tool": tool_name, "result": result}
        return {"success": False, "tool": tool_name, "error": result.get("error", "Execution failed")}
