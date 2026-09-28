from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolDefinition:
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    risk: str
    requires_approval: bool = False
    handler: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None


class ToolRegistry:
    """Central registry for all local tools."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> ToolDefinition:
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list(self) -> List[ToolDefinition]:
        return list(self._tools.values())

    def execute(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        tool = self.get(name)
        if tool is None:
            return {"success": False, "error": f"Unknown tool: {name}"}
        if tool.handler is None:
            return {"success": False, "error": f"Tool {name} has no handler."}
        try:
            payload = arguments or {}
            result = tool.handler(payload)
            if isinstance(result, dict):
                if "success" not in result:
                    result["success"] = True
                return result
            return {"success": True, "result": result}
        except Exception as exc:  # pragma: no cover - defensive
            return {"success": False, "error": str(exc), "tool": name}
