"""
axen/tools/tool_registry.py
=============================
The AXEN tool registry — a safe, typed catalogue of callable tools.

Each tool has:
  - name: unique dotted identifier (e.g. "calculator.evaluate")
  - description: plain-English explanation for the task router
  - input_schema: dict describing expected arguments
  - handler: the Python callable that actually performs the work
  - risk: "low", "medium", or "high"
  - enabled: whether the tool is active

The controller uses this registry to:
  1. Look up which tools are available.
  2. Dispatch to the correct handler.
  3. Return a typed ToolResult with success/error status.

SAFETY RULES:
  - High-risk tools require explicit user confirmation before execution.
  - Tools may never invoke the shell arbitrarily.
  - Tool output is NEVER fabricated — it is always the actual return value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ------------------------------------------------------------------
# Data types
# ------------------------------------------------------------------

@dataclass
class ToolDefinition:
    """Describes a single callable tool with rich capability and safety metadata."""
    name: str
    description: str
    handler: Callable[[dict[str, Any]], Any]
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    capabilities: list[str] = field(default_factory=list)
    risk_level: str = "low"              # "low" | "medium" | "high"
    risk: str = "low"                    # backward compatibility alias
    deterministic: bool = True
    requires_confirmation: bool = False
    supports_batch: bool = False
    supports_verification: bool = True
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.risk != "low" and self.risk_level == "low":
            self.risk_level = self.risk
        elif self.risk_level != "low" and self.risk == "low":
            self.risk = self.risk_level

    def call(self, args: dict[str, Any]) -> "ToolResult":
        """Execute this tool with the given arguments and return a ToolResult."""
        if not self.enabled:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="This tool is currently disabled.",
            )
        try:
            output = self.handler(args)
            return ToolResult(tool_name=self.name, success=True, output=output)
        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(exc),
            )


@dataclass
class ToolResult:
    """The result of executing a tool."""
    tool_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "tool": self.tool_name,
            "success": self.success,
        }
        if self.success:
            result["output"] = self.output
        else:
            result["error"] = self.error
        return result

    @property
    def display(self) -> str:
        """Human-readable summary for the CLI."""
        if self.success:
            return str(self.output)
        return f"Tool error: {self.error}"


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

class ToolRegistry:
    """
    Central registry for all AXEN tools.

    Usage:
        registry = ToolRegistry()
        registry.register(some_tool_definition)
        result = registry.call("calculator.evaluate", {"expression": "2+2"})
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool. Raises ValueError if the name is already taken."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Look up a tool by name."""
        return self._tools.get(name)

    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Execute a tool by name. Returns an error result if not found."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"No tool named '{name}' is registered.",
            )
        return tool.call(args)

    def list_tools(self) -> list[dict[str, Any]]:
        """Return a summary list of all registered tools."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "risk": t.risk,
                "risk_level": t.risk_level,
                "deterministic": t.deterministic,
                "capabilities": t.capabilities,
                "supports_verification": t.supports_verification,
                "enabled": str(t.enabled),
            }
            for t in self._tools.values()
        ]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={list(self._tools.keys())})"
