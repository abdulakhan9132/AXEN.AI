"""
axen/tools/__init__.py
"""
from .tool_registry import ToolDefinition, ToolRegistry, ToolResult
from .calculator import build_calculator_tool
from .python_tool import build_python_tool
from .file_tool import build_file_tool


def build_default_registry() -> ToolRegistry:
    """Create a ToolRegistry pre-loaded with the default AXEN tools."""
    registry = ToolRegistry()
    registry.register(build_calculator_tool())
    registry.register(build_python_tool())
    registry.register(build_file_tool())
    return registry


__all__ = [
    "ToolDefinition",
    "ToolRegistry",
    "ToolResult",
    "build_calculator_tool",
    "build_python_tool",
    "build_file_tool",
    "build_default_registry",
]
