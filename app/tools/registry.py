from __future__ import annotations

from app.agent.tools import ToolRegistry
from .browser import build_browser_tool
from .calculator import build_calculator_tool
from .filesystem import build_filesystem_tools
from .memory import build_memory_tool
from .shell import build_shell_tool
from .knowledge import build_knowledge_tool


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in build_filesystem_tools():
        registry.register(tool)
    registry.register(build_calculator_tool())
    registry.register(build_shell_tool())
    registry.register(build_browser_tool())
    registry.register(build_memory_tool())
    registry.register(build_knowledge_tool())
    return registry
