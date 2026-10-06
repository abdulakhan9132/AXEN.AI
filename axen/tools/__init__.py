"""
axen/tools/__init__.py
======================
Provider-independent AXEN Tool & Safe Execution Runtime (Phase 5).
"""

from .base import (
    ExecutionState,
    Tool,
    ToolDefinition,
    ToolExecutionContext,
    ToolPermission,
    ToolResult,
    ToolRiskLevel,
)
from .calculator import build_calculator_named_tool, build_calculator_tool
from .coding_tools import build_coding_tools
from .command_tool import build_command_tool, build_shell_compat_tool
from .computer.computer_controller import build_computer_tool
from .file_tool import (
    build_file_extended_tools,
    build_file_list_tool,
    build_file_read_tool,
    build_file_search_tool,
    build_file_tool,
)
from .knowledge_tool import build_knowledge_tool
from .python_tool import build_python_named_tool, build_python_tool
from .tool_registry import ToolRegistry
from .verifier import ToolVerifier
from .web_tools import build_web_tools


def build_default_registry() -> ToolRegistry:
    """Create a canonical ToolRegistry pre-loaded with the standard AXEN tools."""
    registry = ToolRegistry()

    # 1. Calculator
    registry.register(build_calculator_tool())
    registry.register(build_calculator_named_tool())

    # 2. Python Execution
    registry.register(build_python_tool())
    registry.register(build_python_named_tool())

    # 3. Allowlisted Command Execution (No shell=True)
    registry.register(build_command_tool())
    registry.register(build_shell_compat_tool())

    # 4. Standardized File Tools (file_read, file_list, file_search, and aliases)
    for f_tool in build_file_extended_tools():
        if not registry.has(f_tool.name):
            registry.register(f_tool)

    # 5. Knowledge Tool
    registry.register(build_knowledge_tool())

    # 6. Coding Tools (for Qwen Specialist / code search & patch)
    for c_tool in build_coding_tools():
        if not registry.has(c_tool.name):
            registry.register(c_tool)

    # 7. Web Tools (extensible placeholder for future Web Agent, currently not executing browser)
    for w_tool in build_web_tools():
        if not registry.has(w_tool.name):
            registry.register(w_tool)

    # 8. Computer Tool
    comp = build_computer_tool()
    if comp is not None and not registry.has(comp.name):
        registry.register(comp)

    return registry


__all__ = [
    "ToolDefinition",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "ToolPermission",
    "ToolRiskLevel",
    "ExecutionState",
    "ToolExecutionContext",
    "ToolVerifier",
    "build_calculator_tool",
    "build_calculator_named_tool",
    "build_python_tool",
    "build_python_named_tool",
    "build_command_tool",
    "build_shell_compat_tool",
    "build_file_tool",
    "build_file_read_tool",
    "build_file_list_tool",
    "build_file_search_tool",
    "build_file_extended_tools",
    "build_knowledge_tool",
    "build_coding_tools",
    "build_web_tools",
    "build_computer_tool",
    "build_default_registry",
]
