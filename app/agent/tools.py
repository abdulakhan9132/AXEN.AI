"""
app/agent/tools.py
==================
Unifies app.agent.tools with the canonical axen.tools Tool Runtime (Phase 5).
Re-exports canonical ToolDefinition and ToolRegistry to eliminate duplicate tool systems.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from axen.tools.base import (
    ExecutionState,
    ToolDefinition,
    ToolExecutionContext,
    ToolPermission,
    ToolResult,
    ToolRiskLevel,
)
from axen.tools.tool_registry import ToolRegistry

__all__ = [
    "ToolDefinition",
    "ToolRegistry",
    "ToolResult",
    "ToolPermission",
    "ToolRiskLevel",
    "ExecutionState",
    "ToolExecutionContext",
]
