"""
axen/tools/tool_registry.py
===========================
The canonical AXEN Tool Registry — a safe, typed, provider-independent Tool Runtime.

Enforces:
  1. Standard Tool Contract (name, description, schemas, permission, risk_level, execute)
  2. Complete Execution Lifecycle:
     REQUEST -> VALIDATE -> AUTHORIZE -> EXECUTE -> LIMIT -> OBSERVE -> RETURN RESULT
  3. Execution states (pending, authorized, running, completed, failed, cancelled, timeout)
  4. Explicit permissions and project sandbox isolation
  5. Strict cancellation and process termination support
  6. Output size limiting and Untrusted Data tagging

SAFETY RULES:
  - Tools may never invoke the shell arbitrarily.
  - The model/runtime never directly executes arbitrary commands.
  - All execution passes through this Tool Runtime.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set

from .base import (
    ExecutionState,
    Tool,
    ToolDefinition,
    ToolExecutionContext,
    ToolPermission,
    ToolResult,
    ToolRiskLevel,
)
from .verifier import ToolVerifier

logger = logging.getLogger("axen.tools.registry")


class ToolRegistry:
    """
    Central authoritative registry for all AXEN tools.

    Usage:
        registry = ToolRegistry()
        registry.register(some_tool)
        result = registry.execute("calculator.evaluate", {"expression": "2+2"}, context=ctx)
    """

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDefinition] = {}
        self._lock = threading.RLock()
        self._execution_history: List[ToolResult] = []
        self._active_executions: Dict[str, ToolExecutionContext] = {}

    def register(self, tool: ToolDefinition) -> ToolDefinition:
        """Register a tool. Raises ValueError if already registered."""
        with self._lock:
            if tool.name in self._tools:
                raise ValueError(f"Tool '{tool.name}' is already registered.")
            self._tools[tool.name] = tool
            return tool

    def unregister(self, name: str) -> bool:
        """Unregister a tool by name."""
        with self._lock:
            if name in self._tools:
                del self._tools[name]
                return True
            return False

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Look up a tool by name."""
        with self._lock:
            return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool exists in registry."""
        with self._lock:
            return name in self._tools

    def list(self) -> List[ToolDefinition]:
        """Return all registered tool definitions."""
        with self._lock:
            return list(self._tools.values())

    def names(self) -> List[str]:
        """Return names of all registered tools."""
        with self._lock:
            return list(self._tools.keys())

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return human-readable metadata summaries of registered tools."""
        with self._lock:
            return [
                {
                    "name": t.name,
                    "description": t.description,
                    "permission": t.permission.value if hasattr(t.permission, "value") else str(t.permission),
                    "risk": t.risk,
                    "risk_level": t.risk_level,
                    "deterministic": t.deterministic,
                    "capabilities": t.capabilities,
                    "supports_verification": t.supports_verification,
                    "enabled": str(t.enabled),
                    "input_schema": t.input_schema,
                }
                for t in self._tools.values()
            ]

    # ------------------------------------------------------------------
    # Canonical Execution Lifecycle
    # ------------------------------------------------------------------

    def execute(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        context: Optional[ToolExecutionContext] = None,
    ) -> ToolResult:
        """
        Execute a tool through the full canonical lifecycle:
        REQUEST -> VALIDATE -> AUTHORIZE -> EXECUTE -> LIMIT -> OBSERVE -> RETURN RESULT

        Args:
            name: tool identifier (e.g. "calculator.evaluate")
            arguments: parameter dictionary
            context: runtime execution context (project_id, cancel_event, limits)

        Returns:
            ToolResult with execution_id, status, and output.
        """
        execution_id = f"exec_{uuid.uuid4().hex[:12]}"
        args = arguments if arguments is not None else {}
        ctx = context or ToolExecutionContext()
        t_start = time.perf_counter()

        # Step 1: REQUEST & CANCELLATION PRE-CHECK
        if ctx.is_cancelled():
            return ToolResult(
                tool_name=name,
                success=False,
                error="Execution was cancelled before tool invocation.",
                state=ExecutionState.CANCELLED,
                execution_id=execution_id,
            )

        tool = self.get(name)

        # Step 2: VALIDATE (Existence, schema, parameter types, project sandbox bounds)
        is_valid, err_msg = ToolVerifier.validate_tool_request(tool, name, args, ctx)
        if not is_valid or tool is None:
            return ToolResult(
                tool_name=name,
                success=False,
                error=err_msg or f"Tool '{name}' failed validation.",
                state=ExecutionState.FAILED,
                execution_id=execution_id,
                duration_seconds=time.perf_counter() - t_start,
            )

        # Step 3: AUTHORIZE (Permission checks)
        if not ctx.is_permission_authorized(tool.permission):
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Unauthorized: Execution context lacks '{tool.permission.value}' permission.",
                state=ExecutionState.FAILED,
                execution_id=execution_id,
                duration_seconds=time.perf_counter() - t_start,
            )

        # Track active execution
        with self._lock:
            self._active_executions[execution_id] = ctx

        # Step 4: EXECUTE
        try:
            # Check cancellation again right before execution
            if ctx.is_cancelled():
                return ToolResult(
                    tool_name=name,
                    success=False,
                    error="Execution cancelled before invocation.",
                    state=ExecutionState.CANCELLED,
                    execution_id=execution_id,
                )

            result = tool.execute(args, context=ctx)
            result.execution_id = execution_id
            result.duration_seconds = time.perf_counter() - t_start

            # Check if cancelled during execution
            if ctx.is_cancelled() and result.state != ExecutionState.CANCELLED:
                result.state = ExecutionState.CANCELLED
                result.success = False
                result.error = "Execution was cancelled during tool run."

            # Step 5: LIMIT (Resource & output size bounds)
            result = ToolVerifier.sanitize_output(result, max_bytes=ctx.max_output_bytes)

            # Step 6: OBSERVE & RECORD
            with self._lock:
                self._execution_history.append(result)
                if len(self._execution_history) > 200:
                    self._execution_history = self._execution_history[-100:]

            return result

        except TimeoutError as exc:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Tool execution timed out: {exc}",
                state=ExecutionState.TIMEOUT,
                execution_id=execution_id,
                duration_seconds=time.perf_counter() - t_start,
            )
        except Exception as exc:
            return ToolResult(
                tool_name=name,
                success=False,
                error=str(exc),
                state=ExecutionState.FAILED,
                execution_id=execution_id,
                duration_seconds=time.perf_counter() - t_start,
            )
        finally:
            with self._lock:
                self._active_executions.pop(execution_id, None)

    def call(
        self,
        name: str,
        args: Dict[str, Any],
        context: Optional[ToolExecutionContext] = None,
    ) -> ToolResult:
        """Backward-compatible call interface."""
        return self.execute(name, arguments=args, context=context)

    def __len__(self) -> int:
        with self._lock:
            return len(self._tools)

    def __repr__(self) -> str:
        with self._lock:
            return f"ToolRegistry(tools={list(self._tools.keys())})"


__all__ = [
    "ToolRegistry",
    "ToolDefinition",
    "ToolResult",
    "ToolPermission",
    "ToolRiskLevel",
    "ExecutionState",
    "ToolExecutionContext",
    "ToolVerifier",
]
