"""
axen/tools/base.py
==================
Canonical Tool Contract for the AXEN Tool & Safe Execution Runtime (Phase 5).

Defines:
  - ToolPermission: Explicit permission types (READ_ONLY, LOCAL_FILE, CODE_EXECUTION, etc.)
  - ToolRiskLevel: Risk categorization (LOW, MEDIUM, HIGH, CRITICAL)
  - ExecutionState: Lifecycle states for tool invocations
  - ToolExecutionContext: Runtime execution parameters (project, user, cancellation, limits)
  - ToolResult: Standardized, typed result of tool execution
  - ToolDefinition / Tool: Canonical tool specification interface
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set


class ToolPermission(str, Enum):
    """Explicit permission requirements for tools. A tool must declare its permission."""
    READ_ONLY = "read_only"
    LOCAL_FILE = "local_file"
    CODE_EXECUTION = "code_execution"
    COMMAND_EXECUTION = "command_execution"
    NETWORK = "network"
    EXTERNAL_ACTION = "external_action"

    @classmethod
    def from_str(cls, value: str) -> "ToolPermission":
        val = str(value).lower().strip()
        for member in cls:
            if member.value == val or member.name.lower() == val:
                return member
        return cls.EXTERNAL_ACTION


class ToolRiskLevel(str, Enum):
    """Risk classification for tool actions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def from_str(cls, value: str) -> "ToolRiskLevel":
        val = str(value).lower().strip()
        for member in cls:
            if member.value == val or member.name.lower() == val:
                return member
        return cls.MEDIUM


class ExecutionState(str, Enum):
    """Lifecycle states of tool execution."""
    PENDING = "pending"
    AUTHORIZED = "authorized"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class ToolExecutionContext:
    """
    Contextual boundary provided to a tool during invocation.
    Enforces project isolation, cancellation events, and resource limits.
    """
    project_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: str = "default_user"
    cancel_event: Optional[threading.Event] = None
    authorized_permissions: Set[ToolPermission] = field(default_factory=set)
    workspace_path: Optional[str] = None
    timeout_seconds: Optional[float] = None
    max_output_bytes: int = 100_000  # Default 100 KB limit for safe truncation
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_cancelled(self) -> bool:
        """Check if execution has been requested to cancel."""
        return bool(self.cancel_event and self.cancel_event.is_set())

    def is_permission_authorized(self, permission: ToolPermission) -> bool:
        """
        Check if the context grants the required permission.
        If authorized_permissions is empty, default permissive policy applies unless restricted.
        """
        if not self.authorized_permissions:
            return True
        return permission in self.authorized_permissions


@dataclass
class ToolResult:
    """
    Standardized typed result of executing an AXEN tool.
    Tool output is strictly treated as UNTRUSTED DATA.
    """
    tool_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_id: str = field(default_factory=lambda: f"exec_{uuid.uuid4().hex[:12]}")
    state: ExecutionState = ExecutionState.COMPLETED
    duration_seconds: float = 0.0
    is_trusted_metadata: bool = False  # Output is untrusted data unless explicitly system-generated

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        res: Dict[str, Any] = {
            "tool": self.tool_name,
            "success": self.success,
            "execution_id": self.execution_id,
            "state": self.state.value if isinstance(self.state, ExecutionState) else str(self.state),
            "duration_seconds": round(self.duration_seconds, 4),
            "metadata": self.metadata,
        }
        if self.success:
            res["output"] = self.output
        else:
            res["error"] = self.error
        return res

    def __getitem__(self, key: str) -> Any:
        if isinstance(self.output, dict) and key in self.output:
            return self.output[key]
        d = self.to_dict()
        if key in d:
            return d[key]
        if key == "result":
            return self.output
        if key in self.metadata:
            return self.metadata[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        d = self.to_dict()
        return key in d or key == "result" or (isinstance(self.output, dict) and key in self.output)

    @property
    def display(self) -> str:
        """Human-readable string representation."""
        if self.state == ExecutionState.CANCELLED:
            return f"Tool execution '{self.tool_name}' was cancelled."
        if self.state == ExecutionState.TIMEOUT:
            return f"Tool execution '{self.tool_name}' timed out."
        if self.success:
            if isinstance(self.output, dict):
                return str(self.output.get("display") or self.output.get("stdout") or self.output)
            return str(self.output)
        return f"Tool error ({self.tool_name}): {self.error}"


@dataclass
class ToolDefinition:
    """
    Canonical tool contract for AXEN.
    Every tool must specify its identity, contract, permission, and execution handler.
    """
    name: str
    description: str
    handler: Optional[Callable[..., Any]] = None
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    result_schema: Dict[str, Any] = field(default_factory=dict)
    permission: ToolPermission = ToolPermission.READ_ONLY
    risk_level: str = "low"
    risk: str = "low"  # backward compatibility alias
    capabilities: List[str] = field(default_factory=list)
    deterministic: bool = True
    requires_confirmation: bool = False
    requires_approval: bool = False  # backward compatibility alias
    supports_batch: bool = False
    supports_verification: bool = True
    enabled: bool = True

    def __post_init__(self) -> None:
        # Normalize risk & risk_level
        if self.risk != "low" and self.risk_level == "low":
            self.risk_level = self.risk
        elif self.risk_level != "low" and self.risk == "low":
            self.risk = self.risk_level

        # Normalize schema aliases
        if self.output_schema and not self.result_schema:
            self.result_schema = self.output_schema
        elif self.result_schema and not self.output_schema:
            self.output_schema = self.result_schema

        # Normalize approval flags
        if self.requires_approval and not self.requires_confirmation:
            self.requires_confirmation = True
        elif self.requires_confirmation and not self.requires_approval:
            self.requires_approval = True

        # Normalize permission type
        if isinstance(self.permission, str):
            self.permission = ToolPermission.from_str(self.permission)

    def execute(self, arguments: Dict[str, Any], context: Optional[ToolExecutionContext] = None) -> ToolResult:
        """
        Execute tool with canonical contract and execution context.
        """
        if not self.enabled:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="This tool is currently disabled.",
                state=ExecutionState.FAILED,
            )

        if self.handler is None:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Tool '{self.name}' has no execution handler.",
                state=ExecutionState.FAILED,
            )

        try:
            # Check if handler accepts context
            import inspect
            sig = inspect.signature(self.handler)
            if "context" in sig.parameters:
                raw_res = self.handler(arguments, context=context)
            else:
                raw_res = self.handler(arguments)

            if isinstance(raw_res, ToolResult):
                return raw_res

            if isinstance(raw_res, dict):
                success = raw_res.get("success", True)
                error = raw_res.get("error") if not success else None
                cancelled = bool(raw_res.get("cancelled", False)) or (context is not None and context.is_cancelled())
                timeout = bool(raw_res.get("timeout", False))
                state = ExecutionState.COMPLETED if success else (
                    ExecutionState.CANCELLED if cancelled else (
                        ExecutionState.TIMEOUT if timeout else ExecutionState.FAILED
                    )
                )
                return ToolResult(
                    tool_name=self.name,
                    success=success,
                    output=raw_res if success else None,
                    error=error,
                    state=state,
                    metadata={"raw": raw_res},
                )

            return ToolResult(
                tool_name=self.name,
                success=True,
                output=raw_res,
                state=ExecutionState.COMPLETED,
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(exc),
                state=ExecutionState.FAILED,
            )

    def call(self, args: Dict[str, Any]) -> ToolResult:
        """Backward-compatible call interface."""
        return self.execute(args)


# Type alias for clarity
Tool = ToolDefinition
