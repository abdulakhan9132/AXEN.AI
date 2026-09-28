"""
axen/agent/state.py
===================
Structured AgentState for AXEN Phase 3 Agentic Reasoning Loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AgentState:
    """
    Structured representation of the agent's task state during reasoning.

    Tracks the request, classification, execution plan, tool calls,
    observations, verification, and final answer.
    Does NOT store private chain-of-thought; stores only structured task state.
    """
    user_request: str
    task_type: str = "conversation"
    status: str = "pending"  # "pending", "planning", "executing", "verifying", "completed", "failed", "verification_failed"
    plan: list[dict[str, Any]] = field(default_factory=list)
    current_step: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    verification: Optional[dict[str, Any]] = None
    final_answer: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return clean JSON-serializable dictionary matching Section 1 spec."""
        return {
            "user_request": self.user_request,
            "task_type": self.task_type,
            "status": self.status,
            "plan": self.plan,
            "current_step": self.current_step,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "verification": self.verification,
            "final_answer": self.final_answer,
        }

    def add_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Record a structured tool call."""
        self.tool_calls.append({"tool": tool_name, "arguments": arguments})

    def add_tool_result(self, tool_name: str, result: Any, success: bool = True, error: Optional[str] = None) -> None:
        """Record an observed tool result."""
        res: dict[str, Any] = {
            "tool": tool_name,
            "success": success,
        }
        if success:
            res["result"] = str(result)
        else:
            res["error"] = str(error or "Execution error")
        self.tool_results.append(res)
