"""
axen/agent/state.py
===================
Structured AgentState and Canonical AgentRun for AXEN Phase 6 Multi-Step Agent Loop.

Enforces:
- Canonical AgentRun state tracking
- Checkpoints without private chain-of-thought
- Hard limits / bounds
- Failure and error tracking
- Tool calls and observations isolation
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import Enum
import uuid
from typing import Any, Dict, List, Optional


class AgentRunState(str, Enum):
    """Canonical lifecycle states for an AgentRun."""
    CREATED = "CREATED"
    UNDERSTANDING = "UNDERSTANDING"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    OBSERVING = "OBSERVING"
    VERIFYING = "VERIFYING"
    RECOVERING = "RECOVERING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"


@dataclass
class AgentCheckpoint:
    """
    State snapshot of an AgentRun at a specific step.
    Contains enough state to understand and recover a run without exposing
    private chain-of-thought or raw secrets.
    """
    step_number: int
    current_state: str
    completed_tool_calls: List[Dict[str, Any]]
    observations: List[Dict[str, Any]]
    errors: List[str]
    remaining_steps: int
    remaining_tool_budget: int
    timestamp: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "current_state": self.current_state,
            "completed_tool_calls": self.completed_tool_calls,
            "observations": self.observations,
            "errors": self.errors,
            "remaining_steps": self.remaining_steps,
            "remaining_tool_budget": self.remaining_tool_budget,
            "timestamp": self.timestamp,
        }


@dataclass
class AgentRun:
    """
    Canonical multi-step execution run record for Phase 6.
    """
    run_id: str = field(default_factory=lambda: f"run_{uuid.uuid4().hex[:12]}")
    project_id: Optional[str] = None
    conversation_id: Optional[str] = None
    user_request: str = ""
    current_state: AgentRunState = AgentRunState.CREATED
    step_number: int = 0
    max_steps: int = 8
    max_tool_calls: int = 8
    max_retries_per_tool: int = 2
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    checkpoints: List[AgentCheckpoint] = field(default_factory=list)
    cancellation_requested: bool = False
    created_at: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())
    completed_at: Optional[str] = None
    final_response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def transition_to(self, new_state: AgentRunState) -> None:
        """Safely transition to a new run state."""
        self.current_state = new_state
        if new_state in (AgentRunState.COMPLETED, AgentRunState.FAILED, AgentRunState.CANCELLED, AgentRunState.TIMEOUT):
            if not self.completed_at:
                self.completed_at = dt.datetime.now(dt.timezone.utc).isoformat()

    def create_checkpoint(self) -> AgentCheckpoint:
        """Create a state snapshot checkpoint without private chain-of-thought."""
        chk = AgentCheckpoint(
            step_number=self.step_number,
            current_state=self.current_state.value if isinstance(self.current_state, AgentRunState) else str(self.current_state),
            completed_tool_calls=list(self.tool_calls),
            observations=list(self.observations),
            errors=list(self.errors),
            remaining_steps=max(0, self.max_steps - self.step_number),
            remaining_tool_budget=max(0, self.max_tool_calls - len(self.tool_calls)),
        )
        self.checkpoints.append(chk)
        return chk

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "project_id": self.project_id,
            "conversation_id": self.conversation_id,
            "user_request": self.user_request,
            "current_state": self.current_state.value if isinstance(self.current_state, AgentRunState) else str(self.current_state),
            "step_number": self.step_number,
            "max_steps": self.max_steps,
            "max_tool_calls": self.max_tool_calls,
            "tool_calls": self.tool_calls,
            "observations": self.observations,
            "errors": self.errors,
            "checkpoints": [c.to_dict() for c in self.checkpoints],
            "cancellation_requested": self.cancellation_requested,
            "created_at": self.created_at,
            "final_response": self.final_response,
            "metadata": self.metadata,
        }

    @classmethod
    def restore_from_checkpoint(cls, checkpoint: AgentCheckpoint, user_request: str = "") -> AgentRun:
        """Construct an AgentRun from a saved AgentCheckpoint."""
        return cls(
            run_id=getattr(checkpoint, "run_id", None) or f"run_{uuid.uuid4().hex[:12]}",
            user_request=user_request,
            current_state=AgentRunState(checkpoint.current_state) if isinstance(checkpoint.current_state, str) else checkpoint.current_state,
            step_number=checkpoint.step_number,
            tool_calls=list(checkpoint.completed_tool_calls),
            observations=list(checkpoint.observations),
            errors=list(checkpoint.errors),
        )


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
    status: str = "pending"  # "pending", "planning", "executing", "verifying", "completed", "failed", "verification_failed", "cancelled", "timeout"
    plan: list[dict[str, Any]] = field(default_factory=list)
    current_step: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    verification: Optional[dict[str, Any]] = None
    final_answer: Optional[str] = None
    attachments: list[str] = field(default_factory=list)
    project_id: Optional[str] = None
    conversation_id: Optional[str] = None
    context: Optional[Any] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    run_record: Optional[AgentRun] = None

    def __post_init__(self) -> None:
        if self.run_record is None:
            self.run_record = AgentRun(
                project_id=self.project_id,
                conversation_id=self.conversation_id,
                user_request=self.user_request,
            )

    def to_dict(self, include_run_record: bool = False) -> dict[str, Any]:
        """Return clean JSON-serializable dictionary matching Section 1 spec."""
        data = {
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
        if self.attachments:
            data["attachments"] = self.attachments
        if self.project_id:
            data["project_id"] = self.project_id
        if self.conversation_id:
            data["conversation_id"] = self.conversation_id
        if self.context:
            data["context"] = self.context.to_dict() if hasattr(self.context, "to_dict") else str(self.context)
        if include_run_record and self.run_record:
            data["run_record"] = self.run_record.to_dict()
        return data

    def add_tool_call(self, tool_name: str, arguments: dict[str, Any], execution_id: Optional[str] = None) -> None:
        """Record a structured tool call."""
        call_entry = {"tool": tool_name, "arguments": arguments}
        if execution_id:
            call_entry["execution_id"] = execution_id
        self.tool_calls.append(call_entry)
        if self.run_record:
            self.run_record.tool_calls.append(call_entry)

    def add_tool_result(
        self,
        tool_name: str,
        result: Any,
        success: bool = True,
        error: Optional[str] = None,
        execution_id: Optional[str] = None,
        state: Optional[str] = None,
    ) -> None:
        """Record an observed tool result."""
        res: dict[str, Any] = {
            "tool": tool_name,
            "success": success,
        }
        if execution_id:
            res["execution_id"] = execution_id
        if state:
            res["state"] = state
        if success:
            res["result"] = str(result)
        else:
            res["error"] = str(error or "Execution error")
        self.tool_results.append(res)
        if self.run_record:
            self.run_record.observations.append(res)
            if not success and error:
                self.run_record.errors.append(f"[{tool_name}] {error}")
