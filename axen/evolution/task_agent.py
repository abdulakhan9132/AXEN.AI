"""
axen/evolution/task_agent.py
===========================
TaskAgent for the AXEN Self-Evolution Engine (HyperAgents architecture).

The TaskAgent executes tasks within a restricted execution environment.
It wraps or leverages the existing AgentLoop / Planner / Verifier / ToolRegistry
to solve programming and reasoning tasks, capturing fine-grained execution
traces (steps, latency, tool calls, verifier reports) for meta-analysis.

Operates strictly under `AgentRole.TASK_AGENT` with read-only and sandbox permissions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time
from typing import Any, Dict, List, Optional

from .permissions import AgentRole, EvolutionPermission, EvolutionSecurityPolicy

logger = logging.getLogger("axen.evolution.task_agent")


@dataclass
class TaskExecutionResult:
    """Detailed record of a task solved by TaskAgent."""

    task_id: str
    success: bool
    output: str
    error: Optional[str] = None
    steps_taken: int = 0
    duration_seconds: float = 0.0
    memory_mb: float = 0.0
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    verifier_passed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "steps_taken": self.steps_taken,
            "duration_seconds": round(self.duration_seconds, 4),
            "memory_mb": round(self.memory_mb, 2),
            "tool_calls_count": len(self.tool_calls),
            "verifier_passed": self.verifier_passed,
            "metadata": self.metadata,
        }


class TaskAgent:
    """
    Worker agent that executes benchmarks and programming tasks.

    Captures telemetry for the MetaAgent to evaluate and optimize.
    """

    def __init__(
        self,
        agent_loop: Optional[Any] = None,
        tool_registry: Optional[Any] = None,
        model: Optional[Any] = None,
        security_policy: Optional[EvolutionSecurityPolicy] = None,
        max_steps: int = 10,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.agent_loop = agent_loop
        self.tool_registry = tool_registry
        self.model = model
        self.security_policy = security_policy or EvolutionSecurityPolicy(
            current_role=AgentRole.TASK_AGENT
        )
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds

    def execute_task(
        self,
        task_id: str,
        prompt: str,
        expected_output: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskExecutionResult:
        """
        Execute a single task and record metrics.

        Args:
            task_id: Identifier for the task/problem.
            prompt: Problem statement or instruction.
            expected_output: Expected answer/outcome for deterministic verification.
            context: Additional runtime parameters (e.g. system prompt overrides).
        """
        self.security_policy.authorize(EvolutionPermission.EXECUTE_SANDBOX)

        start_time = time.perf_counter()
        steps = 0
        tool_calls: List[Dict[str, Any]] = []
        output = ""
        error: Optional[str] = None
        verifier_passed = False

        try:
            if self.agent_loop is not None:
                # Wrap existing AgentLoop
                run_context = context or {}
                if hasattr(self.agent_loop, "run"):
                    result = self.agent_loop.run(prompt, context=run_context)
                    if hasattr(result, "final_answer"):
                        output = str(result.final_answer)
                    elif isinstance(result, dict):
                        output = str(result.get("response", result.get("output", "")))
                    else:
                        output = str(result)
                    steps = getattr(result, "step_count", 1)
                else:
                    raise RuntimeError("Configured agent loop has no run method")
            elif self.model is not None:
                # Direct AIModel generation
                system_prompt = (context or {}).get(
                    "system_prompt",
                    "You are AXEN TaskAgent. Solve the given reasoning or coding problem concisely.",
                )
                if hasattr(self.model, "generate"):
                    response = self.model.generate(
                        prompt=prompt,
                        system=system_prompt,
                        temperature=0.2,
                    )
                    output = getattr(response, "text", str(response))
                else:
                    output = str(self.model(prompt))
                steps = 1
            else:
                raise RuntimeError("TaskAgent requires a real model or agent loop")

            # Verification
            if expected_output is not None:
                verifier_passed = expected_output.strip() in output.strip()
            else:
                verifier_passed = len(output.strip()) > 0

            success = verifier_passed and (error is None)

        except Exception as exc:
            logger.exception("TaskAgent failed executing task %s", task_id)
            error = str(exc)
            output = ""
            success = False
            verifier_passed = False

        duration = time.perf_counter() - start_time

        return TaskExecutionResult(
            task_id=task_id,
            success=success,
            output=output,
            error=error,
            steps_taken=steps,
            duration_seconds=duration,
            memory_mb=0.0,
            tool_calls=tool_calls,
            verifier_passed=verifier_passed,
            metadata=context or {},
        )
