from __future__ import annotations

import datetime as dt
import json
from typing import Any, Dict, List, Optional

from app.ai.context import ContextManager
from app.ai.model_manager import ModelManager
from app.agent.executor import Executor
from app.agent.memory import MemoryStore
from app.agent.observer import Observer
from app.agent.planner import Planner
from app.agent.recovery import RecoveryManager
from app.agent.state import TaskState, TaskStore
from app.agent.tools import ToolRegistry
from app.agent.verifier import Verifier
from app.tools.registry import build_registry


class AgentOrchestrator:
    """Coordinates planning, execution, observation, verification, and recovery."""

    def __init__(
        self,
        *,
        model_manager: Optional[ModelManager] = None,
        tool_registry: Optional[ToolRegistry] = None,
        memory_store: Optional[MemoryStore] = None,
        task_store: Optional[TaskStore] = None,
        max_steps: int = 6,
        max_retries: int = 3,
        timeout_seconds: int = 30,
        circuit_breaker: int = 3,
    ):
        self.model_manager = model_manager or ModelManager()
        self.tool_registry = tool_registry or build_registry()
        self.memory_store = memory_store or MemoryStore()
        self.task_store = task_store or TaskStore()
        self.planner = Planner(self.model_manager)
        self.executor = Executor(self.tool_registry)
        self.observer = Observer()
        self.verifier = Verifier()
        self.context_manager = ContextManager()
        self.recovery_manager = RecoveryManager(max_retries=max_retries)
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.circuit_breaker = circuit_breaker

    def run(self, goal: str) -> Dict[str, Any]:
        task = self.task_store.create_task(goal)
        task.update(state=TaskState.CREATED)
        self.memory_store.add_memory(content=goal, source="user_goal", category="working_memory", confidence=0.9)

        events: List[str] = []
        if not self.model_manager.health_check():
            events.append("Local AI model is unavailable. Falling back to rule-based planning.")
        else:
            events.append("Planning task...")

        task.update(state=TaskState.PLANNING)
        plan = self.planner.plan(goal)
        task.update(plan=plan, state=TaskState.PLANNING)

        events.append("Planning task...")
        relevant_memory = self.memory_store.search(goal[:20], limit=5)
        memory_text = [item["content"] for item in relevant_memory]
        last_result = ""

        for step_index, step in enumerate(plan.get("steps", [])[: self.max_steps], start=1):
            events.append(f"Executing step {step_index}: {step.get('objective', 'Task step')}")
            task.update(state=TaskState.EXECUTING)
            attempts = 0
            while attempts < self.max_retries:
                attempts += 1
                result = self.executor.execute_step(step)
                observation = self.observer.observe(result, context=f"Step {step_index}: {step.get('objective')}.")
                verification = self.verifier.verify(step, result)
                if verification.passed_:
                    self.memory_store.add_memory(
                        content=json.dumps({"step": step, "result": result, "verification": verification.details}, default=str),
                        source="execution_result",
                        category="episodic_memory",
                        confidence=0.8,
                    )
                    task.update(state=TaskState.VERIFYING)
                    events.append("Verifying results...")
                    last_result = result
                    break

                recovery = self.recovery_manager.recover(error=verification.details[0], step=step, attempts=attempts)
                task.update(state=TaskState.RECOVERING)
                events.append(f"Recovery: {recovery['status']}")
                if recovery["status"] == "waiting_for_user":
                    task.update(state=TaskState.WAITING_FOR_USER, summary="Waiting for user approval.")
                    return {
                        "task_id": task.task_id,
                        "status": "waiting_for_user",
                        "summary": "Waiting for user approval before a high-risk action.",
                        "plan": plan,
                        "events": events,
                    }
                if recovery["status"] == "failed":
                    task.update(state=TaskState.FAILED, summary="Task failed after retries.")
                    return {
                        "task_id": task.task_id,
                        "status": "failed",
                        "summary": "Task failed after the retry limit was reached.",
                        "plan": plan,
                        "events": events,
                    }

            if not verification.passed_:
                task.update(state=TaskState.FAILED, summary="Step verification failed.")
                return {
                    "task_id": task.task_id,
                    "status": "failed",
                    "summary": "A required verification failed and no safe recovery path was available.",
                    "plan": plan,
                    "events": events,
                }

            context = self.context_manager.build_context(
                task=goal,
                plan=plan,
                memory=memory_text,
                tool_results=[json.dumps(last_result, default=str)],
                constraints=["Do not expose hidden reasoning.", "Validate all critical actions."],
            )
            self.memory_store.add_memory(
                content=f"Context for goal: {goal}; step summary: {step.get('objective')}",
                source="context_summary",
                category="semantic_memory",
                confidence=0.7,
                metadata={"task_id": task.task_id},
            )

        final_summary = (
            f"Task completed successfully: {goal}. "
            f"Verified {len(plan.get('steps', []))} plan steps and produced the expected results."
        )
        if not self.model_manager.health_check():
            final_summary += " Local AI model is unavailable; fallback planning was used."
        task.update(state=TaskState.COMPLETED, summary=final_summary)
        events.append("Task completed.")
        return {
            "task_id": task.task_id,
            "status": "completed",
            "summary": final_summary,
            "plan": plan,
            "events": events,
        }
