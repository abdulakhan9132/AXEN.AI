from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskState(str, Enum):
    CREATED = "CREATED"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    WAITING_FOR_USER = "WAITING_FOR_USER"
    VERIFYING = "VERIFYING"
    RECOVERING = "RECOVERING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class TaskRecord:
    task_id: str
    goal: str
    state: TaskState = TaskState.CREATED
    plan: Optional[Dict[str, Any]] = None
    summary: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    context: Dict[str, Any] = field(default_factory=dict)

    def update(self, *, state: Optional[TaskState] = None, plan: Optional[Dict[str, Any]] = None, summary: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        if state is not None:
            self.state = state
        if plan is not None:
            self.plan = plan
        if summary is not None:
            self.summary = summary
        if context is not None:
            self.context = context
        self.updated_at = datetime.now(timezone.utc).isoformat()


class TaskStore:
    def __init__(self):
        self._tasks: Dict[str, TaskRecord] = {}

    def create_task(self, goal: str) -> TaskRecord:
        task_id = f"task-{len(self._tasks) + 1}"
        task = TaskRecord(task_id=task_id, goal=goal)
        self._tasks[task_id] = task
        return task

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        return self._tasks.get(task_id)

    def list_tasks(self) -> List[TaskRecord]:
        return list(self._tasks.values())

    def update_task(self, task_id: str, **kwargs) -> Optional[TaskRecord]:
        task = self._tasks.get(task_id)
        if task is None:
            return None
        task.update(**kwargs)
        return task
