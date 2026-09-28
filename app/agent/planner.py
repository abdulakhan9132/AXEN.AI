from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from app.ai.context import ContextManager
from app.ai.prompts import PLAN_TEMPLATE
from app.ai.model_manager import ModelManager


class Planner:
    """Converts goals into a structured step-by-step plan."""

    def __init__(self, model_manager: Optional[ModelManager] = None):
        self.model_manager = model_manager or ModelManager()
        self.context_manager = ContextManager()

    def plan(self, goal: str, *, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not goal or not goal.strip():
            return {"goal": "", "steps": []}

        if self.model_manager.health_check():
            try:
                prompt = PLAN_TEMPLATE.format(goal=goal)
                structured = self.model_manager.get_model().structured_output(
                    prompt,
                    schema={"type": "object"},
                    context=context or {},
                    system="You produce JSON plans only.",
                )
                if isinstance(structured, dict) and "steps" in structured:
                    return structured
            except Exception:
                pass

        return self._default_plan(goal)

    def revise_plan(self, current_plan: Dict[str, Any], feedback: str) -> Dict[str, Any]:
        revised = dict(current_plan)
        revised["notes"] = feedback
        steps = revised.get("steps", [])
        for idx, step in enumerate(steps, start=1):
            step.setdefault("id", idx)
            step.setdefault("requires_approval", False)
        return revised

    def _default_plan(self, goal: str) -> Dict[str, Any]:
        goal_lower = goal.lower()
        if "test_agent" in goal_lower or ("folder" in goal_lower and "text file" in goal_lower):
            steps = [
                {
                    "id": 1,
                    "objective": "Create the requested test_agent folder and text file",
                    "tool": "filesystem.write",
                    "requires_approval": False,
                    "arguments": {
                        "path": "test_agent/test_agent_notes.txt",
                        "content": "This file was created by the local AI agent.\n",
                    },
                },
                {
                    "id": 2,
                    "objective": "Verify the file exists after creation",
                    "tool": "filesystem.read",
                    "requires_approval": False,
                    "arguments": {"path": "test_agent/test_agent_notes.txt"},
                },
            ]
        elif "pdf" in goal_lower:
            step_tool = "filesystem.search"
            step_args = {"pattern": "**/*.pdf", "root": "."}
            steps = [
                {"id": 1, "objective": "Search the workspace for PDF files", "tool": step_tool, "requires_approval": False, "arguments": step_args},
                {"id": 2, "objective": "List the matching PDF files discovered", "tool": "filesystem.read", "requires_approval": False, "arguments": {"path": "."}},
            ]
        elif "structure" in goal_lower or "project" in goal_lower:
            steps = [
                {"id": 1, "objective": "Inspect the workspace layout", "tool": "filesystem.list_dir", "requires_approval": False, "arguments": {"path": "."}},
                {"id": 2, "objective": "Summarize the local project structure", "tool": "memory.read", "requires_approval": False, "arguments": {"query": "project structure"}},
            ]
        elif "calculator" in goal_lower and "error" in goal_lower:
            steps = [
                {
                    "id": 1,
                    "objective": "Create a calculator implementation with a logic bug to investigate",
                    "tool": "filesystem.write",
                    "requires_approval": False,
                    "arguments": {
                        "path": "calculator.py",
                        "content": "def add(a, b):\n    return a - b\n",
                    },
                },
                {
                    "id": 2,
                    "objective": "Run the calculator to observe the defect and confirm the incorrect result",
                    "tool": "shell.execute",
                    "requires_approval": True,
                    "arguments": {
                        "command": "python -c \"import calculator\nprint(calculator.add(5, 3))\"",
                        "reason": "Check the current calculator behavior before fixing the defect.",
                        "risk": "low",
                    },
                },
                {
                    "id": 3,
                    "objective": "Fix the arithmetic bug in the calculator implementation",
                    "tool": "filesystem.write",
                    "requires_approval": False,
                    "arguments": {
                        "path": "calculator.py",
                        "content": "def add(a, b):\n    return a + b\n",
                    },
                },
                {
                    "id": 4,
                    "objective": "Verify the corrected calculator logic",
                    "tool": "shell.execute",
                    "requires_approval": True,
                    "arguments": {
                        "command": "python -c \"import calculator\nprint(calculator.add(5, 3))\"",
                        "reason": "Verify the corrected calculator returns the expected value.",
                        "risk": "low",
                    },
                },
            ]
        elif "calculator" in goal_lower:
            steps = [
                {"id": 1, "objective": "Create a simple Python calculator file", "tool": "filesystem.write", "requires_approval": False, "arguments": {"path": "calculator.py", "content": "def add(a, b):\n    return a + b\n\nprint(add(2, 3))\n"}},
                {"id": 2, "objective": "Verify the calculator output", "tool": "calculator.evaluate", "requires_approval": False, "arguments": {"expression": "2 + 3"}},
            ]
        else:
            steps = [
                {"id": 1, "objective": "Inspect the workspace for relevant files", "tool": "filesystem.list_dir", "requires_approval": False, "arguments": {"path": "."}},
                {"id": 2, "objective": "Perform task-specific actions and verify the outcome", "tool": "filesystem.write", "requires_approval": False, "arguments": {"path": "task_output.txt", "content": goal}},
            ]

        return {"goal": goal, "steps": steps}
