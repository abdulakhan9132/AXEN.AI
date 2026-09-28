"""
axen/agent/planner.py
=====================
Planner for AXEN Phase 3 Agentic Reasoning Loop.

Creates structured execution plans for simple, single-tool, and mixed multi-step tasks.
Does NOT expose private chain-of-thought; stores only structured step definitions.
"""

from __future__ import annotations

import re
from typing import Any, Optional


class Planner:
    """Builds structured step plans for tasks."""

    def create_plan(
        self,
        task_type: str,
        user_request: str,
        selected_tool: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Generate a list of structured plan steps.

        Each step contains:
          - step: int
          - description: human-readable action
          - tool: tool identifier or None
          - arguments: dict of arguments for tool
        """
        req = user_request.strip()

        # 1. Simple conversation / knowledge questions (no tool needed)
        if task_type in ("conversation", "programming", "knowledge_retrieval") or not selected_tool:
            return [{
                "step": 1,
                "description": "Generate direct conversational response using local language model",
                "tool": None,
                "arguments": {},
            }]

        # 2. Calculation task
        if task_type == "calculation" or selected_tool == "calculator":
            return [{
                "step": 1,
                "description": f"Calculate arithmetic expression from: {req}",
                "tool": "calculator",
                "arguments": {"raw_input": req},
            }]

        # 3. Symbolic math task
        if task_type == "math" or selected_tool == "math_solver":
            return [{
                "step": 1,
                "description": f"Solve and verify mathematical problem with MathSolver: {req}",
                "tool": "math_solver",
                "arguments": {"problem": req},
            }]

        # 4. Code execution
        if task_type == "code_execution" or selected_tool == "python":
            return [{
                "step": 1,
                "description": "Execute Python code safely in isolated sandbox",
                "tool": "python",
                "arguments": {"raw_input": req},
            }]

        # 5. File operation
        if task_type == "file_operation" or selected_tool == "file":
            match = re.search(r"\b(?:read|show|view|open|cat)\s+(?:this\s+)?file[:\s]+([^\s,;]+)", req, re.I)
            path = match.group(1).strip().strip("'\"") if match else req
            return [{
                "step": 1,
                "description": f"Read and inspect local file content: {path}",
                "tool": "file",
                "arguments": {"path": path},
            }]

        # 6. Memory operation
        if task_type == "memory" or selected_tool == "memory":
            return [{
                "step": 1,
                "description": "Perform memory retrieval or store operation",
                "tool": "memory",
                "arguments": {"raw_input": req},
            }]

        # 7. Mixed / Multi-step task
        if task_type == "mixed":
            return self._plan_mixed_task(req)

        # Fallback
        return [{
            "step": 1,
            "description": f"Execute task: {req}",
            "tool": selected_tool,
            "arguments": {"raw_input": req},
        }]

    def _plan_mixed_task(self, req: str) -> list[dict[str, Any]]:
        """Construct multi-step pipeline plan for compound requests."""
        steps: list[dict[str, Any]] = []
        step_num = 1

        # Check for file read component
        file_match = re.search(r"\b(?:read|open|load)\s+(?:this\s+)?file[:\s]+([^\s,;]+)", req, re.I)
        if file_match:
            path = file_match.group(1).strip().strip("'\"")
            steps.append({
                "step": step_num,
                "description": f"Read input file: {path}",
                "tool": "file",
                "arguments": {"path": path},
            })
            step_num += 1

        # Check for calculation or code component
        if re.search(r"\b(?:calculate|compute|average|sum|total|percentage)\b", req, re.I):
            steps.append({
                "step": step_num,
                "description": "Execute verified arithmetic / math operation",
                "tool": "calculator",
                "arguments": {"raw_input": req},
            })
            step_num += 1
        elif re.search(r"\b(?:python|run|test|execute)\b", req, re.I):
            steps.append({
                "step": step_num,
                "description": "Execute generated code in Python sandbox",
                "tool": "python",
                "arguments": {"raw_input": req},
            })
            step_num += 1

        # Final explanation step
        steps.append({
            "step": step_num,
            "description": "Synthesize observations and explain result via local LLM",
            "tool": "llm",
            "arguments": {"raw_input": req},
        })

        return steps
