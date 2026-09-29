"""
axen/agent/planner.py
=====================
Planner for AXEN Agentic Reasoning Loop.

Creates structured execution plans for simple, single-tool, and multi-step tasks.
Maintains structured execution state:
- step_id
- action
- tool
- status (pending, executing, completed, failed)
- input_summary
- result_summary
- verification_status
"""

from __future__ import annotations

import re
from typing import Any, Optional

from .intent import IntentInfo


def _make_step(
    step_id: int,
    action: str,
    tool: Optional[str],
    arguments: dict[str, Any],
    input_summary: str = "",
) -> dict[str, Any]:
    """Helper to construct a structured plan step with backward-compatible aliases."""
    return {
        "step_id": step_id,
        "step": step_id,  # backward compatibility alias
        "action": action,
        "description": action,  # backward compatibility alias
        "tool": tool,
        "status": "pending",
        "input_summary": input_summary or action,
        "result_summary": None,
        "verification_status": None,
        "arguments": arguments,
    }


class Planner:
    """Builds structured step plans for tasks."""

    def create_plan(
        self,
        task_type: str,
        user_request: str,
        selected_tool: Optional[str] = None,
        intent_info: Optional[IntentInfo] = None,
    ) -> list[dict[str, Any]]:
        """
        Generate a list of structured plan steps.
        """
        req = user_request.strip()
        lower = req.lower()

        # 0. Check for mixed tasks / multi-step tasks
        if task_type in ("mixed", "multi_step") or (intent_info and intent_info.intent in ("mixed", "multi_step")):
            return self._plan_mixed_task(req, intent_info)

        # 1. Simple conversation / knowledge / reasoning questions (no tool needed)
        if task_type in ("conversation", "programming", "knowledge_retrieval", "knowledge", "reasoning", "writing", "translation", "planning", "ambiguous", "follow_up") and not selected_tool:
            return [_make_step(
                step_id=1,
                action="Generate direct conversational or reasoned response using local language model",
                tool=None,
                arguments={},
                input_summary=req[:60],
            )]

        # 2. Calculation task
        if task_type == "calculation" or selected_tool == "calculator":
            return [_make_step(
                step_id=1,
                action=f"Calculate arithmetic expression from: {req}",
                tool="calculator",
                arguments={"raw_input": req},
                input_summary=req,
            )]

        # 3. Symbolic math task
        if task_type == "math" or selected_tool == "math_solver":
            # Multi-operation within math (e.g. "Calculate the derivative and simplify it")
            if re.search(r"\b(?:derivative|differentiate)\b", lower) and re.search(r"\b(?:simplify|expand)\b", lower):
                return self._plan_mixed_task(req, intent_info)

            return [_make_step(
                step_id=1,
                action=f"Solve and verify mathematical problem with MathSolver: {req}",
                tool="math_solver",
                arguments={"problem": req},
                input_summary=req,
            )]

        # 4. Code execution
        if task_type == "code_execution" or selected_tool == "python":
            return [_make_step(
                step_id=1,
                action="Execute Python code safely in isolated sandbox",
                tool="python",
                arguments={"raw_input": req},
                input_summary=req[:60],
            )]

        # 4b. Computer Systems domain
        if task_type == "computer_systems" or selected_tool == "computer":
            return [_make_step(
                step_id=1,
                action=f"Process computer systems query deterministically: {req[:80]}",
                tool="computer",
                arguments={"request": req},
                input_summary=req[:60],
            )]

        # 5. File operation
        if task_type == "file_operation" or selected_tool == "file":
            match = re.search(r"\b(?:read|show|view|open|cat)\s+(?:this\s+)?file[:\s]+([^\s,;]+)", req, re.I)
            path = match.group(1).strip().strip("'\"") if match else req
            return [_make_step(
                step_id=1,
                action=f"Read and inspect local file content: {path}",
                tool="file",
                arguments={"path": path},
                input_summary=f"Path: {path}",
            )]

        # 6. Memory operation
        if task_type == "memory" or selected_tool == "memory":
            return [_make_step(
                step_id=1,
                action="Perform memory retrieval or store operation",
                tool="memory",
                arguments={"raw_input": req},
                input_summary=req[:60],
            )]

        # Fallback
        return [_make_step(
            step_id=1,
            action=f"Execute task: {req}",
            tool=selected_tool,
            arguments={"raw_input": req},
            input_summary=req[:60],
        )]

    def _plan_mixed_task(self, req: str, intent_info: Optional[IntentInfo] = None) -> list[dict[str, Any]]:
        """Construct multi-step pipeline plan for compound requests."""
        steps: list[dict[str, Any]] = []
        step_num = 1
        lower = req.lower()

        # Check for derivative + simplify multi-step math
        if re.search(r"\b(?:derivative|differentiate)\b", lower) and re.search(r"\b(?:simplify|expand)\b", lower):
            steps.append(_make_step(
                step_id=step_num,
                action="Calculate symbolic derivative with MathSolver",
                tool="math_solver",
                arguments={"problem": req, "operation": "differentiate"},
                input_summary="Derivative request",
            ))
            step_num += 1
            steps.append(_make_step(
                step_id=step_num,
                action="Simplify derivative expression algebraically with MathSolver",
                tool="math_solver",
                arguments={"problem": req, "operation": "simplify", "from_step": 1},
                input_summary="Simplification of derivative",
            ))
            step_num += 1
            if re.search(r"\b(?:explain|describe|why)\b", lower):
                steps.append(_make_step(
                    step_id=step_num,
                    action="Synthesize explanation of mathematical steps using local LLM",
                    tool="llm",
                    arguments={"raw_input": req},
                    input_summary="Mathematical explanation",
                ))
            return steps

        # Check for file read component
        file_match = re.search(r"\b(?:read|open|load|check)\s+(?:this\s+)?file[:\s]+([^\s,;]+)", req, re.I)
        if not file_match:
            file_match = re.search(r"\b(?:read|check|inspect)\s+(?:my\s+|the\s+)?(?:python\s+)?file\b[:\s]*([^\s,;]+)?", req, re.I)

        path = ""
        if file_match and file_match.group(1):
            path = file_match.group(1).strip().strip("'\"")
        elif "file" in lower:
            fn_m = re.search(r"([a-zA-Z0-9_\-\.\/\\]+\.(?:txt|csv|py|json|md))", req)
            if fn_m:
                path = fn_m.group(1).strip()

        if path or re.search(r"\b(?:file|read|inspect)\b", req, re.I):
            steps.append(_make_step(
                step_id=step_num,
                action=f"Read input file: {path or 'target file'}",
                tool="file",
                arguments={"path": path, "raw_input": req},
                input_summary=f"Path: {path or 'target'}",
            ))
            step_num += 1

        # Check for calculation or arithmetic component
        if re.search(r"\b(?:calculate|compute|average|sum|total|totals|numbers|percentage)\b", lower):
            steps.append(_make_step(
                step_id=step_num,
                action="Execute verified arithmetic / math operation",
                tool="calculator",
                arguments={"raw_input": req, "from_step": 1 if path else None},
                input_summary=req[:60],
            ))
            step_num += 1
        elif re.search(r"\b(?:math|derivative|integral|solve)\b", lower):
            steps.append(_make_step(
                step_id=step_num,
                action="Solve mathematical calculation with MathSolver",
                tool="math_solver",
                arguments={"problem": req, "from_step": 1 if path else None},
                input_summary=req[:60],
            ))
            step_num += 1
        elif re.search(r"\b(?:convert|binary|hex|bitwise)\b", lower):
            steps.append(_make_step(
                step_id=step_num,
                action="Execute computer systems conversion deterministically",
                tool="computer",
                arguments={"request": req, "from_step": 1 if path else None},
                input_summary=req[:60],
            ))
            step_num += 1
        elif re.search(r"\b(?:python|run|test|execute)\b", lower):
            steps.append(_make_step(
                step_id=step_num,
                action="Execute code in Python sandbox",
                tool="python",
                arguments={"raw_input": req, "from_step": 1 if path else None},
                input_summary=req[:60],
            ))
            step_num += 1

        # Final explanation / synthesis step if explanation is requested or mixed
        steps.append(_make_step(
            step_id=step_num,
            action="Synthesize observations and explain result via local LLM",
            tool="llm",
            arguments={"raw_input": req},
            input_summary="Explanation synthesis",
        ))

        return steps
