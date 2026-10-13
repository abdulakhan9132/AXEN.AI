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

        # 5. File operation & Document Analysis
        if task_type in ("file_operation", "document_analysis", "file_analysis") or selected_tool in ("file", "file.read", "file_read", "file_search", "file.search", "file.inspect"):
            if selected_tool in ("file_search", "file.search") or re.search(r"\b(?:find|search|where|locate)\b", lower):
                m_search = re.search(r"\b(?:find|search|where|locate)\s+(?:for\s+)?(.+)", req, re.I)
                q = m_search.group(1).strip() if m_search else req
                return [_make_step(
                    step_id=1,
                    action=f"Search files for pattern: {q}",
                    tool="file_search",
                    arguments={"query": q, "request": req},
                    input_summary=f"Search: {q[:40]}",
                )]
            match = re.search(r"\b(?:read|show|view|open|cat)\s+(?:this\s+)?file[:\s]+([^\s,;]+)", req, re.I)
            if not match:
                match = re.search(r"\b(?:read|show|view|open)\s+([a-zA-Z0-9_\-\.\/\\]+\.(?:md|txt|py|json|csv))\b", req, re.I)
            path = match.group(1).strip().strip("'\"") if match else req
            return [_make_step(
                step_id=1,
                action=f"Read and inspect file content: {path}",
                tool="file_read" if selected_tool in ("file_read", "file.read") else "file",
                arguments={"path": path, "request": req},
                input_summary=f"File: {path[:40]}",
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

        # 7. Knowledge / RAG retrieval operation
        if task_type in ("knowledge_retrieval", "rag") or selected_tool in ("knowledge", "knowledge.retrieve"):
            return [_make_step(
                step_id=1,
                action=f"Search and retrieve grounded passages from knowledge base: {req[:80]}",
                tool="knowledge",
                arguments={"query": req},
                input_summary=req[:60],
            )]

        # 8. Software engineering / Coding Agent operation
        if task_type == "coding" or selected_tool in ("coding", "coding.inspect", "coding.search", "coding.patch", "coding.test"):
            return [_make_step(
                step_id=1,
                action=f"Execute software engineering workflow in workspace: {req[:80]}",
                tool="coding",
                arguments={"request": req},
                input_summary=req[:60],
            )]

        # 9. Web research / Web Agent operation
        if task_type in ("web_search", "web_research", "current_information") or selected_tool in ("web", "web.research", "web.search", "web.read"):
            return [_make_step(
                step_id=1,
                action=f"Execute web research and multi-source verification: {req[:80]}",
                tool="web.research",
                arguments={"query": req},
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

        # Check for search or find component
        if re.search(r"\b(?:find|search|locate)\b", lower) and ("csv" in lower or "data" in lower or "file" in lower or "reference" in lower or "authentication" in lower):
            search_query = "csv" if "csv" in lower else ("data" if "data" in lower else "authentication")
            steps.append(_make_step(
                step_id=step_num,
                action=f"Search for file or references: {search_query}",
                tool="file_search",
                arguments={"query": search_query, "raw_input": req},
                input_summary=f"Search: {search_query}",
            ))
            step_num += 1

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

        if path or re.search(r"\b(?:file|read|inspect)\b", req, re.I) or (step_num > 1 and ("csv" in lower or "calculate" in lower or "average" in lower)):
            steps.append(_make_step(
                step_id=step_num,
                action=f"Read input file: {path or 'target file'}",
                tool="file_read" if not path and not re.search(r"\b(?:read|open)\s+this\s+file\b", req, re.I) else "file",
                arguments={"path": path, "raw_input": req, "from_step": 1 if step_num > 1 else None},
                input_summary=f"Path: {path or 'target'}",
            ))
            step_num += 1

        # Check for calculation or arithmetic component
        if re.search(r"\b(?:calculate|compute|average|sum|total|totals|numbers|percentage)\b", lower):
            target_tool = "python" if ("average" in lower and "csv" in lower) else "calculator"
            steps.append(_make_step(
                step_id=step_num,
                action=f"Execute computation: {target_tool}",
                tool=target_tool,
                arguments={"raw_input": req, "from_step": step_num - 1},
                input_summary=req[:60],
            ))
            step_num += 1
        elif re.search(r"\b(?:math|derivative|integral|solve)\b", lower):
            steps.append(_make_step(
                step_id=step_num,
                action="Solve mathematical calculation with MathSolver",
                tool="math_solver",
                arguments={"problem": req, "from_step": step_num - 1},
                input_summary=req[:60],
            ))
            step_num += 1
        elif re.search(r"\b(?:convert|binary|hex|bitwise)\b", lower):
            steps.append(_make_step(
                step_id=step_num,
                action="Execute computer systems conversion deterministically",
                tool="computer",
                arguments={"request": req, "from_step": step_num - 1},
                input_summary=req[:60],
            ))
            step_num += 1
        elif re.search(r"\b(?:python|run|test|execute)\b", lower):
            steps.append(_make_step(
                step_id=step_num,
                action="Execute code in Python sandbox",
                tool="python",
                arguments={"raw_input": req, "from_step": step_num - 1},
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

    def replan(
        self,
        current_plan: list[dict[str, Any]],
        failed_step_id: int,
        error_message: str,
        user_request: str,
        intent_info: Optional[IntentInfo] = None,
    ) -> list[dict[str, Any]]:
        """
        Produce a revised plan starting from the failed step.
        Preserves completed steps and replaces the failed step with an alternative strategy.
        """
        revised: list[dict[str, Any]] = []
        for step in current_plan:
            sid = step.get("step_id", step.get("step", 0))
            if sid < failed_step_id:
                revised.append(step)
            elif sid == failed_step_id:
                # Construct alternative for failed step
                tool = step.get("tool")
                args = step.get("arguments", {})
                req = user_request.strip()

                if tool in ("calculator", "calculator.evaluate"):
                    revised.append(_make_step(
                        step_id=sid,
                        action=f"Execute fallback computation in Python sandbox: {req}",
                        tool="python",
                        arguments={"code": f"print({req})", "raw_input": req},
                        input_summary=f"Re-plan fallback: python for {req[:40]}",
                    ))
                elif tool == "math_solver":
                    revised.append(_make_step(
                        step_id=sid,
                        action=f"Execute fallback symbolic computation in Python: {req}",
                        tool="python",
                        arguments={"code": f"import sympy as sp\nprint(sp.sympify('{req}'))", "raw_input": req},
                        input_summary=f"Re-plan fallback: sympy for {req[:40]}",
                    ))
                elif tool in ("file", "file_read"):
                    revised.append(_make_step(
                        step_id=sid,
                        action=f"Search for file matching request: {req}",
                        tool="file_search",
                        arguments={"query": args.get("path") or req, "raw_input": req},
                        input_summary="Re-plan fallback: file search",
                    ))
                else:
                    revised.append(_make_step(
                        step_id=sid,
                        action=f"Retry with relaxed parameters or alternate approach: {req}",
                        tool=tool,
                        arguments=args,
                        input_summary=f"Re-plan retry: {error_message[:40]}",
                    ))
            else:
                # Keep remaining steps
                revised.append(step)

        return revised

