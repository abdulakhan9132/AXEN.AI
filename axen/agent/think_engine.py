"""
axen/agent/think_engine.py
==========================
Think Engine 2.0 — Production-Ready Reasoning Orchestrator.

Implements the complete Think Engine 2.0 lifecycle:
1. Intelligent Task Classification & Mode Selection (FAST, BALANCED, DEEP)
2. Resource Budget Enforcement (tokens, runtime, iterations, tool calls, retries)
3. Advanced Reasoning Orchestration (objective tracking, subtask decomposition, dependency ordering)
4. Dynamic Re-planning on verification failure or tool errors
5. Claim Verification, Self-Checking, and Error Recovery
6. Observability, Telemetry, and Structured Metrics
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


from .intent import IntentInfo
from .think_config import (
    ThinkBudget,
    ThinkEngineConfig,
    ThinkMode,
    ThinkModeConfig,
    ThinkModeSelector,
)
from .think_metrics import (
    PhaseMetric,
    ThinkLogger,
    ThinkRequestMetrics,
    ToolCallMetric,
)
from .verifier import FAILED, NOT_APPLICABLE, UNCERTAIN, VERIFIED, VerificationStatus

logger = logging.getLogger("axen.agent.think_engine")


# ---------------------------------------------------------------------------
# Subtask & Planning Data Structures
# ---------------------------------------------------------------------------


@dataclass
class Subtask:
    """A granular subtask decomposed from a complex objective."""
    id: int
    description: str
    tool: Optional[str] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[int] = field(default_factory=list)
    status: str = "pending"  # pending, executing, completed, failed, skipped
    result: Optional[Any] = None
    observation: Optional[str] = None
    error: Optional[str] = None

    def is_ready(self, completed_ids: Set[int]) -> bool:
        """Check if all prerequisite dependencies have been satisfied."""
        return all(dep_id in completed_ids for dep_id in self.depends_on)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "tool": self.tool,
            "arguments": self.arguments,
            "depends_on": self.depends_on,
            "status": self.status,
            "observation": self.observation,
            "error": self.error,
        }


@dataclass
class DecomposedObjective:
    """Actionable breakdown of a complex user request with tracked state."""
    objective: str
    subtasks: List[Subtask] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    unresolved_questions: List[str] = field(default_factory=list)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)

    def get_ready_subtasks(self) -> List[Subtask]:
        completed = {st.id for st in self.subtasks if st.status == "completed"}
        return [st for st in self.subtasks if st.status == "pending" and st.is_ready(completed)]

    def mark_completed(self, subtask_id: int, result: Any, observation: str = "") -> None:
        for st in self.subtasks:
            if st.id == subtask_id:
                st.status = "completed"
                st.result = result
                st.observation = observation
                self.intermediate_results[f"subtask_{subtask_id}"] = result
                break

    def mark_failed(self, subtask_id: int, error: str) -> None:
        for st in self.subtasks:
            if st.id == subtask_id:
                st.status = "failed"
                st.error = error
                break

    def all_completed(self) -> bool:
        return all(st.status in ("completed", "skipped") for st in self.subtasks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective": self.objective,
            "subtasks": [st.to_dict() for st in self.subtasks],
            "constraints": self.constraints,
            "unresolved_questions": self.unresolved_questions,
            "intermediate_results": {k: str(v)[:200] for k, v in self.intermediate_results.items()},
        }


# ---------------------------------------------------------------------------
# Subtask Decomposer
# ---------------------------------------------------------------------------


class SubtaskDecomposer:
    """Decomposes complex requests into actionable subtasks with dependencies."""

    def decompose(
        self,
        user_request: str,
        intent_info: Optional[IntentInfo] = None,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> DecomposedObjective:
        req = user_request.strip()
        lower = req.lower()
        subtasks: List[Subtask] = []
        constraints: List[str] = []
        unresolved: List[str] = []

        # Detect constraints from prompt text
        if re.search(r"\b(?:without|don't|do not|never|exclude|skip)\b", lower):
            neg_match = re.findall(r"\b(?:without|don't|do not|never|exclude)\s+([^,\.;]+)", lower)
            for nm in neg_match:
                constraints.append(f"Avoid: {nm.strip()}")

        if re.search(r"\b(?:only|strictly|must|format as|in json|in markdown)\b", lower):
            pos_match = re.findall(r"\b(?:strictly|format as|output as)\s+([^,\.;]+)", lower)
            for pm in pos_match:
                constraints.append(f"Require: {pm.strip()}")

        # Multi-stage file + computation / search pattern
        has_file = bool(re.search(r"\b(?:file|csv|json|read|find)\b", lower))
        has_math = bool(re.search(r"\b(?:calculate|average|sum|total|derivative|integral)\b", lower))
        has_code = bool(re.search(r"\b(?:code|python|script|function|program|bug|fix)\b", lower))

        step_id = 1

        if has_file and has_math:
            # Step 1: Locate or read file
            subtasks.append(Subtask(
                id=step_id,
                description=f"Inspect or read source data: {req[:50]}",
                tool="file",
                arguments={"raw_input": req},
                depends_on=[],
            ))
            step_id += 1

            # Step 2: Compute
            subtasks.append(Subtask(
                id=step_id,
                description="Perform mathematical calculation or data analysis",
                tool="calculator" if "derivative" not in lower else "math_solver",
                arguments={"raw_input": req, "from_step": step_id - 1},
                depends_on=[step_id - 1],
            ))
            step_id += 1

            # Step 3: Synthesize
            subtasks.append(Subtask(
                id=step_id,
                description="Synthesize findings and present verified answer",
                tool=None,
                arguments={},
                depends_on=[step_id - 1],
            ))

        elif has_code and ("test" in lower or "verify" in lower or "run" in lower):
            # Coding with verification
            subtasks.append(Subtask(
                id=step_id,
                description="Generate or inspect source implementation",
                tool=None,
                arguments={},
                depends_on=[],
            ))
            step_id += 1

            subtasks.append(Subtask(
                id=step_id,
                description="Execute code in verification sandbox",
                tool="python",
                arguments={"raw_input": req},
                depends_on=[step_id - 1],
            ))
            step_id += 1

            subtasks.append(Subtask(
                id=step_id,
                description="Analyze execution output and format response",
                tool=None,
                arguments={},
                depends_on=[step_id - 1],
            ))

        elif " and " in lower or " then " in lower or ";" in req:
            # Conjunction-based task decomposition
            clauses = re.split(r"\s+(?:and then|then|and)\s+|;\s*", req)
            prev_id = 0
            for idx, clause in enumerate(clauses):
                clause_text = clause.strip()
                if not clause_text:
                    continue
                c_tool = None
                if re.search(r"\b(?:calculate|compute|\d+\s*[\+\-\*\/]\s*\d+)\b", clause_text.lower()):
                    c_tool = "calculator"
                elif re.search(r"\b(?:derivative|integral|solve|factor)\b", clause_text.lower()):
                    c_tool = "math_solver"
                elif re.search(r"\b(?:file|read|inspect)\b", clause_text.lower()):
                    c_tool = "file"
                elif re.search(r"\b(?:python|code|run)\b", clause_text.lower()):
                    c_tool = "python"

                deps = [prev_id] if prev_id > 0 else []
                subtasks.append(Subtask(
                    id=step_id,
                    description=clause_text,
                    tool=c_tool,
                    arguments={"raw_input": clause_text},
                    depends_on=deps,
                ))
                prev_id = step_id
                step_id += 1

        # Fallback single subtask if nothing was decomposed
        if not subtasks:
            primary_tool = intent_info.primary_tool if intent_info else None
            subtasks.append(Subtask(
                id=1,
                description=req,
                tool=primary_tool,
                arguments={"raw_input": req},
                depends_on=[],
            ))

        return DecomposedObjective(
            objective=req,
            subtasks=subtasks,
            constraints=constraints,
            unresolved_questions=unresolved,
        )


# ---------------------------------------------------------------------------
# Claim Verification Report
# ---------------------------------------------------------------------------


@dataclass
class ClaimVerificationReport:
    """Results of self-checking factual claims, syntax, and tool consistency."""
    total_claims: int = 0
    verified_claims: int = 0
    failed_claims: int = 0
    uncertain_claims: int = 0
    details: List[Dict[str, Any]] = field(default_factory=list)
    has_contradiction: bool = False
    has_syntax_error: bool = False
    passed: bool = True

    @property
    def consistency_score(self) -> float:
        if self.total_claims == 0:
            return 1.0
        return round(self.verified_claims / self.total_claims, 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_claims": self.total_claims,
            "verified_claims": self.verified_claims,
            "failed_claims": self.failed_claims,
            "uncertain_claims": self.uncertain_claims,
            "consistency_score": self.consistency_score,
            "has_contradiction": self.has_contradiction,
            "has_syntax_error": self.has_syntax_error,
            "passed": self.passed,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Re-planning Engine
# ---------------------------------------------------------------------------


class ReplanEngine:
    """Generates alternative execution strategies when steps fail or verification fails."""

    def generate_alternative_step(
        self,
        failed_step: Dict[str, Any],
        error_detail: str,
        attempt: int,
    ) -> Optional[Dict[str, Any]]:
        tool = failed_step.get("tool")
        args = failed_step.get("arguments", {})
        raw_input = args.get("raw_input") or args.get("problem") or ""

        # Alternative 1: Calculator failed -> try Python sandbox
        if tool in ("calculator", "calculator.evaluate"):
            return {
                "step_id": failed_step.get("step_id", 1),
                "step": failed_step.get("step", 1),
                "action": f"Re-attempt calculation via Python runtime sandbox: {raw_input}",
                "tool": "python",
                "arguments": {
                    "code": f"result = {raw_input}\nprint(result)",
                    "raw_input": raw_input,
                },
                "status": "pending",
                "input_summary": "Re-plan fallback: calculator -> python",
                "result_summary": None,
                "verification_status": None,
            }

        # Alternative 2: MathSolver failed -> try Python with sympy or math
        if tool == "math_solver":
            return {
                "step_id": failed_step.get("step_id", 1),
                "step": failed_step.get("step", 1),
                "action": f"Re-attempt math problem via Python sympy sandbox: {raw_input}",
                "tool": "python",
                "arguments": {
                    "code": f"import sympy as sp\nx = sp.Symbol('x')\nprint(sp.simplify('{raw_input}'))",
                    "raw_input": raw_input,
                },
                "status": "pending",
                "input_summary": "Re-plan fallback: math_solver -> python sympy",
                "result_summary": None,
                "verification_status": None,
            }

        # Alternative 3: File read failed -> try file search or directory list
        if tool in ("file", "file_read"):
            path = args.get("path", "")
            return {
                "step_id": failed_step.get("step_id", 1),
                "step": failed_step.get("step", 1),
                "action": f"Re-attempt by searching for file matching '{path or raw_input}'",
                "tool": "file_search",
                "arguments": {
                    "query": Path(path).name if path else raw_input,
                    "raw_input": raw_input,
                },
                "status": "pending",
                "input_summary": "Re-plan fallback: file read -> search",
                "result_summary": None,
                "verification_status": None,
            }

        # Alternative 4: Python execution failed with syntax error -> attempt cleanup
        if tool == "python" and "SyntaxError" in error_detail:
            code = args.get("code", "")
            cleaned_code = re.sub(r"^[>\$]\s*", "", code, flags=re.MULTILINE)
            return {
                "step_id": failed_step.get("step_id", 1),
                "step": failed_step.get("step", 1),
                "action": "Re-attempt Python execution with sanitized syntax",
                "tool": "python",
                "arguments": {"code": cleaned_code, "raw_input": raw_input},
                "status": "pending",
                "input_summary": "Re-plan fallback: sanitized python syntax",
                "result_summary": None,
                "verification_status": None,
            }

        return None


# ---------------------------------------------------------------------------
# Think Engine 2.0 Orchestrator
# ---------------------------------------------------------------------------


class ThinkEngine:
    """
    Think Engine 2.0 Central Orchestrator.

    Wraps the reasoning pipeline with:
    - Dynamic mode selection (FAST, BALANCED, DEEP)
    - Hard budget constraints (tokens, time, iterations, tool calls)
    - Subtask decomposition & dependency tracking
    - Dynamic re-planning on verification failure
    - Claim self-checking & consistency verification
    - Comprehensive observability & telemetry
    """

    def __init__(self, config: Optional[ThinkEngineConfig] = None) -> None:
        self.config = config or ThinkEngineConfig.from_yaml()
        self.selector = ThinkModeSelector(self.config)
        self.logger = ThinkLogger(self.config.redact_fields)
        self.decomposer = SubtaskDecomposer()
        self.replan_engine = ReplanEngine()

    # -----------------------------------------------------------------------
    # Lifecycle & Mode Selection
    # -----------------------------------------------------------------------

    def select_mode(
        self,
        user_request: str,
        intent_info: Optional[IntentInfo] = None,
        override: Optional[str | ThinkMode] = None,
    ) -> Tuple[ThinkMode, ThinkModeConfig, str]:
        """
        Dynamically determine the thinking mode for a given request.
        Returns: (mode, mode_config, rationale)
        """
        mode, rationale = self.selector.select(
            user_request=user_request,
            intent_info=intent_info,
            override=override,
        )
        mode_config = self.config.get_mode_config(mode)
        return mode, mode_config, rationale

    def start_request(
        self,
        user_request: str,
        intent_info: Optional[IntentInfo] = None,
        override: Optional[str | ThinkMode] = None,
        request_id: Optional[str] = None,
    ) -> Tuple[ThinkRequestMetrics, ThinkModeConfig]:
        """Initialize metrics and configuration for a new request."""
        mode, mode_config, rationale = self.select_mode(user_request, intent_info, override)

        metrics = ThinkRequestMetrics(
            think_mode=mode,
            intent=intent_info.intent if intent_info else "unknown",
            complexity=intent_info.complexity if intent_info else "LOW",
        )
        if request_id:
            metrics.request_id = request_id

        self.logger.log_request_start(metrics)
        logger.debug("[ThinkEngine] Mode selected: %s (%s)", mode.value, rationale)
        return metrics, mode_config

    def finish_request(
        self,
        metrics: ThinkRequestMetrics,
        status: str = "completed",
    ) -> None:
        """Finalize metrics and emit structured summary log."""
        metrics.finish(status=status)
        self.logger.log_request_end(metrics)

    # -----------------------------------------------------------------------
    # Budget Enforcement
    # -----------------------------------------------------------------------

    def check_budget(
        self,
        metrics: ThinkRequestMetrics,
        mode_config: ThinkModeConfig,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check whether any resource budget has been exhausted.
        Returns (is_exceeded, reason_str).
        """
        budget = mode_config.budget
        elapsed = time.perf_counter() - metrics.start_time

        # 1. Runtime budget
        if budget.is_time_budget_exhausted(elapsed):
            metrics.was_timeout = True
            msg = f"Time budget exceeded: {elapsed:.2f}s >= {budget.max_runtime_seconds}s"
            metrics.record_error(msg)
            return True, msg

        # 2. Token budget
        if budget.is_token_budget_exhausted(metrics.total_tokens):
            msg = f"Token budget exceeded: {metrics.total_tokens} >= {budget.max_tokens}"
            metrics.record_error(msg)
            return True, msg

        # 3. Tool call budget
        if budget.is_tool_budget_exhausted(metrics.tool_calls_count):
            msg = f"Tool call budget exceeded: {metrics.tool_calls_count} >= {budget.max_tool_calls}"
            metrics.record_error(msg)
            return True, msg

        # 4. Iteration budget
        if budget.is_iteration_budget_exhausted(metrics.budget_iterations_used):
            msg = f"Iteration budget exceeded: {metrics.budget_iterations_used} >= {budget.max_iterations}"
            metrics.record_error(msg)
            return True, msg

        return False, None

    # -----------------------------------------------------------------------
    # Decomposition & Re-planning
    # -----------------------------------------------------------------------

    def decompose(
        self,
        user_request: str,
        intent_info: Optional[IntentInfo] = None,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> DecomposedObjective:
        """Decompose a complex request into a dependency-aware plan."""
        return self.decomposer.decompose(user_request, intent_info, context_data)

    def attempt_replan(
        self,
        failed_step: Dict[str, Any],
        error_detail: str,
        metrics: ThinkRequestMetrics,
        mode_config: ThinkModeConfig,
    ) -> Optional[Dict[str, Any]]:
        """
        Attempt dynamic re-planning if permitted by the thinking mode.
        Returns replacement step dictionary, or None if replanning is disabled/exhausted.
        """
        if not mode_config.allow_replan:
            return None

        if metrics.replan_count >= mode_config.max_replan_attempts:
            logger.info(
                "[ThinkEngine] Max replan attempts reached (%d/%d)",
                metrics.replan_count, mode_config.max_replan_attempts,
            )
            return None

        replacement = self.replan_engine.generate_alternative_step(
            failed_step=failed_step,
            error_detail=error_detail,
            attempt=metrics.replan_count + 1,
        )

        if replacement:
            metrics.record_replan()
            self.logger.log_replan(metrics, f"Step {failed_step.get('step')} fallback -> {replacement.get('tool')}")
            return replacement

        return None

    # -----------------------------------------------------------------------
    # Verification & Self-Checking
    # -----------------------------------------------------------------------

    def verify_claims(
        self,
        answer: str,
        step_observations: List[Dict[str, Any]],
        tools_used: List[str],
    ) -> ClaimVerificationReport:
        """
        Self-checking: Verify claims in generated answer against tool observations.

        Checks:
        1. Numerical consistency: Numbers claimed in text match tool results.
        2. Code block syntax: Python code in fenced blocks passes AST parse.
        3. Contradiction detection: Negative phrases like 'failed' vs claiming success.
        """
        report = ClaimVerificationReport()
        cleaned_ans = answer.strip()

        # 1. Check Python code blocks
        code_blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)\n```", cleaned_ans, re.DOTALL)
        for code in code_blocks:
            report.total_claims += 1
            code_str = code.strip()
            if not code_str:
                continue
            try:
                ast.parse(code_str)
                report.verified_claims += 1
                report.details.append({"type": "code_syntax", "status": "VERIFIED", "snippet": code_str[:60]})
            except SyntaxError as syn_err:
                report.failed_claims += 1
                report.has_syntax_error = True
                report.passed = False
                report.details.append({
                    "type": "code_syntax",
                    "status": "FAILED",
                    "error": str(syn_err),
                    "snippet": code_str[:60],
                })

        # 2. Check numerical results from tool observations
        # Collect observed numbers from tool results
        observed_numbers: Set[str] = set()
        for obs in step_observations:
            raw_res = obs.get("result") or obs.get("observation")
            if raw_res is not None:
                # Find all integers and decimals in observation
                found = re.findall(r"\b\d+(?:\.\d+)?\b", str(raw_res))
                observed_numbers.update(found)

        if observed_numbers:
            # Check numbers mentioned in the answer
            answer_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", cleaned_ans)
            for num in answer_numbers:
                report.total_claims += 1
                if num in observed_numbers:
                    report.verified_claims += 1
                    report.details.append({"type": "numeric_claim", "value": num, "status": "VERIFIED"})
                else:
                    # Number not in direct tool output — could be natural text or discrepancy
                    report.uncertain_claims += 1
                    report.details.append({"type": "numeric_claim", "value": num, "status": "UNCERTAIN"})

        # 3. Contradiction check: tool failed but answer claims success without mentioning failure
        failed_tools = [obs for obs in step_observations if obs.get("success") is False]
        if failed_tools and re.search(r"\b(?:successfully|computed exactly|verified result)\b", cleaned_ans, re.I):
            # Check if answer acknowledged the error
            if not re.search(r"\b(?:failed|error|unable to|could not)\b", cleaned_ans, re.I):
                report.has_contradiction = True
                report.passed = False
                report.failed_claims += 1
                report.total_claims += 1
                report.details.append({
                    "type": "contradiction",
                    "status": "FAILED",
                    "detail": "Answer claims success despite failed tool execution",
                })

        if report.total_claims == 0:
            # Natural language answer without verifiable numbers or code
            report.passed = True
            report.total_claims = 1
            report.verified_claims = 1
            report.details.append({"type": "conversational", "status": "NOT_APPLICABLE"})

        return report

    # -----------------------------------------------------------------------
    # Debug & UI Helpers
    # -----------------------------------------------------------------------

    def format_debug_block(
        self,
        metrics: ThinkRequestMetrics,
        mode_config: ThinkModeConfig,
        claim_report: Optional[ClaimVerificationReport] = None,
    ) -> str:
        """Format clean developer debug telemetry block."""
        budget = mode_config.budget
        lines = [
            "=================== THINK ENGINE 2.0 ===================",
            f"Mode:               {metrics.think_mode.value.upper()}",
            f"Intent:             {metrics.intent} (Complexity: {metrics.complexity})",
            f"Latency:            {metrics.total_latency_ms:.1f}ms",
            f"Tokens:             {metrics.total_tokens} (In: {metrics.tokens_input}, Out: {metrics.tokens_output})",
            f"Model Calls:        {metrics.model_calls}",
            f"Tool Calls:         {metrics.tool_calls_count} / {budget.max_tool_calls}",
            f"Replans:            {metrics.replan_count} / {mode_config.max_replan_attempts}",
            f"Verification:       {'PASSED' if metrics.verification_passed else metrics.verification_status}",
        ]
        if claim_report:
            lines.append(f"Claim Consistency:  {claim_report.consistency_score * 100:.1f}% ({claim_report.verified_claims}/{claim_report.total_claims})")
        if metrics.errors:
            lines.append(f"Errors Logged:      {len(metrics.errors)}")
        lines.append("=======================================================")
        return "\n".join(lines)
