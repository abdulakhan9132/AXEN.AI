from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from .analyzer import analyze_problem
from .difficulty_estimator import estimate_difficulty
from .representations import MathResult
from .solver import MathSolver
from .strategy_validator import strategies_for_domains, validate_candidate_strategies
from .structured_adapter import StructuredProblem, _decode_json, validate_structured_problem


class ControllerModelManager(Protocol):
    def health_check(self) -> bool: ...
    def get_model(self): ...
    def unavailable_message(self) -> str: ...


@dataclass
class ControllerResult:
    status: str
    domains: list[str] = field(default_factory=list)
    strategy: str | None = None
    answer: Any = None
    verified: bool = False
    verification: str | None = None
    explanation: str | None = None
    structured_problem: dict[str, Any] | None = None
    difficulty: int | None = None
    confidence: float = 0.0
    reason: str | None = None
    attempts: list[dict[str, Any]] = field(default_factory=list)
    subproblems: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": self.status,
            "domains": self.domains,
            "strategy": self.strategy,
            "answer": self.answer,
            "verified": self.verified,
            "verification": self.verification,
            "explanation": self.explanation,
            "structured_problem": self.structured_problem,
            "difficulty": self.difficulty,
            "confidence": self.confidence,
            "attempts": self.attempts,
            "subproblems": self.subproblems,
            "timings": self.timings,
        }
        if self.reason is not None:
            result["reason"] = self.reason
        return result


class ReasoningController:
    def __init__(self, solver: MathSolver | None = None, *, max_strategy_attempts: int = 5, max_formalization_attempts: int = 3) -> None:
        if max_strategy_attempts <= 0:
            raise ValueError("max_strategy_attempts must be positive.")
        if max_formalization_attempts <= 0:
            raise ValueError("max_formalization_attempts must be positive.")
        self.solver = solver or MathSolver()
        self.max_strategy_attempts = max_strategy_attempts
        self.max_formalization_attempts = max_formalization_attempts

    def solve(self, problem: str, model_manager: ControllerModelManager) -> ControllerResult:
        started = time.perf_counter()
        if not model_manager.health_check():
            return ControllerResult(
                status="llm_unavailable",
                confidence=0.0,
                reason=model_manager.unavailable_message(),
                timings={"total": time.perf_counter() - started},
            )
        formalization_started = time.perf_counter()
        model = model_manager.get_model()
        prompt = self._formalization_prompt(problem)
        payload: Any = None
        last_error = "The local model did not return a valid structured problem."
        for attempt in range(1, self.max_formalization_attempts + 1):
            try:
                payload = model.structured_output(prompt, system="Formalize the problem as strict mathematical JSON.")
                validate_structured_problem(payload)
                break
            except (ValueError, SyntaxError) as exc:
                last_error = str(exc)
                if attempt == self.max_formalization_attempts:
                    timings = {"formalization": time.perf_counter() - formalization_started, "total": time.perf_counter() - started}
                    return ControllerResult("formalization_failed", confidence=0.0, reason=f"Formalization failed after {attempt} attempts: {last_error}", timings=timings)
                prompt = self._repair_prompt(problem, last_error)
            except Exception as exc:
                return ControllerResult("llm_unavailable", confidence=0.0, reason=str(exc), timings={"total": time.perf_counter() - started, "formalization": time.perf_counter() - formalization_started})
        formalization_time = time.perf_counter() - formalization_started
        return self.solve_structured(payload, model=model, original_problem=problem, initial_timings={"formalization": formalization_time}, started=started)

    @staticmethod
    def _formalization_prompt(problem: str) -> str:
        return (
            "Return JSON only for this mathematical problem. "
            "Required keys: domain, problem, variables, constraints, target. "
            "Optional key: candidate_strategies. "
            "The domain field means the mathematical PROBLEM CATEGORY, NOT the number system. "
            "Supported categories are: algebra, arithmetic, calculus, combinatorics, geometry, "
            "linear algebra, number theory, probability, statistics, discrete mathematics, "
            "sequences/series, differential equations, optimization, equations, inequalities. "
            "Never use 'real numbers', 'real', 'positive real numbers', 'integers', 'natural numbers', "
            "or 'complex numbers' as the domain. Number systems belong in constraints instead. "
            "Examples:\n"
            "- 'If x + 1/x = 3, find x^2 + 1/x^2' -> domain: 'algebra', variables: ['x'], constraints: ['x + 1/x = 3'], target: 'x^2 + 1/x^2'\n"
            "- 'Solve x + 5 = 12' -> domain: 'algebra', variables: ['x'], constraints: ['x + 5 = 12'], target: 'x'\n"
            "The target must be an expression or variable to solve for (not an equation or sentence). "
            "Return valid JSON only. Do not return code or explanations. Mathematical problem: " + problem
        )

    @staticmethod
    def _repair_prompt(problem: str, error: str) -> str:
        guidance = ""
        if "unsupported mathematical domain" in error.lower():
            guidance = (
                " The 'domain' field must contain a supported mathematical problem category "
                "(such as algebra, calculus, number theory, combinatorics, probability, geometry, linear algebra), "
                "NOT a number system like 'real numbers' or 'integers'. "
                "Move number system information into constraints. Choose the correct supported problem category based on the actual problem."
            )
        elif "invalid syntax" in error.lower() or "invalid mathematical expression" in error.lower():
            guidance = (
                " Constraints and targets must be pure mathematical formulas or equations without English command words like 'Solve' or 'Find'. "
                "The target must be the variable or expression to evaluate (e.g. 'x' or 'x^2 + 1/x^2'), NOT an equation like 'x = 7'."
            )
        return (
            f"Previous structured output is invalid.\n"
            f"Validation error: {error}\n"
            f"{guidance}\n"
            "Return only a corrected strict JSON object with keys: domain, problem, variables, constraints, target. "
            "Do not return explanations or markdown commentary outside JSON.\n"
            f"Original problem: {problem}"
        )

    def solve_structured(self, payload: str | dict[str, Any], *, model: Any = None, original_problem: str | None = None, initial_timings: dict[str, float] | None = None, started: float | None = None) -> ControllerResult:
        total_started = started or time.perf_counter()
        timings = dict(initial_timings or {})
        formalization_started = time.perf_counter()
        try:
            raw = _decode_json(payload)
            structured = validate_structured_problem(raw)
        except ValueError as exc:
            timings.setdefault("formalization", time.perf_counter() - formalization_started)
            timings["total"] = time.perf_counter() - total_started
            return ControllerResult("needs_clarification", confidence=0.0, reason=str(exc), timings=timings)
        timings.setdefault("formalization", time.perf_counter() - formalization_started)

        classification_started = time.perf_counter()
        analyzed = analyze_problem(structured.problem)
        domains = list(dict.fromkeys([structured.domain, *analyzed.categories]))
        difficulty, _ = estimate_difficulty(structured.problem)
        timings["classification"] = time.perf_counter() - classification_started

        strategy_started = time.perf_counter()
        try:
            candidates = validate_candidate_strategies(domains, structured.candidate_strategies)
        except ValueError as exc:
            timings["strategy_selection"] = time.perf_counter() - strategy_started
            timings["total"] = time.perf_counter() - total_started
            return ControllerResult("needs_clarification", domains=domains, structured_problem=structured.to_dict(), difficulty=difficulty, confidence=0.0, reason=str(exc), timings=timings)
        strategies = candidates or strategies_for_domains(domains)
        strategies = strategies[: self.max_strategy_attempts]
        timings["strategy_selection"] = time.perf_counter() - strategy_started
        subproblems = self._decompose(structured)

        attempts: list[dict[str, Any]] = []
        final: MathResult | None = None
        solver_started = time.perf_counter()
        for strategy in strategies:
            result = self.solver.solve(self._solver_input(structured, original_problem))
            attempts.append({"strategy": strategy, "verified": result.verified, "method": result.method, "verification": result.verification})
            final = result
            if result.verified:
                break
        timings["solver"] = time.perf_counter() - solver_started
        timings["verification"] = 0.0
        if final is None or not final.verified:
            timings["total"] = time.perf_counter() - total_started
            return ControllerResult("verification_failed", domains=domains, strategy=attempts[-1]["strategy"] if attempts else None, verified=False, verification=final.verification if final else "No supported strategy was available.", structured_problem=structured.to_dict(), difficulty=difficulty, confidence=0.5, reason="All bounded strategy attempts failed verification.", attempts=attempts, subproblems=subproblems, timings=timings)

        explanation_started = time.perf_counter()
        explanation = self._explain(final, structured, model)
        timings["explanation"] = time.perf_counter() - explanation_started
        timings["total"] = time.perf_counter() - total_started
        return ControllerResult("solved", domains=domains, strategy=attempts[-1]["strategy"], answer=final.answer, verified=True, verification=final.verification, explanation=explanation, structured_problem=structured.to_dict(), difficulty=difficulty, confidence=0.95, attempts=attempts, subproblems=subproblems, timings=timings)

    @staticmethod
    def _solver_input(problem: StructuredProblem, original_problem: str | None = None) -> str:
        source = (original_problem or problem.problem).strip()
        if "solve " in source.lower() and "=" in source:
            return source
        # Prioritize equation constraint with '='
        eq_c = next((c for c in problem.constraints if "=" in c), None)
        if eq_c:
            if problem.target.strip() in problem.variables or re.fullmatch(r"[a-zA-Z_]\w*", problem.target.strip()):
                return f"solve {eq_c}"
            return f"If {eq_c} find {problem.target}"
        if problem.constraints:
            math_c = next((c for c in problem.constraints if not c.lower().startswith("number system:")), None)
            if math_c:
                return f"If {math_c} find {problem.target}"
        if "=" in source and "find" in source.lower():
            return source
        specialized_markers = (
            "gcd", "prime", "phi", "modular inverse", "choose", "permutation", "probability",
            "derivative of", "integral of", "limit of", "determinant", "distance", "mean of",
            "variance of", "fibonacci", "shortest path", "triangle area", "circle area", "slope",
            "solve inequality", "differential equation", "calculate", "solve ",
        )
        if any(marker in source.lower() for marker in specialized_markers):
            return source.lower()
        return problem.target

    @staticmethod
    def _decompose(problem: StructuredProblem) -> list[str]:
        parts = [f"Use constraint: {constraint}" for constraint in problem.constraints]
        parts.append(f"Evaluate target: {problem.target}")
        return parts

    @staticmethod
    def _explain(result: MathResult, problem: StructuredProblem, model: Any) -> str:
        concise = " ".join(result.steps[-4:])
        fallback = f"Answer: {result.answer}. Method: {result.method}. Steps: {concise} Verification: {result.verification}"
        if model is None or not hasattr(model, "generate"):
            return fallback
        prompt = json.dumps({"answer": str(result.answer), "method": result.method, "steps": result.steps[-4:], "verification": result.verification})
        try:
            explanation = model.generate("Explain this verified mathematical result in 2-4 concise sentences. Do not add new calculations or private reasoning. Data: " + prompt, system="Give only a concise mathematical explanation.")
        except (RuntimeError, OSError, ValueError):
            return fallback
        return explanation.strip() or fallback
