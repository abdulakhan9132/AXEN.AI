from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from .analyzer import analyze_problem
from .problem_types import ProblemCategory
from .solver import MathSolver, _safe_expression


SUPPORTED_DOMAINS = frozenset(category.value for category in ProblemCategory)
_REQUIRED_FIELDS = ("domain", "problem", "variables", "constraints", "target")
_ALLOWED_FIELDS = frozenset((*_REQUIRED_FIELDS, "candidate_strategies"))

DOMAIN_ALIASES: dict[str, str] = {
    "number_theory": "number theory",
    "linear_algebra": "linear algebra",
    "discrete_math": "discrete mathematics",
    "discrete_mathematics": "discrete mathematics",
    "discrete": "discrete mathematics",
    "sequences_series": "sequences/series",
    "sequences": "sequences/series",
    "series": "sequences/series",
    "differential_equations": "differential equations",
    "graph_theory": "graph theory",
}

NUMBER_SYSTEM_DOMAINS: frozenset[str] = frozenset({
    "real numbers", "real", "positive real numbers", "nonzero real numbers",
    "integers", "positive integers", "negative integers", "natural numbers",
    "rational numbers", "complex numbers", "nonzero numbers",
})


import re


def normalize_structured_payload(raw: dict[str, Any]) -> dict[str, Any]:
    norm = dict(raw)

    # 1. Normalize variables
    raw_vars = norm.get("variables")
    if isinstance(raw_vars, dict):
        norm["variables"] = [str(k).strip() for k in raw_vars.keys() if str(k).strip().isidentifier()]
    elif isinstance(raw_vars, list):
        clean_vars: list[str] = []
        for item in raw_vars:
            if isinstance(item, str):
                s = item.strip()
                if s.isidentifier() and s not in clean_vars:
                    clean_vars.append(s)
                else:
                    for token in re.findall(r"\b[a-zA-Z_]\w*\b", s):
                        if token.isidentifier() and token not in ("sin", "cos", "tan", "sqrt", "log", "exp", "abs", "real", "numbers") and token not in clean_vars:
                            clean_vars.append(token)
        if clean_vars:
            norm["variables"] = clean_vars

    # 2. Normalize constraints
    raw_constraints = norm.get("constraints")
    if isinstance(raw_constraints, dict):
        norm["constraints"] = [str(k).strip() for k in raw_constraints.keys() if str(k).strip()]
    elif isinstance(raw_constraints, str):
        norm["constraints"] = [raw_constraints.strip()]
    elif not isinstance(raw_constraints, list):
        norm["constraints"] = []

    # Filter out placeholder keys like "domain", "none", "equation"
    clean_constraints: list[str] = []
    for c in norm["constraints"]:
        clean_c = c.strip()
        if clean_c.lower() in ("domain", "none", "n/a", "equation"):
            continue
        for prefix in ("solve ", "given ", "if ", "the equation ", "equation: ", "where "):
            if clean_c.lower().startswith(prefix):
                clean_c = clean_c[len(prefix):].strip()
        clean_c = clean_c.rstrip(".?;:,")
        if clean_c and clean_c not in clean_constraints:
            clean_constraints.append(clean_c)
    norm["constraints"] = clean_constraints

    # If constraints list has no equation with '=', extract equation from problem statement if present
    prob = str(norm.get("problem", ""))
    extracted_equations = []
    for match in re.findall(r"[^,;\n]+=[^,;\n]+", prob):
        clean_match = match.strip()
        for prefix in ("solve ", "given ", "if ", "the equation ", "equation: ", "where "):
            if clean_match.lower().startswith(prefix):
                clean_match = clean_match[len(prefix):].strip()
        clean_match = clean_match.rstrip(".?;:,")
        if clean_match:
            extracted_equations.append(clean_match)

    if extracted_equations:
        full_eq = extracted_equations[0]
        # Remove partial matches like '1/x = 3' if 'x + 1/x = 3' is available
        norm["constraints"] = [c for c in norm["constraints"] if c not in full_eq or c == full_eq]
        if full_eq not in norm["constraints"]:
            norm["constraints"].insert(0, full_eq)

    # 3. Normalize target expression
    raw_target = norm.get("target")
    if isinstance(raw_target, str):
        target_str = raw_target.strip()
        for prefix in (
            "find the value of ", "find the value for ", "find value of ",
            "find the ", "find ", "compute the value of ", "compute ",
            "calculate the value of ", "calculate ", "evaluate ", "determine the value of ", "determine "
        ):
            if target_str.lower().startswith(prefix):
                target_str = target_str[len(prefix):].strip()
        target_str = re.sub(r"\s+that satisfies.*$", "", target_str, flags=re.IGNORECASE).strip()
        target_str = target_str.rstrip(".?;:,")
        if "=" in target_str:
            parts = [p.strip() for p in target_str.split("=")]
            if parts[0].isidentifier():
                target_str = parts[0]
            elif len(parts) > 1 and parts[1].isidentifier():
                target_str = parts[1]
        norm["target"] = target_str

    # 4. Normalize domain
    domain = norm.get("domain")
    if isinstance(domain, str):
        clean = domain.strip().lower()
        if clean in DOMAIN_ALIASES:
            norm["domain"] = DOMAIN_ALIASES[clean]
            return norm

        if clean in NUMBER_SYSTEM_DOMAINS:
            prob = str(norm.get("problem", ""))
            target = str(norm.get("target", ""))
            full = f"{prob} {target}".strip()
            constraints = list(norm.get("constraints", []))

            analyzed = analyze_problem(full) if full else None
            if analyzed and analyzed.categories:
                if "algebra" in analyzed.categories:
                    candidate = "algebra"
                else:
                    candidate = next((cat for cat in analyzed.categories if cat in SUPPORTED_DOMAINS), None)
                if candidate:
                    norm["domain"] = candidate
                    assumption = f"number system: {clean}"
                    if assumption not in constraints and not any(clean in c.lower() for c in constraints):
                        constraints.append(assumption)
                    norm["constraints"] = constraints
                    return norm

    return norm


class ModelManagerLike(Protocol):
    def health_check(self) -> bool: ...
    def get_model(self): ...
    def unavailable_message(self) -> str: ...


@dataclass
class StructuredProblem:
    domain: str
    problem: str
    variables: list[str]
    constraints: list[str]
    target: str
    candidate_strategies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "problem": self.problem,
            "variables": self.variables,
            "constraints": self.constraints,
            "target": self.target,
            "candidate_strategies": self.candidate_strategies,
        }


@dataclass
class AdapterResult:
    status: str
    confidence: float
    answer: Any = None
    steps: list[str] | None = None
    verification: str | None = None
    structured_problem: dict[str, Any] | None = None
    reason: str | None = None
    method: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"status": self.status, "confidence": self.confidence}
        if self.answer is not None: result["answer"] = self.answer
        if self.steps is not None: result["steps"] = self.steps
        if self.verification is not None: result["verification"] = self.verification
        if self.structured_problem is not None: result["structured_problem"] = self.structured_problem
        if self.reason is not None: result["reason"] = self.reason
        if self.method is not None: result["method"] = self.method
        return result


def _decode_json(payload: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, dict):
        value = payload
    elif isinstance(payload, str):
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM output is not valid JSON: {exc.msg}.") from exc
    else:
        raise ValueError("LLM output must be a JSON object or JSON text.")
    if not isinstance(value, dict):
        raise ValueError("Structured mathematical output must be a JSON object.")
    return value


def _validate_expression(expression: str, *, allow_equation: bool = False) -> None:
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Mathematical expressions must be non-empty strings.")
    lowered = expression.lower()
    if any(marker in lowered for marker in ("number system:", "is a ", "is real", "are real", "positive integer", "nonzero")):
        return
    candidates = [expression]
    if allow_equation and "=" in expression:
        left, right = expression.split("=", 1)
        candidates = [left, right]
    for candidate in candidates:
        try:
            _safe_expression(candidate.strip())
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Invalid mathematical expression {candidate!r}: {exc}") from exc


def validate_structured_problem(payload: str | dict[str, Any]) -> StructuredProblem:
    value = _decode_json(payload)
    value = normalize_structured_payload(value)
    missing = [field for field in _REQUIRED_FIELDS if field not in value]
    if missing:
        raise ValueError("Missing required fields: " + ", ".join(missing) + ".")
    unexpected = sorted(set(value) - _ALLOWED_FIELDS)
    if unexpected:
        raise ValueError("Unexpected fields: " + ", ".join(unexpected) + ".")
    domain = value["domain"]
    if not isinstance(domain, str) or domain not in SUPPORTED_DOMAINS:
        raise ValueError(f"Unsupported mathematical domain: {domain!r}.")
    problem = value["problem"]
    target = value["target"]
    variables = value["variables"]
    constraints = value["constraints"]
    candidate_strategies = value.get("candidate_strategies", [])
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("problem must be a non-empty string.")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("target must be a non-empty string.")
    if not isinstance(variables, list) or not all(isinstance(item, str) and item.isidentifier() for item in variables):
        raise ValueError("variables must be a list of identifier names.")
    if not isinstance(constraints, list) or not all(isinstance(item, str) and item.strip() for item in constraints):
        raise ValueError("constraints must be a list of non-empty strings.")
    if not isinstance(candidate_strategies, list) or not all(isinstance(item, str) and item.strip() for item in candidate_strategies):
        raise ValueError("candidate_strategies must be a list of non-empty strings.")
    for constraint in constraints:
        _validate_expression(constraint, allow_equation=True)
    _validate_expression(target)
    return StructuredProblem(domain, problem.strip(), variables, [item.strip() for item in constraints], target.strip(), [item.strip() for item in candidate_strategies])


def _solver_input(problem: StructuredProblem) -> str:
    eq_c = next((c for c in problem.constraints if "=" in c), None)
    if eq_c:
        if problem.target.strip() in problem.variables or re.fullmatch(r"[a-zA-Z_]\w*", problem.target.strip()):
            return f"solve {eq_c}"
        return f"If {eq_c} find {problem.target}"
    if problem.constraints:
        return f"If {problem.constraints[0]} find {problem.target}"
    return problem.target


class StructuredMathAdapter:
    def __init__(self, solver: MathSolver | None = None) -> None:
        self.solver = solver or MathSolver()

    def solve_structured(self, payload: str | dict[str, Any]) -> AdapterResult:
        try:
            problem = validate_structured_problem(payload)
        except ValueError as exc:
            return AdapterResult("needs_clarification", 0.0, reason=str(exc))
        result = self.solver.solve(_solver_input(problem))
        structured = problem.to_dict()
        if not result.verified:
            return AdapterResult("verification_failed", 0.65, structured_problem=structured, reason=result.verification, method=result.method)
        return AdapterResult("ready", 0.95, answer=result.answer, steps=result.steps[-4:], verification=result.verification, structured_problem=structured, method=result.method)

    def solve_from_llm_output(self, payload: str | dict[str, Any]) -> AdapterResult:
        return self.solve_structured(payload)

    def solve_with_model(self, problem: str, model_manager: ModelManagerLike) -> AdapterResult:
        if not model_manager.health_check():
            return AdapterResult("llm_unavailable", 0.0, reason=model_manager.unavailable_message())
        prompt = (
            "Return JSON only for this mathematical problem. "
    "Required keys are domain, problem, variables, constraints, target. "
    "The domain field means the mathematical PROBLEM CATEGORY, not the "
    "number system or type of numbers. "
    "Use exactly one supported problem category. "
    "Valid examples include algebra, number_theory, calculus, "
    "combinatorics, probability, geometry, linear_algebra, and "
    "sequences_series, using the exact category names supported by the program. "
    "Never use real numbers, real, positive real numbers, integers, "
    "positive integers, natural numbers, rational numbers, or complex numbers "
    "as the domain. Put number-system information in constraints instead. "
    "For example, 'If x + 1/x = 3, find x^2 + 1/x^2' is an algebra problem, "
    "not a 'real numbers' problem. "
    "Do not include explanations, code, or extra keys. "
    "Return valid JSON only. Problem: "
        )
        try:
            payload = model_manager.get_model().structured_output(prompt, system="Formalize mathematics as strict JSON only.")
        except (RuntimeError, OSError, ValueError) as exc:
            return AdapterResult("llm_unavailable", 0.0, reason=str(exc))
        return self.solve_from_llm_output(payload)
