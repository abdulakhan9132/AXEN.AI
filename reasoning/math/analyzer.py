from __future__ import annotations

import re

from .problem_types import ProblemCategory
from .representations import MathematicalProblem

_KEYWORDS: dict[ProblemCategory, tuple[str, ...]] = {
    ProblemCategory.NUMBER_THEORY: ("prime", "gcd", "divisor", "mod", "congruen", "factor", "integer", "phi"),
    ProblemCategory.CALCULUS: ("derivative", "differentiate", "integral", "integrate", "limit", "taylor", "series"),
    ProblemCategory.LINEAR_ALGEBRA: ("matrix", "determinant", "eigen", "vector", "linear system", "rank"),
    ProblemCategory.GEOMETRY: ("distance", "slope", "circle", "triangle", "point", "coordinate", "intersection"),
    ProblemCategory.PROBABILITY: ("probability", "bayes", "expected", "variance", "distribution", "random"),
    ProblemCategory.STATISTICS: ("mean", "median", "standard deviation", "variance", "sample"),
    ProblemCategory.COMBINATORICS: ("permutation", "combination", "choose", "binomial", "arrange", "count"),
    ProblemCategory.GRAPH_THEORY: ("graph", "vertex", "vertices", "edge", "path", "tree"),
    ProblemCategory.OPTIMIZATION: ("maximize", "minimize", "optimization", "optimal", "minimum", "maximum"),
    ProblemCategory.DIFFERENTIAL_EQUATIONS: ("differential equation", "dy/dx", "ordinary differential"),
    ProblemCategory.SEQUENCES_SERIES: ("sequence", "recurrence", "arithmetic progression", "geometric progression", "series", "term", "fibonacci"),
    ProblemCategory.INEQUALITIES: ("inequality", "inequalities", "less than", "greater than", "maximum", "minimum"),
    ProblemCategory.EQUATIONS: ("solve", "equation", "find x", "root", "roots"),
    ProblemCategory.ALGEBRA: ("polynomial", "factor", "expression", "identity", "variable", "x ", "simplify", "expand", "f(x)", "g(x)"),
    ProblemCategory.DISCRETE_MATH: ("boolean", "set", "logic", "recurrence", "discrete"),
    ProblemCategory.ARITHMETIC: ("calculate", "compute", "what is", "sum", "product", "plus", "minus"),
}


def analyze_problem(text: str) -> MathematicalProblem:
    if not text or not text.strip():
        raise ValueError("Mathematical problem cannot be empty.")
    normalized = " ".join(text.lower().split())
    categories: list[str] = []
    for category, keywords in _KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            categories.append(category.value)
    if not categories:
        categories = [ProblemCategory.ARITHMETIC.value]
    constraints: list[str] = []
    for match in re.findall(r"[^,;]+=[^,;]+", text):
        if match.strip() not in constraints:
            constraints.append(match.strip())
    target = None
    target_match = re.search(r"(?:find|compute|calculate|evaluate|determine)\s+(.+?)(?:\?|$)", text, re.IGNORECASE)
    if target_match:
        target = target_match.group(1).strip().rstrip(".")
    return MathematicalProblem(text=text, categories=categories, constraints=constraints, target=target)
