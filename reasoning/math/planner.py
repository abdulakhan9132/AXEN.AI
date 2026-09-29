from __future__ import annotations

from dataclasses import dataclass

from .representations import MathematicalProblem


@dataclass(frozen=True)
class Strategy:
    name: str
    categories: tuple[str, ...]


STRATEGIES = (
    Strategy("substitution", ("algebra", "equations")),
    Strategy("factoring", ("algebra", "equations", "inequalities")),
    Strategy("symmetry", ("algebra", "combinatorics", "geometry")),
    Strategy("modular_arithmetic", ("number theory",)),
    Strategy("recurrence", ("sequences/series", "discrete mathematics")),
    Strategy("differentiation", ("calculus", "optimization")),
    Strategy("integration", ("calculus",)),
    Strategy("matrix_method", ("linear algebra",)),
    Strategy("dynamic_programming", ("combinatorics", "graph theory", "discrete mathematics")),
    Strategy("simplification", ("algebra", "equations", "arithmetic")),
    Strategy("numerical_method", ("arithmetic", "equations", "calculus", "optimization")),
)


def select_strategies(problem: MathematicalProblem) -> list[str]:
    selected = [strategy.name for strategy in STRATEGIES if any(category in strategy.categories for category in problem.categories)]
    return selected or ["symbolic_method", "numerical_method"]
