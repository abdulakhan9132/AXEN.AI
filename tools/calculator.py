from __future__ import annotations

from reasoning.math.solver import solve


def calculate(problem: str):
    """Solve a controlled mathematical expression without evaluating Python input."""
    return solve(problem)
