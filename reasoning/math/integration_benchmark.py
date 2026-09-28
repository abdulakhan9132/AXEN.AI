from __future__ import annotations

import time
from typing import Any

from .solver import MathSolver


INTEGRATION_CASES: tuple[tuple[str, str], ...] = (
    ("If x + 1/x = 3, find x^2 + 1/x^2", "algebra"),
    ("Calculate: 18 + 24", "arithmetic"),
    ("Calculate: 7 * 8", "arithmetic"),
    ("solve x^2 - 16 = 0", "algebra"),
    ("GCD of 144 and 96", "number theory"),
    ("Is 101 prime?", "number theory"),
    ("Euler phi(45)", "number theory"),
    ("Modular inverse of 7 modulo 26", "number theory"),
    ("derivative of x^4 with respect to x", "calculus"),
    ("integral of 3*x^2 with respect to x", "calculus"),
    ("limit of sin(x)/x as x 0", "calculus"),
    ("Choose 4 from 10", "combinatorics"),
    ("Permutation of 6 items taken 2", "combinatorics"),
    ("Fibonacci 12", "sequences/series"),
    ("Probability choose 1 favorable from 2 successes in 6 trials selecting 2", "probability"),
    ("Distance between points (0, 0) and (5, 12)", "geometry"),
    ("Triangle area with base 8 and height 5", "geometry"),
    ("Mean of 3, 6, 9", "statistics"),
    ("Determinant [[1, 2], [2, 5]]", "linear algebra"),
    ("Solve inequality x >= 3", "inequalities"),
)


def run_integration_benchmark(solver: MathSolver | None = None) -> dict[str, Any]:
    engine = solver or MathSolver()
    records = []
    for problem, category in INTEGRATION_CASES:
        started = time.perf_counter()
        result = engine.solve(problem)
        elapsed = time.perf_counter() - started
        records.append({"problem": problem, "category": category, "answer": result.answer, "verified": result.verified, "latency": elapsed, "method": result.method})
    latencies = [record["latency"] for record in records]
    return {
        "problems": len(records),
        "solved": sum(record["answer"] is not None for record in records),
        "verified": sum(record["verified"] for record in records),
        "failed": sum(record["answer"] is None for record in records),
        "llm_formalization_failures": 0,
        "solver_failures": sum(record["answer"] is None for record in records),
        "verification_failures": sum(not record["verified"] for record in records),
        "average_latency": sum(latencies) / len(latencies),
        "maximum_latency": max(latencies),
        "records": records,
    }
