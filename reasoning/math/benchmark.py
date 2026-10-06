from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .analyzer import analyze_problem
from .solver import MathSolver


@dataclass(frozen=True)
class BenchmarkCase:
    problem: str
    category: str
    expected: Any
    level: int


# Original deterministic cases; these are not presented as official contest problems.
CASES = [
    BenchmarkCase("calculate: 2 + 3", "arithmetic", "5", 1),
    BenchmarkCase("calculate: 12 * 7", "arithmetic", "84", 1),
    BenchmarkCase("calculate: 144 / 12", "arithmetic", "12", 1),
    BenchmarkCase("calculate: 2 ^ 10", "arithmetic", "1024", 1),
    BenchmarkCase("calculate: sqrt(81)", "arithmetic", "9", 1),
    BenchmarkCase("calculate: (x + 1) ^ 2", "algebra", "(x + 1)**2", 2),
    BenchmarkCase("solve x^2 - 9 = 0", "equations", "[-3, 3]", 2),
    BenchmarkCase("solve 2*x + 5 = 17", "equations", "[6]", 2),
    BenchmarkCase("solve x^2 + 5*x + 6 = 0", "equations", "[-3, -2]", 2),
    BenchmarkCase("solve inequality x > 2", "inequalities", "(2 < x) & (x < oo)", 2),
    BenchmarkCase("solve inequality x <= 4", "inequalities", "(-oo < x) & (x <= 4)", 2),
    BenchmarkCase("gcd of 84 and 126", "number theory", "42", 2),
    BenchmarkCase("gcd of 391 and 299", "number theory", "23", 3),
    BenchmarkCase("is 97 prime?", "number theory", "True", 2),
    BenchmarkCase("is 221 prime?", "number theory", "False", 2),
    BenchmarkCase("Euler phi(36)", "number theory", "12", 3),
    BenchmarkCase("Euler phi(101)", "number theory", "100", 3),
    BenchmarkCase("modular inverse of 3 modulo 11", "number theory", "4", 3),
    BenchmarkCase("modular inverse of 17 modulo 3120", "number theory", "2753", 3),
    BenchmarkCase("calculate: 7^5", "arithmetic", "16807", 2),
    BenchmarkCase("choose 5 from 12", "combinatorics", "792", 2),
    BenchmarkCase("choose 3 from 15", "combinatorics", "455", 2),
    BenchmarkCase("permutation of 5 items taken 3", "combinatorics", "60", 2),
    BenchmarkCase("permutation of 8 items taken 2", "combinatorics", "56", 2),
    BenchmarkCase("derivative of x^3 + 2*x with respect to x", "calculus", "3*x**2 + 2", 3),
    BenchmarkCase("derivative of sin(x) with respect to x", "calculus", "cos(x)", 3),
    BenchmarkCase("integral of 2*x with respect to x", "calculus", "x**2", 3),
    BenchmarkCase("integral of cos(x) with respect to x", "calculus", "sin(x)", 3),
    BenchmarkCase("limit of sin(x)/x as x 0", "calculus", "1", 3),
    BenchmarkCase("determinant [[1, 2], [3, 4]]", "linear algebra", "-2", 3),
    BenchmarkCase("determinant [[2, 0, 1], [3, 4, 5], [1, 0, 2]]", "linear algebra", "12", 3),
    BenchmarkCase("determinant [[1, 2, 3], [0, 1, 4], [5, 6, 0]]", "linear algebra", "1", 4),
    BenchmarkCase("distance between points (0, 0) and (3, 4)", "geometry", "5.0", 2),
    BenchmarkCase("distance between points (-1, 2) and (2, 6)", "geometry", "5.0", 3),
    BenchmarkCase("triangle area with base 10 and height 7", "geometry", "35.0", 2),
    BenchmarkCase("circle area with radius 2", "geometry", "12.566370614359172", 2),
    BenchmarkCase("slope between points (1, 2) and (3, 8)", "geometry", "3.0", 2),
    BenchmarkCase("probability choose 2 favorable from 5 successes in 10 trials selecting 4", "probability", "0.47619047619047616", 3),
    BenchmarkCase("mean of 2, 4, 6, 8", "statistics", "5.0", 2),
    BenchmarkCase("variance of 2, 4, 6, 8", "statistics", "5.0", 3),
    BenchmarkCase("fibonacci 10", "sequences/series", "55", 2),
    BenchmarkCase("fibonacci 20", "sequences/series", "6765", 3),
    BenchmarkCase("shortest path graph edges 1-2, 2-3, 3-4", "graph theory", "3", 3),
    BenchmarkCase("shortest path graph edges 5-6, 6-8, 5-7, 7-8", "graph theory", "2", 3),
    BenchmarkCase("solve differential equation y' = y", "differential equations", "Eq(y(x), C1*exp(x))", 5),
    BenchmarkCase("probability: choose 1 favorable from 3 successes in 8 trials selecting 2", "probability", "0.5357142857142857", 3),
    BenchmarkCase("calculate: (3 + 4) * (5 - 2)", "arithmetic", "21", 1),
    BenchmarkCase("calculate: 100 / (2 + 3)", "arithmetic", "20", 1),
    BenchmarkCase("calculate: (x + 2) * (x - 2)", "algebra", "x**2 - 4", 2),
    BenchmarkCase("gcd of 1001 and 143", "number theory", "143", 2),
]


def _matches(actual: Any, expected: Any) -> bool:
    if str(actual) == str(expected):
        return True
    try:
        import sympy as sp
        diff = sp.simplify(sp.sympify(str(actual)) - sp.sympify(str(expected)))
        return bool(diff == 0)
    except Exception:
        return False


def run_benchmark(solver: MathSolver | None = None) -> dict[str, Any]:
    engine = solver or MathSolver()
    records: list[dict[str, Any]] = []
    category_hits = 0
    for case in CASES:
        started = time.perf_counter()
        result = engine.solve(case.problem)
        elapsed = time.perf_counter() - started
        categories = analyze_problem(case.problem).categories
        category_ok = case.category in categories
        category_hits += int(category_ok)
        records.append({"problem": case.problem, "category": case.category, "expected": case.expected, "answer": result.answer, "solver_time": elapsed, "verified": result.verified, "solved": result.answer is not None and _matches(result.answer, case.expected), "strategy": result.method, "category_match": category_ok, "level": case.level})
    solved = sum(int(record["solved"]) for record in records)
    verified = sum(int(record["verified"]) for record in records)
    times = [record["solver_time"] for record in records]
    return {"problems": len(records), "solved": solved, "verified": verified, "failed": len(records) - solved, "average_time": sum(times) / len(times), "maximum_time": max(times), "category_accuracy": category_hits / len(records), "records": records}
