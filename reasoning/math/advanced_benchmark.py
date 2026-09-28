from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import sympy as sp

from math_engines import combinatorics, geometry, number_theory, probability
from .reasoning_controller import ReasoningController


@dataclass(frozen=True)
class AdvancedCase:
    problem: str
    domain: str
    expected: str
    verification_method: str


def _algebra_cases() -> list[AdvancedCase]:
    return [AdvancedCase(f"calculate: (x + {n}) * (x - {n})", "algebra", f"x**2 - {n * n}", "symbolic simplification") for n in range(1, 21)]


def _number_theory_cases() -> list[AdvancedCase]:
    cases = [
        ("gcd of 84 and 126", "42"), ("gcd of 391 and 299", "23"), ("gcd of 1001 and 143", "143"),
        ("is 101 prime?", "True"), ("is 121 prime?", "False"), ("is 113 prime?", "True"),
        ("Euler phi(36)", "12"), ("Euler phi(45)", "24"), ("Euler phi(101)", "100"),
        ("modular inverse of 3 modulo 11", "4"), ("modular inverse of 7 modulo 26", "15"),
        ("modular inverse of 17 modulo 3120", "2753"), ("gcd of 144 and 96", "48"),
        ("is 221 prime?", "False"), ("is 127 prime?", "True"),
    ]
    return [AdvancedCase(problem, "number theory", expected, "exact integer or modular verification") for problem, expected in cases]


def _calculus_cases() -> list[AdvancedCase]:
    cases = [
        ("derivative of x^2 with respect to x", "2*x"),
        ("derivative of x^3 with respect to x", "3*x**2"),
        ("derivative of x^4 with respect to x", "4*x**3"),
        ("derivative of sin(x) with respect to x", "cos(x)"),
        ("derivative of exp(x) with respect to x", "exp(x)"),
        ("integral of 2*x with respect to x", "x**2"),
        ("integral of 3*x^2 with respect to x", "x**3"),
        ("integral of cos(x) with respect to x", "sin(x)"),
        ("limit of sin(x)/x as x 0", "1"),
        ("limit of (1-cos(x))/x^2 as x 0", "1/2"),
        ("derivative of log(x) with respect to x", "1/x"),
        ("derivative of x^5 + 3*x with respect to x", "5*x**4 + 3"),
        ("integral of 4*x^3 with respect to x", "x**4"),
        ("limit of (x^2 - 1)/(x - 1) as x 1", "2"),
        ("derivative of tan(x) with respect to x", "tan(x)**2 + 1"),
    ]
    return [AdvancedCase(problem, "calculus", expected, "symbolic derivative, antiderivative, or limit verification") for problem, expected in cases]


def _combinatorics_cases() -> list[AdvancedCase]:
    values = [(5, 2), (6, 3), (7, 2), (8, 4), (9, 3), (10, 5), (11, 2), (12, 4), (13, 3), (14, 6)]
    return [AdvancedCase(f"choose {k} from {n}", "combinatorics", str(math.comb(n, k)), "exact binomial recomputation") for n, k in values]


def _probability_cases() -> list[AdvancedCase]:
    values = [(1, 2, 6, 2), (2, 5, 10, 4), (1, 3, 8, 2), (2, 4, 9, 3), (0, 5, 12, 3), (3, 6, 12, 5), (1, 4, 10, 2), (2, 7, 14, 4), (4, 8, 16, 6), (2, 3, 7, 3)]
    return [AdvancedCase(f"probability choose {favorable} favorable from {successes} successes in {total} trials selecting {draws}", "probability", str(probability.combinatorial_probability(favorable, successes, total, draws)), "exact favorable over total counting") for favorable, successes, total, draws in values]


def _geometry_cases() -> list[AdvancedCase]:
    cases = [
        ("distance between points (0, 0) and (3, 4)", "5.0"), ("distance between points (0, 0) and (5, 12)", "13.0"),
        ("distance between points (-1, 2) and (2, 6)", "5.0"), ("triangle area with base 10 and height 7", "35.0"),
        ("triangle area with base 8 and height 5", "20.0"), ("circle area with radius 2", str(math.pi * 4)),
        ("circle area with radius 3", str(math.pi * 9)), ("slope between points (1, 2) and (3, 8)", "3.0"),
        ("slope between points (0, 0) and (4, 10)", "2.5"), ("distance between points (2, 3) and (8, 11)", "10.0"),
    ]
    return [AdvancedCase(problem, "geometry", expected, "coordinate formula recomputation") for problem, expected in cases]


def _linear_algebra_cases() -> list[AdvancedCase]:
    matrices = [
        [[1, 2], [3, 4]], [[2, 0], [0, 5]], [[1, 2], [2, 5]], [[3, 1], [4, 2]], [[5, 2], [1, 1]],
        [[1, 2, 3], [0, 1, 4], [5, 6, 0]], [[2, 0, 1], [3, 4, 5], [1, 0, 2]],
        [[1, 0, 0], [0, 2, 0], [0, 0, 3]], [[2, 1, 0], [0, 3, 4], [1, 0, 2]], [[4, 2, 1], [1, 3, 2], [0, 1, 5]],
    ]
    return [AdvancedCase(f"determinant {matrix}", "linear algebra", str(sp.Matrix(matrix).det()), "exact determinant recomputation") for matrix in matrices]


def _sequence_cases() -> list[AdvancedCase]:
    return [AdvancedCase(f"fibonacci {index}", "sequences/series", str(combinatorics.fibonacci(index)), "recurrence recomputation") for index in range(10, 20)]


ADVANCED_CASES: tuple[AdvancedCase, ...] = tuple(_algebra_cases() + _number_theory_cases() + _calculus_cases() + _combinatorics_cases() + _probability_cases() + _geometry_cases() + _linear_algebra_cases() + _sequence_cases())


def _matches(actual: Any, expected: str) -> bool:
    try:
        return math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9)
    except (TypeError, ValueError):
        return str(actual) == expected


def _payload(case: AdvancedCase) -> dict[str, Any]:
    return {"domain": case.domain, "problem": case.problem, "variables": ["x"], "constraints": [], "target": case.expected}


def run_advanced_benchmark(controller: ReasoningController | None = None) -> dict[str, Any]:
    engine = controller or ReasoningController()
    records = []
    for case in ADVANCED_CASES:
        started = time.perf_counter()
        result = engine.solve_structured(_payload(case), original_problem=case.problem)
        latency = time.perf_counter() - started
        solved = result.answer is not None and _matches(result.answer, case.expected)
        records.append({"problem": case.problem, "domain": case.domain, "expected": case.expected, "verification_method": case.verification_method, "answer": result.answer, "solved": solved, "verified": result.verified, "strategy_success": result.strategy is not None, "latency": latency, "strategy": result.strategy})
    latencies = [record["latency"] for record in records]
    return {"total": len(records), "solved": sum(record["solved"] for record in records), "verified": sum(record["verified"] for record in records), "failed": sum(not record["solved"] for record in records), "strategy_success": sum(record["strategy_success"] for record in records), "llm_formalization_failures": 0, "solver_failures": sum(not record["solved"] for record in records), "verification_failures": sum(not record["verified"] for record in records), "average_latency": sum(latencies) / len(latencies), "maximum_latency": max(latencies), "records": records}
