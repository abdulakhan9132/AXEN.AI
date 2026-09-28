from __future__ import annotations

from reasoning.math.analyzer import analyze_problem
from reasoning.math.benchmark import run_benchmark
from reasoning.math.solver import solve
from math_engines.linear_algebra import determinant
from math_engines.number_theory import chinese_remainder, modular_inverse


def test_reciprocal_identity_is_formalized_and_verified():
    result = solve("If x + 1/x = 3, find x^2 + 1/x^2")

    assert result.answer == 7
    assert result.verified is True
    assert result.problem is not None
    assert result.problem.constraints == ["x + 1/x = 3"]
    assert result.problem.target == "x^2 + 1/x^2"
    assert result.steps


def test_analyzer_supports_mixed_categories():
    problem = analyze_problem("Find the probability that a prime integer satisfies an equation.")

    assert "probability" in problem.categories
    assert "number theory" in problem.categories
    assert "equations" in problem.categories


def test_math_parser_rejects_unsafe_python_constructs():
    result = solve("__import__('os').system('echo unsafe')")

    assert result.verified is False
    assert result.answer is None


def test_domain_engines_verify_exact_operations():
    assert determinant([[1, 2], [3, 4]]) == -2
    assert modular_inverse(3, 11) == 4
    assert chinese_remainder([2, 3], [3, 5]) == 8


def test_benchmark_is_deterministic_and_fully_verified():
    report = run_benchmark()

    assert report["problems"] == 50
    assert report["solved"] == 50
    assert report["verified"] == 50
    assert report["failed"] == 0
    assert 0.9 <= report["category_accuracy"] <= 1.0
