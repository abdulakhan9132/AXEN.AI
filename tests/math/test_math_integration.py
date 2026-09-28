from __future__ import annotations

import pytest

from reasoning.math.integration_benchmark import INTEGRATION_CASES, run_integration_benchmark
from reasoning.math.solver import solve


@pytest.mark.parametrize("problem,category", INTEGRATION_CASES)
def test_original_natural_language_math_problem(problem: str, category: str):
    result = solve(problem)

    assert result.answer is not None, f"failed category={category}: {problem}"
    assert result.verified is True, f"unverified category={category}: {problem}"


def test_integration_benchmark_has_twenty_verified_cases():
    report = run_integration_benchmark()

    assert report["problems"] == 20
    assert report["solved"] == 20
    assert report["verified"] == 20
    assert report["failed"] == 0
