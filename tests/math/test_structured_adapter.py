from __future__ import annotations

from dataclasses import dataclass, field

from reasoning.math.representations import MathResult
from reasoning.math.structured_adapter import StructuredMathAdapter, validate_structured_problem


VALID = {
    "domain": "algebra",
    "problem": "If x + 1/x = 3, find x^2 + 1/x^2",
    "variables": ["x"],
    "constraints": ["x + 1/x = 3"],
    "target": "x^2 + 1/x^2",
}


class FakeModel:
    def __init__(self, output):
        self.output = output

    def structured_output(self, prompt, **kwargs):
        return self.output


@dataclass
class FakeManager:
    available: bool = True
    output: object = field(default_factory=lambda: VALID)

    def health_check(self):
        return self.available

    def get_model(self):
        return FakeModel(self.output)

    def unavailable_message(self):
        return "Local model unavailable for test."


class FailedSolver:
    def solve(self, text):
        return MathResult(None, False, "test", "not verified", None, [], ["test"], 0.0)


def test_valid_structured_problem():
    problem = validate_structured_problem(VALID)

    assert problem.domain == "algebra"
    assert problem.constraints == ["x + 1/x = 3"]


def test_invalid_structured_problem_missing_field():
    payload = dict(VALID)
    del payload["target"]

    result = StructuredMathAdapter().solve_structured(payload)

    assert result.status == "needs_clarification"
    assert "target" in result.reason


def test_unsupported_domain():
    payload = dict(VALID, domain="quantum_astrology")

    result = StructuredMathAdapter().solve_structured(payload)

    assert result.status == "needs_clarification"
    assert "Unsupported" in result.reason


def test_unexpected_schema_field_is_rejected():
    result = StructuredMathAdapter().solve_structured(dict(VALID, extra="not allowed"))

    assert result.status == "needs_clarification"
    assert "Unexpected fields" in result.reason


def test_malformed_llm_output():
    result = StructuredMathAdapter().solve_from_llm_output("not json")

    assert result.status == "needs_clarification"
    assert "valid JSON" in result.reason


def test_successful_algebra_problem():
    result = StructuredMathAdapter().solve_structured(VALID)

    assert result.status == "ready"
    assert result.answer == 7
    assert result.verification.startswith("verified")


def test_successful_arithmetic_problem():
    payload = {
        "domain": "arithmetic",
        "problem": "Calculate 2 + 2",
        "variables": [],
        "constraints": [],
        "target": "2 + 2",
    }

    result = StructuredMathAdapter().solve_structured(payload)

    assert result.status == "ready"
    assert result.answer == 4


def test_verification_failure_is_not_reported_as_ready():
    result = StructuredMathAdapter(FailedSolver()).solve_structured(VALID)

    assert result.status == "verification_failed"
    assert result.answer is None


def test_llm_unavailable_is_explicit():
    result = StructuredMathAdapter().solve_with_model("2 + 2", FakeManager(available=False))

    assert result.status == "llm_unavailable"
    assert "unavailable" in result.reason.lower()


def test_unsafe_expression_is_rejected():
    payload = dict(VALID, target="__import__('os').system('echo unsafe')")

    result = StructuredMathAdapter().solve_structured(payload)

    assert result.status == "needs_clarification"
    assert "unsupported or unsafe" in result.reason


def test_llm_output_is_validated_before_solving():
    result = StructuredMathAdapter().solve_with_model("example", FakeManager(output="{bad json"))

    assert result.status == "needs_clarification"
