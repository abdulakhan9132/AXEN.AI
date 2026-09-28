from __future__ import annotations

from dataclasses import dataclass, field

from reasoning.math.reasoning_controller import ReasoningController
from reasoning.math.representations import MathResult
from reasoning.math.strategy_validator import validate_candidate_strategies


VALID = {
    "domain": "algebra",
    "problem": "If x + 1/x = 3, find x^2 + 1/x^2",
    "variables": ["x"],
    "constraints": ["x + 1/x = 3"],
    "target": "x^2 + 1/x^2",
    "candidate_strategies": ["polynomial_identity", "substitution"],
}


class FakeModel:
    def __init__(self, payload):
        self.payload = payload

    def structured_output(self, prompt, **kwargs):
        return self.payload

    def generate(self, prompt, **kwargs):
        return "The verified answer is 7. Squaring the constraint gives the target as 7."


@dataclass
class FakeManager:
    available: bool = True
    payload: object = field(default_factory=lambda: VALID)

    def health_check(self):
        return self.available

    def get_model(self):
        return FakeModel(self.payload)

    def unavailable_message(self):
        return "Local model unavailable."


class RetrySolver:
    def __init__(self):
        self.calls = 0

    def solve(self, text):
        self.calls += 1
        if self.calls == 1:
            return MathResult(None, False, "first", "not verified", None, [], ["first"], 0.0)
        return MathResult(7, True, "second", "verified: test identity", None, ["Use the identity."], ["second"], 0.0)


def test_controller_solves_and_explains_verified_result():
    result = ReasoningController().solve("If x + 1/x = 3, find x^2 + 1/x^2", FakeManager())

    assert result.status == "solved"
    assert result.answer == 7
    assert result.verified is True
    assert result.strategy == "polynomial_identity"
    assert result.explanation
    assert result.timings["total"] >= 0


def test_controller_rejects_invalid_strategy_before_solving():
    payload = dict(VALID, candidate_strategies=["shell_command"])

    result = ReasoningController().solve_structured(payload)

    assert result.status == "needs_clarification"
    assert "Unsupported strategy" in result.reason


def test_controller_retries_after_verification_failure():
    solver = RetrySolver()
    result = ReasoningController(solver=solver, max_strategy_attempts=2).solve_structured(VALID)

    assert result.status == "solved"
    assert result.answer == 7
    assert solver.calls == 2
    assert len(result.attempts) == 2


def test_controller_has_bounded_attempts():
    solver = RetrySolver()
    result = ReasoningController(solver=solver, max_strategy_attempts=1).solve_structured(VALID)

    assert result.status == "verification_failed"
    assert solver.calls == 1


def test_controller_reports_llm_unavailable():
    result = ReasoningController().solve("2 + 2", FakeManager(available=False))

    assert result.status == "llm_unavailable"
    assert result.verified is False


def test_strategy_validation_enforces_domain_compatibility():
    assert validate_candidate_strategies(["number theory"], ["gcd"]) == ["gcd"]
    try:
        validate_candidate_strategies(["geometry"], ["gcd"])
    except ValueError as exc:
        assert "incompatible" in str(exc)
    else:
        raise AssertionError("Incompatible strategy was accepted")
