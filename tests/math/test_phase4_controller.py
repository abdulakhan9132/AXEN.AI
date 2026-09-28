from __future__ import annotations

from dataclasses import dataclass, field

from reasoning.math.reasoning_controller import ReasoningController
from reasoning.math.representations import MathResult


VALID = {
    "domain": "algebra",
    "problem": "If x + 1/x = 3, find x^2 + 1/x^2",
    "variables": ["x"],
    "constraints": ["x + 1/x = 3"],
    "target": "x^2 + 1/x^2",
    "candidate_strategies": ["polynomial_identity", "substitution"],
}


class SequenceModel:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = 0

    def structured_output(self, prompt, **kwargs):
        self.calls += 1
        return next(self.outputs)

    def generate(self, prompt, **kwargs):
        return "Verified answer: 7."


@dataclass
class FakeManager:
    outputs: list[object] = field(default_factory=lambda: [VALID])
    available: bool = True

    def __post_init__(self):
        self.model = SequenceModel(self.outputs)

    def health_check(self):
        return self.available

    def get_model(self):
        return self.model

    def unavailable_message(self):
        return "Local model unavailable."


class FailedSolver:
    def solve(self, text):
        return MathResult(None, False, "test", "verification failed", None, [], ["test"], 0.0)


def test_ollama_available_valid_structured_output():
    result = ReasoningController().solve("solve this", FakeManager())

    assert result.status == "solved"
    assert result.verified is True
    assert result.answer == 7


def test_invalid_output_triggers_repair_attempt():
    manager = FakeManager(outputs=["not json", VALID])
    result = ReasoningController().solve("solve this", manager)

    assert result.status == "solved"
    assert manager.model.calls == 2


def test_repair_failure_returns_formalization_failed():
    manager = FakeManager(outputs=["bad", "bad", "bad"])
    result = ReasoningController().solve("solve this", manager)

    assert result.status == "formalization_failed"
    assert manager.model.calls == 3


def test_solver_failure_is_reported_without_success_fallback():
    result = ReasoningController(solver=FailedSolver()).solve_structured(VALID)

    assert result.status == "verification_failed"
    assert result.verified is False


def test_timeout_is_reported_as_unavailable():
    class TimeoutModel(SequenceModel):
        def structured_output(self, prompt, **kwargs):
            raise TimeoutError("timed out")

    manager = FakeManager()
    manager.model = TimeoutModel([])
    result = ReasoningController().solve("solve this", manager)

    assert result.status == "llm_unavailable"
    assert "timed out" in result.reason


def test_model_unavailable_is_reported():
    result = ReasoningController().solve("solve this", FakeManager(available=False))

    assert result.status == "llm_unavailable"
