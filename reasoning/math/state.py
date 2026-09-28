from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .representations import MathematicalProblem


@dataclass
class ReasoningState:
    problem: MathematicalProblem
    steps: list[str] = field(default_factory=list)
    intermediate_results: list[Any] = field(default_factory=list)
    strategies_tried: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    verification: str | None = None
    answer: Any = None
    verified: bool = False

    def record(self, step: str, value: Any = None) -> None:
        self.steps.append(step)
        if value is not None:
            self.intermediate_results.append(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem": self.problem.text,
            "type": self.problem.categories,
            "steps": self.steps,
            "intermediate_results": self.intermediate_results,
            "strategies_tried": self.strategies_tried,
            "assumptions": self.assumptions,
            "verification": self.verification,
            "answer": self.answer,
            "verified": self.verified,
        }
