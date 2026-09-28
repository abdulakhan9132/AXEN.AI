from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MathematicalProblem:
    text: str
    categories: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    target: str | None = None
    operation: str | None = None
    assumptions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem": self.text,
            "type": self.categories,
            "constraints": self.constraints,
            "target": self.target,
            "operation": self.operation,
            "assumptions": self.assumptions,
            "metadata": self.metadata,
        }


@dataclass
class VerificationRecord:
    verified: bool
    detail: str


@dataclass
class MathResult:
    answer: Any
    verified: bool
    method: str
    verification: str
    problem: MathematicalProblem | None = None
    steps: list[str] = field(default_factory=list)
    attempts: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "verified": self.verified,
            "method": self.method,
            "verification": self.verification,
            "problem": self.problem.to_dict() if self.problem else None,
            "steps": self.steps,
            "attempts": self.attempts,
            "elapsed_seconds": self.elapsed_seconds,
        }
