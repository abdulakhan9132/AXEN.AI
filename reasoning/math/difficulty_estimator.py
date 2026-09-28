from __future__ import annotations

import re

from .representations import MathematicalProblem


def estimate_difficulty(problem: MathematicalProblem | str) -> tuple[int, str]:
    text = problem.text if isinstance(problem, MathematicalProblem) else problem
    normalized = text.lower()
    score = 1
    reasons: list[str] = []
    if len(text) > 120 or len(re.findall(r"[,;]", text)) >= 2:
        score += 1; reasons.append("multi-clause statement")
    advanced_terms = ("differential", "eigen", "optimization", "proof", "recurrence", "conditional", "modular")
    if any(term in normalized for term in advanced_terms):
        score += 1; reasons.append("advanced domain terminology")
    mixed_problem = normalized.count("and") >= 2
    if isinstance(problem, MathematicalProblem):
        mixed_problem = mixed_problem or len(problem.categories) >= 2
    if mixed_problem:
        score += 1; reasons.append("mixed constraints or domains")
    if any(term in normalized for term in ("olympiad", "competition", "prove that", "for all")):
        score += 1; reasons.append("proof-oriented wording")
    score = min(5, score)
    return score, "; ".join(reasons) or "short single-operation problem"
