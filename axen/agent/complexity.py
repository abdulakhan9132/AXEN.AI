"""
axen/agent/complexity.py
========================
Complexity Estimator for AXEN Intelligence Core / Brain.

Estimates the cognitive and operational complexity tier:
- LOW: Direct responses, single deterministic tool execution (2+2, 25*48, 255 to binary, greetings).
- MEDIUM: Single symbolic derivation, conceptual explanation, single code generation, bounded translation.
- HIGH: Multi-step workflows, multi-domain mixed tasks (file + math + explanation), deep debugging, multi-constraint planning.
"""

from __future__ import annotations

import re
from typing import Any, Optional


class Complexity(str):
    """
    Case-insensitive, backward-compatible complexity descriptor.
    Allows transparent comparison:
      Complexity("LOW") == "LOW" -> True
      Complexity("LOW") == "simple" -> True
      Complexity("MEDIUM") == "moderate" -> True
      Complexity("HIGH") == "complex" -> True
      Complexity("HIGH") == "multi_step" -> True
    """

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, str):
            return False
        s_self = str(self).upper()
        s_other = str(other).upper()
        if s_self == s_other:
            return True
        mapping = {
            "LOW": {"LOW", "SIMPLE"},
            "SIMPLE": {"LOW", "SIMPLE"},
            "MEDIUM": {"MEDIUM", "MODERATE"},
            "MODERATE": {"MEDIUM", "MODERATE"},
            "HIGH": {"HIGH", "COMPLEX", "MULTI_STEP"},
            "COMPLEX": {"HIGH", "COMPLEX", "MULTI_STEP"},
            "MULTI_STEP": {"HIGH", "COMPLEX", "MULTI_STEP"},
        }
        return s_other in mapping.get(s_self, {s_self})

    def __hash__(self) -> int:
        return hash(str(self).upper())


class ComplexityEstimator:
    """Lightweight heuristic complexity estimator for orchestration decisions."""

    LOW = Complexity("LOW")
    MEDIUM = Complexity("MEDIUM")
    HIGH = Complexity("HIGH")

    @classmethod
    def estimate(
        cls,
        text: str,
        intent: str = "conversation",
        domain: str = "general",
        requires_multiple_steps: bool = False,
        tools_needed: Optional[list[str]] = None,
        has_context: bool = False,
    ) -> Complexity:
        """
        Estimate task complexity tier (LOW, MEDIUM, HIGH) without invoking an LLM.
        """
        clean = text.strip()
        lower = clean.lower()
        tools = tools_needed or []

        # 1. Multi-step or mixed tasks with multiple tools are automatically HIGH
        if requires_multiple_steps or len(tools) > 1 or intent in ("multi_step", "mixed"):
            return cls.HIGH

        # 2. Multi-domain indicator keywords
        multi_domain_triggers = [
            r"\bread\b.*\band\b.*\b(?:calculate|compute|analyze|explain)\b",
            r"\b(?:calculate|compute)\b.*\band\b.*\bexplain\b",
            r"\bcompare\b.*\b(?:three|multiple|different|algorithms|frameworks)\b",
            r"\b(?:analyze|statistics)\b.*\band\b.*\bexplain\b",
            r"\bdebug\b.*\band\b.*\bfix\b",
        ]
        for trigger in multi_domain_triggers:
            if re.search(trigger, lower, re.I):
                return cls.HIGH

        # 3. Simple greetings and identity -> LOW
        if intent in ("conversation", "greeting") or re.match(r"^(?:hi|hello|hey|good\s+morning|who\s+are\s+you)\b", lower):
            return cls.LOW

        # 4. Pure arithmetic calculation -> LOW
        if intent == "calculation" or (tools == ["calculator"]):
            return cls.LOW

        # 5. Pure computer systems conversions / bitwise -> LOW
        if intent == "computer_systems" or (tools == ["computer"]):
            if any(term in lower for term in ["convert", "binary", "hex", "bitwise", "ascii", "two's complement"]):
                # Simple conversion vs multi-stage architecture explanation
                if not any(term in lower for term in ["explain in detail", "compare", "pipeline"]):
                    return cls.LOW

        # 6. Single file read without heavy analysis -> LOW
        if intent == "file_operation" and not any(term in lower for term in ["analyze", "calculate", "find all"]):
            return cls.LOW

        # 7. Symbolic math (differentiation, integration, simplification) -> MEDIUM
        if intent == "math" or domain in ("calculus", "algebra", "symbolic_math"):
            if any(conj in lower for conj in ["and then", "after that", "and simplify", "and explain"]):
                return cls.HIGH
            return cls.MEDIUM

        # 8. Single coding task or concept explanation -> MEDIUM
        if intent in ("programming", "code_execution", "writing", "translation"):
            if len(clean.split()) > 40 or "and then" in lower:
                return cls.HIGH
            return cls.MEDIUM

        # 9. Planning and reasoning
        if intent in ("planning", "reasoning", "debugging"):
            return cls.HIGH if any(w in lower for w in ["detailed", "step-by-step", "compare", "multi-day", "budget"]) else cls.MEDIUM

        # Default fallback: MEDIUM if moderate length, otherwise LOW
        if len(clean.split()) > 25:
            return cls.MEDIUM
        return cls.LOW
