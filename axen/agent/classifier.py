"""
axen/agent/classifier.py
========================
Task classification for AXEN Phase 3 Agentic Loop.

Classifies incoming user requests into standard task types:
- conversation: greetings, open banter, identity questions
- calculation: arithmetic expressions (e.g., 25 * 48, 12345 * 678)
- math: symbolic mathematics, calculus, number theory (e.g., derivatives, integrals, gcd)
- programming: coding explanations, conceptual discussions (e.g., explain recursion)
- code_execution: requests to execute Python code (e.g., run this python code)
- file_operation: reading, viewing, or checking local files (e.g., read this file)
- memory: saving or recalling user notes (e.g., remember that..., recall notes)
- knowledge_retrieval: factual informational lookup (e.g., what is Python?)
- mixed: multi-step tasks requiring multiple systems
- unknown: unclassified
"""

from __future__ import annotations

import re
from typing import Any, Optional


class TaskClassifier:
    """Deterministic and LLM-assisted request classifier."""

    # Pre-compiled patterns for deterministic categorization
    _GREETINGS = re.compile(
        r"^(hello|hi|hey|greetings|good\s+(morning|afternoon|evening|day)|howdy|sup)[\s\.\!\?]*$",
        re.IGNORECASE,
    )
    _IDENTITY = re.compile(
        r"\b(?:who\s+are\s+you|what\s+is\s+your\s+name|what\s+company|your\s+hobbies|do\s+you\s+have\s+hobbies)\b",
        re.IGNORECASE,
    )
    _CALCULATION = re.compile(
        r"\b(calculate|compute|what\s+is|eval)\b.*?[\d\+\-\*\/\(\)\^\.]+|"
        r"\b[\d\.]+\s*(?:multiplied\s+by|times|divided\s+by|plus|minus|\*|\+|\/|\-|\*\*)\s*[\d\.]+|"
        r"^\s*[\d\+\-\*\/\(\)\^\.% \t]+\s*$",
        re.IGNORECASE | re.DOTALL,
    )
    _SYMBOLIC_MATH = re.compile(
        r"\b(derivative\s+of|differentiate|integral\s+of|integrate|limit\s+of|gcd\s+of|greatest\s+common\s+divisor|is\s+\d+\s+prime|prime\?|euler\s+phi|modular\s+inverse)\b|"
        r"^\s*solve\s+[a-z0-9\^\+\-\*\/\s=]+$",
        re.IGNORECASE,
    )
    _CODE_EXECUTION = re.compile(
        r"\b(?:run|execute)\s*(?::\s*)?(?:python|this python|code|this code)?\b.*?\bprint\b|"
        r"\b(?:run|execute)\s+(?:this\s+)?python\b|"
        r"\b(?:run|execute)\s+(?:this\s+)?code\b|"
        r"```python|"
        r"\bprint\s*\(.*?\)",
        re.IGNORECASE | re.DOTALL,
    )
    _FILE_OPERATION = re.compile(
        r"\b(?:read|show|view|open|cat)\s+(?:this\s+)?file[:\s]+[^\s,;]+",
        re.IGNORECASE,
    )
    _MEMORY_OP = re.compile(
        r"\b(remember\s+that|please\s+note|save\s+that|note\s+that|keep\s+in\s+mind|recall|do\s+you\s+remember)\b",
        re.IGNORECASE,
    )
    _PROGRAMMING_CONCEPT = re.compile(
        r"\b(?:explain|how\s+does|what\s+is)\b.*\b(?:recursion|algorithm|function|class|decorator|generator|pointer|async|promise|variable)\b|"
        r"\b(?:explain\s+python|python\s+recursion)\b",
        re.IGNORECASE,
    )
    _MIXED_TASK = re.compile(
        r"\b(?:and\s+then|after\s+that|first\s+read.*then\s+calculate|write\s+.*and\s+(?:run|test)\s+it|read\s+.*and\s+(?:calculate|explain|compute))\b",
        re.IGNORECASE,
    )

    def classify(self, text: str, model: Optional[Any] = None) -> tuple[str, bool, Optional[str]]:
        """
        Classify text and determine tool necessity.

        Returns:
            (task_type, tool_required, selected_tool)
        """
        cleaned = text.strip()
        if not cleaned:
            return "unknown", False, None

        lower = cleaned.lower()

        # 1. Multi-step / mixed task detection
        if self._MIXED_TASK.search(cleaned):
            return "mixed", True, "multi_step"

        # 2. Memory operation
        if self._MEMORY_OP.search(cleaned):
            return "memory", True, "memory"

        # 3. File operation
        if self._FILE_OPERATION.search(cleaned):
            return "file_operation", True, "file"

        # 4. Code execution
        if self._CODE_EXECUTION.search(cleaned):
            return "code_execution", True, "python"

        # 5. Symbolic mathematics (MathSolver)
        if self._SYMBOLIC_MATH.search(cleaned):
            return "math", True, "math_solver"

        # 6. Arithmetic calculation
        if self._CALCULATION.search(cleaned):
            if re.search(r"\d", cleaned) and re.search(r"[+\-*/^%]|multiplied|times|divided|plus|minus", lower):
                return "calculation", True, "calculator"

        # 7. Greetings / identity -> conversation (NO tool)
        if self._GREETINGS.match(cleaned) or self._IDENTITY.search(cleaned):
            return "conversation", False, None

        # 8. Programming concept explanation -> programming (NO tool)
        if self._PROGRAMMING_CONCEPT.search(cleaned):
            return "programming", False, None

        # 9. Knowledge questions (e.g., "What is Python?") -> knowledge_retrieval (NO tool)
        if re.search(r"^(what|who|when|where|why|how)\s+(?:is|are|was|were)\b", lower):
            return "knowledge_retrieval", False, None

        # 10. Fallback for random noise / unclassified (no alphanumeric content)
        if not re.search(r"[a-zA-Z0-9]", cleaned):
            return "unknown", False, None

        return "conversation", False, None
