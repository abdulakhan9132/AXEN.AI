"""
axen/agent/intent.py
====================
Structured Intent definition for AXEN General-Purpose AI Assistant Brain.

Defines the intent classification schema, complexity levels, tool requirements,
and structured intent metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .complexity import Complexity


@dataclass
class IntentInfo:
    """
    Structured representation of user intent and execution requirements.

    Example schema:
    {
        "intent": "math",
        "domain": "calculus",
        "complexity": "medium",
        "requires_tool": true,
        "requires_multiple_steps": false,
        "requires_memory": false,
        "requires_clarification": false,
        "confidence": 0.94
    }
    """
    intent: str
    domain: str = "general"
    requires_reasoning: bool = False
    requires_tool: bool = False
    requires_multiple_steps: bool = False
    requires_memory: bool = False
    requires_verification: bool = False
    complexity: Any = "simple"  # Supports Complexity("LOW") / "simple" / "medium" / "high"
    tools_needed: list[str] = field(default_factory=list)
    primary_tool: Optional[str] = None
    needs_clarification: bool = False
    requires_clarification: bool = False
    clarification_prompt: Optional[str] = None
    confidence: float = 1.0
    context_reference: Optional[str] = None
    extracted_data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.needs_clarification and not self.requires_clarification:
            self.requires_clarification = True
        elif self.requires_clarification and not self.needs_clarification:
            self.needs_clarification = True

        if not isinstance(self.complexity, Complexity):
            self.complexity = Complexity(str(self.complexity))

    def to_dict(self) -> dict[str, Any]:
        """Convert to standard JSON-serializable dictionary."""
        return {
            "intent": self.intent,
            "domain": self.domain,
            "requires_reasoning": self.requires_reasoning,
            "requires_tool": self.requires_tool,
            "requires_multiple_steps": self.requires_multiple_steps,
            "requires_memory": self.requires_memory,
            "requires_verification": self.requires_verification,
            "complexity": self.complexity,
            "tools_needed": self.tools_needed,
            "primary_tool": self.primary_tool,
            "needs_clarification": self.needs_clarification,
            "requires_clarification": self.requires_clarification,
            "clarification_prompt": self.clarification_prompt,
            "confidence": round(self.confidence, 2),
            "context_reference": self.context_reference,
            "extracted_data": self.extracted_data,
        }
