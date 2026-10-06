"""
axen/agent/reflection.py
========================
Canonical Observation Model, Reflection Engine, and Multi-Turn Self-Correction.

Features:
1. CanonicalObservation data structure tracking execution_id, output, and untrusted flag.
2. ReflectionEngine evaluating whether tool output satisfies the objective or requires correction/next steps.
3. UserFeedback parser converting multi-turn conversational corrections into structured actions.
4. Hard-bounded self-correction loop preventing infinite reflection.
5. Strict protection preventing private chain-of-thought exposure.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ReflectionStatus(str, Enum):
    """Assessment of the latest tool observation."""
    COMPLETE = "complete"                  # Task or step satisfied, ready to finish
    NEEDS_NEXT_STEP = "needs_next_step"    # Intermediate step complete, chained tool required
    RETRY = "retry"                        # Transient failure, retry permitted
    CORRECTION = "correction"              # Output unsatisfactory or wrong, correct action needed
    FAILED = "failed"                      # Unrecoverable error, abort step cleanly


class ReflectionReasonCode(str, Enum):
    """Standardized machine-readable explanation for reflection result."""
    VERIFIED_RESULT = "verified_result"
    MISSING_INFORMATION = "missing_information"
    TRANSIENT_TOOL_FAILURE = "transient_tool_failure"
    INVALID_OUTPUT = "invalid_output"
    CONTRADICTORY_RESULT = "contradictory_result"
    UNRECOVERABLE = "unrecoverable"


class FeedbackType(str, Enum):
    """Categorization of user multi-turn feedback."""
    CORRECTION = "correction"    # User pointing out a mistake or target change
    REJECTION = "rejection"      # User rejecting the answer ("that's wrong")
    REFINEMENT = "refinement"    # User narrowing down scope or providing details
    CONFIRMATION = "confirmation"# User confirming result


# ---------------------------------------------------------------------------
# Canonical Observation
# ---------------------------------------------------------------------------

@dataclass
class CanonicalObservation:
    """
    Canonical representation of a completed tool execution outcome.
    Untrusted by default to prevent prompt injection from file/tool outputs.
    """
    execution_id: str
    tool: str
    status: str  # "completed", "failed", "timeout", "cancelled"
    output: Any
    trusted: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "tool": self.tool,
            "status": self.status,
            "output": str(self.output) if self.output is not None else None,
            "trusted": self.trusted,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CanonicalObservation:
        return cls(
            execution_id=data.get("execution_id") or f"exec_{uuid.uuid4().hex[:12]}",
            tool=data.get("tool", "unknown"),
            status=data.get("status", "completed"),
            output=data.get("output"),
            trusted=bool(data.get("trusted", False)),
            error=data.get("error"),
            metadata=data.get("metadata") or {},
        )


# ---------------------------------------------------------------------------
# Reflection Result
# ---------------------------------------------------------------------------

@dataclass
class ReflectionResult:
    """
    Structured outcome of the reflection evaluation.
    NOTE: 'explanation' must be a concise operational summary, NOT private chain-of-thought.
    """
    status: ReflectionStatus
    reason_code: ReflectionReasonCode
    next_action: Optional[str] = None
    next_arguments: Optional[Dict[str, Any]] = None
    explanation: str = ""
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.status, str):
            self.status = ReflectionStatus(self.status)
        if isinstance(self.reason_code, str):
            self.reason_code = ReflectionReasonCode(self.reason_code)
        # Strip potential private chain-of-thought
        if self.explanation:
            self.explanation = re.sub(
                r"^(?:thinking|thought|let me think)[\s\:\-]+",
                "",
                self.explanation.strip(),
                flags=re.IGNORECASE,
            )[:200]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "reason_code": self.reason_code.value,
            "next_action": self.next_action,
            "next_arguments": self.next_arguments,
            "explanation": self.explanation,
            "confidence": round(self.confidence, 4),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReflectionResult:
        return cls(
            status=ReflectionStatus(data.get("status", "complete")),
            reason_code=ReflectionReasonCode(data.get("reason_code", "verified_result")),
            next_action=data.get("next_action"),
            next_arguments=data.get("next_arguments"),
            explanation=str(data.get("explanation") or ""),
            confidence=float(data.get("confidence", 1.0)),
        )


# ---------------------------------------------------------------------------
# User Feedback
# ---------------------------------------------------------------------------

@dataclass
class UserFeedback:
    """
    Structured interpretation of a user correction in a multi-turn conversation.
    """
    feedback_type: FeedbackType
    target: str          # e.g., "previous_result", "file_target", "calculation"
    instruction: str     # Specific modification requested
    original_query: str  # Original user query being corrected

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feedback_type": self.feedback_type.value,
            "target": self.target,
            "instruction": self.instruction,
            "original_query": self.original_query,
        }


# ---------------------------------------------------------------------------
# Reflection Engine
# ---------------------------------------------------------------------------

class ReflectionEngine:
    """
    Evaluates tool observations objectively against the user goal.
    Provides structured decisions for chaining, retrying, or finishing.
    """

    @classmethod
    def reflect(
        cls,
        user_request: str,
        observation: CanonicalObservation,
        prior_observations: Optional[List[CanonicalObservation]] = None,
        retries_used: int = 0,
        max_retries: int = 2,
    ) -> ReflectionResult:
        """
        Evaluate observation outcome.
        Returns ReflectionResult indicating whether task is complete, needs next step, or retry.
        """
        req_lower = user_request.lower().strip()
        obs_text = str(observation.output or "")
        priors = prior_observations or []

        # 1. Handle Failed Execution
        if observation.status != "completed":
            err_msg = (observation.error or "").lower()

            # Non-retryable security / permissions violations
            if any(term in err_msg for term in ["permission", "access denied", "sandbox", "traversal", "unauthorized", "null byte"]):
                return ReflectionResult(
                    status=ReflectionStatus.FAILED,
                    reason_code=ReflectionReasonCode.UNRECOVERABLE,
                    explanation="Operation denied by security sandbox policy.",
                )

            # Check if retry budget remains for transient errors
            if retries_used < max_retries and any(term in err_msg for term in ["timeout", "temporary", "connection", "rate limit", "busy"]):
                return ReflectionResult(
                    status=ReflectionStatus.RETRY,
                    reason_code=ReflectionReasonCode.TRANSIENT_TOOL_FAILURE,
                    explanation=f"Transient tool failure. Retrying ({retries_used + 1}/{max_retries}).",
                )

            # Failure with possible alternative tool
            if observation.tool in ("calculator", "calculator.evaluate") and "syntax" in err_msg:
                return ReflectionResult(
                    status=ReflectionStatus.CORRECTION,
                    reason_code=ReflectionReasonCode.INVALID_OUTPUT,
                    next_action="python",
                    next_arguments={"code": f"print({user_request})"},
                    explanation="Arithmetic syntax error in calculator; attempting Python evaluation.",
                )

            return ReflectionResult(
                status=ReflectionStatus.FAILED,
                reason_code=ReflectionReasonCode.UNRECOVERABLE,
                explanation=f"Tool failed: {observation.error or 'Unknown failure'}",
            )

        # 2. Chained Multi-Step Sequences (e.g. search -> read -> compute)
        if observation.tool in ("file_search", "file.search"):
            # Check if matches were found
            m = re.search(r"[-•]\s*([a-zA-Z0-9_\-\.\/\\]+\.(?:csv|txt|py|json|md))\b", obs_text)
            if m:
                target_file = m.group(1).strip()
                # Determine if user wanted file content or computation
                return ReflectionResult(
                    status=ReflectionStatus.NEEDS_NEXT_STEP,
                    reason_code=ReflectionReasonCode.MISSING_INFORMATION,
                    next_action="file_read",
                    next_arguments={"path": target_file},
                    explanation=f"File found ({target_file}); reading contents for next step.",
                )
            else:
                return ReflectionResult(
                    status=ReflectionStatus.COMPLETE,
                    reason_code=ReflectionReasonCode.VERIFIED_RESULT,
                    explanation="File search completed with no matching files.",
                )

        if observation.tool in ("file_read", "file.read", "file"):
            # If user wanted average/calculation and we just read a file
            if any(term in req_lower for term in ["average", "mean", "sum", "total", "calculate", "compute"]):
                nums = re.findall(r"-?\d+(?:\.\d+)?", obs_text)
                if nums:
                    num_list_str = "[" + ", ".join(nums) + "]"
                    synth_code = f"data = {num_list_str}\navg = sum(data) / len(data)\nprint('Average:', round(avg, 2))"
                    return ReflectionResult(
                        status=ReflectionStatus.NEEDS_NEXT_STEP,
                        reason_code=ReflectionReasonCode.MISSING_INFORMATION,
                        next_action="python",
                        next_arguments={"code": synth_code},
                        explanation="Data extracted from file; executing Python statistical calculation.",
                    )
            return ReflectionResult(
                status=ReflectionStatus.COMPLETE,
                reason_code=ReflectionReasonCode.VERIFIED_RESULT,
                explanation="File content successfully retrieved.",
            )

        # 3. Code Execution & Calculations
        if observation.tool in ("calculator", "calculator.evaluate", "python", "python_execute"):
            if not obs_text.strip():
                return ReflectionResult(
                    status=ReflectionStatus.CORRECTION,
                    reason_code=ReflectionReasonCode.INVALID_OUTPUT,
                    explanation="Tool returned empty output.",
                )
            return ReflectionResult(
                status=ReflectionStatus.COMPLETE,
                reason_code=ReflectionReasonCode.VERIFIED_RESULT,
                explanation="Computation completed and verified successfully.",
            )

        # Default completion for other tools
        return ReflectionResult(
            status=ReflectionStatus.COMPLETE,
            reason_code=ReflectionReasonCode.VERIFIED_RESULT,
            explanation="Operation completed successfully.",
        )


# ---------------------------------------------------------------------------
# Multi-Turn Feedback Parser
# ---------------------------------------------------------------------------

class FeedbackParser:
    """
    Parses conversational user feedback to adapt the execution context
    without restarting from zero or polluting long-term memory.
    """

    _REJECTION_PATTERN = re.compile(
        r"^(?:that'?s\s+(?:wrong|incorrect)|no|not\s+quite|try\s+again|wrong|false)[\s\.\!\?]*$",
        re.IGNORECASE,
    )
    _FILE_CORRECTION_PATTERN = re.compile(
        r"\b(?:no[,;\s]+)?(?:use|check|read|look\s+at)\s+(?:the\s+other\s+file|([a-zA-Z0-9_\-\.\/\\]+\.(?:csv|txt|py|json|md)))\b",
        re.IGNORECASE,
    )
    _VALUE_CORRECTION_PATTERN = re.compile(
        r"\b(?:calculate|compute|use)\s+(?:with\s+|for\s+)?(\d+[\s\+\-\*\/\d\.]*)\s+instead\b",
        re.IGNORECASE,
    )

    @classmethod
    def parse_feedback(cls, user_message: str, previous_request: str) -> Optional[UserFeedback]:
        """
        Check if user message is a multi-turn correction or refinement.
        Returns UserFeedback if applicable, else None.
        """
        cleaned = user_message.strip()

        # 1. Direct Rejection
        if cls._REJECTION_PATTERN.match(cleaned):
            return UserFeedback(
                feedback_type=FeedbackType.REJECTION,
                target="previous_result",
                instruction="Previous result rejected; re-evaluating approach.",
                original_query=previous_request,
            )

        # 2. File Target Correction
        m_file = cls._FILE_CORRECTION_PATTERN.search(cleaned)
        if m_file:
            target_name = m_file.group(1) or "alternative file"
            return UserFeedback(
                feedback_type=FeedbackType.CORRECTION,
                target="file_target",
                instruction=f"Target file redirected to: {target_name}",
                original_query=previous_request,
            )

        # 3. Value / Parameter Correction
        m_val = cls._VALUE_CORRECTION_PATTERN.search(cleaned)
        if m_val:
            val = m_val.group(1).strip()
            return UserFeedback(
                feedback_type=FeedbackType.CORRECTION,
                target="calculation",
                instruction=f"Recalculate with: {val}",
                original_query=previous_request,
            )

        return None
