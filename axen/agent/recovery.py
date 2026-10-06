"""
axen/agent/recovery.py
======================
Controlled Failure Recovery for AXEN Phase 6 Multi-Step Agent Loop.

Classifies failure types:
- validation error
- permission error
- timeout
- cancellation
- tool failure
- provider failure
- malformed model output
- resource limit
- unknown error

Enforces:
- Deterministic classification
- Safe retry decisions (only retry retryable errors)
- Alternative fallback strategies
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional


class FailureType(str, Enum):
    VALIDATION_ERROR = "validation_error"
    PERMISSION_ERROR = "permission_error"
    TIMEOUT = "timeout"
    CANCELLATION = "cancellation"
    TOOL_FAILURE = "tool_failure"
    PROVIDER_FAILURE = "provider_failure"
    MALFORMED_MODEL_OUTPUT = "malformed_model_output"
    RESOURCE_LIMIT = "resource_limit"
    UNKNOWN_ERROR = "unknown_error"


class RecoveryManager:
    """Manages failure classification and retry decisions."""

    def __init__(self, max_retries_per_tool: int = 2):
        self.max_retries_per_tool = max_retries_per_tool

    def classify_failure(self, error: Optional[str], state_hint: Optional[str] = None) -> FailureType:
        """Deterministically categorize the failure type from error text or state hint."""
        if state_hint:
            sh = state_hint.lower()
            if "cancel" in sh:
                return FailureType.CANCELLATION
            if "timeout" in sh:
                return FailureType.TIMEOUT

        if not error:
            return FailureType.UNKNOWN_ERROR

        err_lower = error.lower()

        if "cancel" in err_lower or "abort" in err_lower:
            return FailureType.CANCELLATION

        if "timeout" in err_lower or "timed out" in err_lower:
            return FailureType.TIMEOUT

        if any(w in err_lower for w in ("permission", "denied", "forbidden", "unauthorized", "cross-project")):
            return FailureType.PERMISSION_ERROR

        if any(w in err_lower for w in ("schema", "validation", "invalid argument", "missing required", "type")):
            return FailureType.VALIDATION_ERROR

        if any(w in err_lower for w in ("malformed", "jsondecodeerror", "invalid format", "parse error")):
            return FailureType.MALFORMED_MODEL_OUTPUT

        if any(w in err_lower for w in ("rate limit", "oversized", "quota", "resource limit", "memory limit", "budget")):
            return FailureType.RESOURCE_LIMIT

        if any(w in err_lower for w in ("connection", "502", "503", "provider", "unavailable", "server error")):
            return FailureType.PROVIDER_FAILURE

        if any(w in err_lower for w in ("zero division", "syntax error", "file not found", "error")):
            return FailureType.TOOL_FAILURE

        return FailureType.UNKNOWN_ERROR

    @classmethod
    def is_retryable(cls, failure_type: FailureType, attempts: int = 0, max_retries: int = 2) -> bool:
        """
        Determine if the failure is safe to retry.
        Rules:
        - Permission errors, cancellations, and validation errors are NEVER retryable without input change.
        - Tool failures like division by zero or fatal syntax errors are not retryable.
        - Timeouts or provider glitches can be retried up to max_retries_per_tool.
        """
        if attempts >= max_retries:
            return False

        if failure_type in (
            FailureType.CANCELLATION,
            FailureType.PERMISSION_ERROR,
            FailureType.VALIDATION_ERROR,
            FailureType.RESOURCE_LIMIT,
        ):
            return False

        if failure_type in (
            FailureType.TIMEOUT,
            FailureType.PROVIDER_FAILURE,
            FailureType.MALFORMED_MODEL_OUTPUT,
        ):
            return True

        if failure_type == FailureType.TOOL_FAILURE:
            # Only retry if not an intrinsic logic error
            return False

        return False

    def get_recovery_action(
        self,
        tool_name: str,
        error: str,
        attempts: int,
        state_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Produce a structured recovery action recommendation."""
        ft = self.classify_failure(error, state_hint=state_hint)
        retryable = self.is_retryable(ft, attempts)

        action = {
            "failure_type": ft.value,
            "retryable": retryable,
            "attempts": attempts,
            "error": error,
            "recommended_strategy": "retry" if retryable else "fallback",
        }

        # Select alternative strategy if not retryable
        if not retryable:
            if tool_name == "calculator":
                action["alternative_tool"] = "python"
            elif tool_name == "math_solver":
                action["alternative_tool"] = "calculator"
            elif tool_name == "file_search":
                action["alternative_tool"] = "file_list"
            else:
                action["alternative_tool"] = None

        return action
