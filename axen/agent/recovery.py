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


import re

class FailureType(str, Enum):
    VALIDATION_ERROR = "validation_error"
    PERMISSION_ERROR = "permission_error"
    TIMEOUT = "timeout"
    CANCELLATION = "cancellation"
    TOOL_FAILURE = "tool_failure"
    PROVIDER_FAILURE = "provider_failure"
    MALFORMED_MODEL_OUTPUT = "malformed_model_output"
    RESOURCE_LIMIT = "resource_limit"
    CODE_SYNTAX_ERROR = "code_syntax_error"
    CODE_RUNTIME_ERROR = "code_runtime_error"
    UNKNOWN_ERROR = "unknown_error"


class RecoveryManager:
    """Manages failure classification and retry decisions."""

    def __init__(self, max_retries_per_tool: int = 2):
        self.max_retries_per_tool = max_retries_per_tool

    @classmethod
    def extract_code_diagnostic(cls, error: str) -> Dict[str, Any]:
        """Deterministically extract structured diagnostic info from Python errors/tracebacks."""
        diagnostic: Dict[str, Any] = {
            "exception_type": "UnknownException",
            "line_number": None,
            "column": None,
            "message": error.strip(),
            "repair_guidance": "",
        }
        if not error:
            return diagnostic

        # Line number matching: line 42 or Line 42
        line_match = re.search(r"(?:line|Line)\s+(\d+)", error)
        if line_match:
            diagnostic["line_number"] = int(line_match.group(1))

        # Exception type matching
        exc_match = re.search(r"(?:E\s+)?([A-Za-z_]\w*(?:Error|Exception|Warning))(?::\s*(.*))?", error)
        if exc_match:
            diagnostic["exception_type"] = exc_match.group(1)
            if exc_match.group(2):
                diagnostic["message"] = exc_match.group(2).strip()

        exc_type = diagnostic["exception_type"]
        if exc_type in ("SyntaxError", "IndentationError"):
            diagnostic["repair_guidance"] = (
                f"Fix syntax at line {diagnostic['line_number'] or 'unspecified'}: verify balanced delimiters, indentation, and trailing colons."
            )
        elif exc_type == "NameError":
            diagnostic["repair_guidance"] = f"Define missing identifier or add necessary import: {diagnostic['message']}."
        elif exc_type == "TypeError":
            diagnostic["repair_guidance"] = f"Correct function argument or type mismatch: {diagnostic['message']}."
        elif exc_type == "ZeroDivisionError":
            diagnostic["repair_guidance"] = "Add zero divisor check before division operation."
        else:
            diagnostic["repair_guidance"] = f"Investigate and correct {exc_type}: {diagnostic['message']}."

        return diagnostic

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

        # Specific code syntax checks (higher precedence than generic tool error)
        if any(w in err_lower for w in ("syntaxerror", "syntax error", "indentationerror", "indentation error", "unbalanced delimiter", "unclosed delimiter")):
            return FailureType.CODE_SYNTAX_ERROR

        # Specific code runtime checks
        if any(w in err_lower for w in ("nameerror", "typeerror", "attributeerror", "zerodivisionerror", "indexerror", "keyerror", "unboundlocalerror")):
            return FailureType.CODE_RUNTIME_ERROR

        if any(w in err_lower for w in ("schema", "validation", "invalid argument", "missing required", "type")):
            return FailureType.VALIDATION_ERROR

        if any(w in err_lower for w in ("malformed", "jsondecodeerror", "invalid format", "parse error")):
            return FailureType.MALFORMED_MODEL_OUTPUT

        if any(w in err_lower for w in ("rate limit", "oversized", "quota", "resource limit", "memory limit", "budget")):
            return FailureType.RESOURCE_LIMIT

        if any(w in err_lower for w in ("connection", "502", "503", "provider", "unavailable", "server error")):
            return FailureType.PROVIDER_FAILURE

        if any(w in err_lower for w in ("zero division", "file not found", "error")):
            return FailureType.TOOL_FAILURE

        return FailureType.UNKNOWN_ERROR

    @classmethod
    def is_retryable(cls, failure_type: FailureType, attempts: int = 0, max_retries: int = 2) -> bool:
        """
        Determine if the failure is safe to retry.
        Rules:
        - Permission errors, cancellations, and validation errors are NEVER retryable without input change.
        - Tool failures like division by zero or file missing are not retryable without re-planning.
        - Code syntax and runtime errors are retryable via guided repair up to max_retries.
        - Timeouts or provider glitches can be retried up to max_retries.
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
            FailureType.CODE_SYNTAX_ERROR,
            FailureType.CODE_RUNTIME_ERROR,
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
        retryable = self.is_retryable(ft, attempts, max_retries=self.max_retries_per_tool)

        action: Dict[str, Any] = {
            "failure_type": ft.value,
            "retryable": retryable,
            "attempts": attempts,
            "error": error,
            "recommended_strategy": "retry" if retryable else "fallback",
        }

        # Handle code-specific repair strategy
        if ft in (FailureType.CODE_SYNTAX_ERROR, FailureType.CODE_RUNTIME_ERROR):
            diag = self.extract_code_diagnostic(error)
            action["diagnostic"] = diag
            action["recommended_strategy"] = "code_repair"
            action["repair_prompt"] = diag.get("repair_guidance", "")
            return action

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
