"""
axen/agent/think_metrics.py
============================
Think Engine 2.0 — Observability, Structured Metrics, and Telemetry.

Tracks per-request:
- Latency (total, per-phase)
- Token consumption (input, output)
- Model calls count
- Tool calls count and results
- Retries and recovery attempts
- Verification outcomes
- Think mode used
- Re-plan count
- Budget utilization

Provides:
- Structured log entries (with sensitive field redaction)
- User-facing progress summaries (no hidden reasoning exposed)
- JSON-serializable metrics for storage/analysis
"""

from __future__ import annotations

import datetime as dt
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .think_config import ThinkMode

logger = logging.getLogger("axen.agent.think_metrics")

# Fields to always redact from structured logs
_DEFAULT_REDACT_FIELDS = frozenset({"api_key", "password", "token", "secret", "authorization"})


# ---------------------------------------------------------------------------
# Metric Events
# ---------------------------------------------------------------------------


@dataclass
class ToolCallMetric:
    """Metrics for a single tool call."""
    tool_name: str
    start_time: float
    end_time: float = 0.0
    success: bool = False
    error: Optional[str] = None
    retry_count: int = 0
    verification_status: str = "pending"

    @property
    def duration_ms(self) -> float:
        if self.end_time <= 0:
            return 0.0
        return round((self.end_time - self.start_time) * 1000, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool_name,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
            "retry_count": self.retry_count,
            "verification": self.verification_status,
        }


@dataclass
class PhaseMetric:
    """Timing for an execution phase (understanding, planning, executing, etc.)."""
    phase: str
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration_ms(self) -> float:
        if self.end_time <= 0 or self.start_time <= 0:
            return 0.0
        return round((self.end_time - self.start_time) * 1000, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {"phase": self.phase, "duration_ms": self.duration_ms}


# ---------------------------------------------------------------------------
# Request Metrics (per-request telemetry)
# ---------------------------------------------------------------------------


@dataclass
class ThinkRequestMetrics:
    """Complete metrics for a single Think Engine request."""
    request_id: str = field(default_factory=lambda: f"treq_{uuid.uuid4().hex[:12]}")
    think_mode: ThinkMode = ThinkMode.BALANCED
    intent: str = "unknown"
    complexity: str = "LOW"
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float = 0.0

    # Token tracking
    tokens_input: int = 0
    tokens_output: int = 0

    # Call counts
    model_calls: int = 0
    tool_calls_count: int = 0
    retries: int = 0
    replan_count: int = 0

    # Verification
    verification_status: str = "pending"
    verification_passed: bool = False

    # Budget tracking
    budget_tokens_used: int = 0
    budget_time_used: float = 0.0
    budget_iterations_used: int = 0
    budget_tool_calls_used: int = 0

    # Phase timings
    phases: List[PhaseMetric] = field(default_factory=list)

    # Tool call details
    tool_calls: List[ToolCallMetric] = field(default_factory=list)

    # Error tracking
    errors: List[str] = field(default_factory=list)

    # Outcome
    status: str = "pending"  # pending, completed, failed, cancelled, timeout

    # Cancellation / timeout
    was_cancelled: bool = False
    was_timeout: bool = False

    @property
    def total_latency_ms(self) -> float:
        if self.end_time <= 0:
            return round((time.perf_counter() - self.start_time) * 1000, 2)
        return round((self.end_time - self.start_time) * 1000, 2)

    @property
    def total_tokens(self) -> int:
        return self.tokens_input + self.tokens_output

    def start_phase(self, phase: str) -> PhaseMetric:
        """Start timing a phase."""
        pm = PhaseMetric(phase=phase, start_time=time.perf_counter())
        self.phases.append(pm)
        return pm

    def end_phase(self, phase_metric: PhaseMetric) -> None:
        """End timing a phase."""
        phase_metric.end_time = time.perf_counter()

    def start_tool_call(self, tool_name: str) -> ToolCallMetric:
        """Start tracking a tool call."""
        tcm = ToolCallMetric(tool_name=tool_name, start_time=time.perf_counter())
        self.tool_calls.append(tcm)
        self.tool_calls_count += 1
        self.budget_tool_calls_used += 1
        return tcm

    def end_tool_call(
        self,
        metric: ToolCallMetric,
        success: bool = True,
        error: Optional[str] = None,
        verification: str = "pending",
    ) -> None:
        """Complete a tool call metric."""
        metric.end_time = time.perf_counter()
        metric.success = success
        metric.error = error
        metric.verification_status = verification

    def record_model_call(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Record a model inference call."""
        self.model_calls += 1
        self.tokens_input += input_tokens
        self.tokens_output += output_tokens
        self.budget_tokens_used += input_tokens + output_tokens

    def record_retry(self) -> None:
        """Record a retry attempt."""
        self.retries += 1

    def record_replan(self) -> None:
        """Record a re-planning event."""
        self.replan_count += 1

    def record_error(self, error: str) -> None:
        """Record an error."""
        self.errors.append(error)

    def finish(self, status: str = "completed") -> None:
        """Finalize the request metrics."""
        self.end_time = time.perf_counter()
        self.status = status
        self.budget_time_used = self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "request_id": self.request_id,
            "think_mode": self.think_mode.value,
            "intent": self.intent,
            "complexity": self.complexity,
            "total_latency_ms": self.total_latency_ms,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "total_tokens": self.total_tokens,
            "model_calls": self.model_calls,
            "tool_calls_count": self.tool_calls_count,
            "retries": self.retries,
            "replan_count": self.replan_count,
            "verification_status": self.verification_status,
            "verification_passed": self.verification_passed,
            "status": self.status,
            "was_cancelled": self.was_cancelled,
            "was_timeout": self.was_timeout,
            "errors": self.errors,
            "budget": {
                "tokens_used": self.budget_tokens_used,
                "time_used_s": round(self.budget_time_used, 3),
                "iterations_used": self.budget_iterations_used,
                "tool_calls_used": self.budget_tool_calls_used,
            },
            "phases": [p.to_dict() for p in self.phases],
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
        }

    def to_user_summary(self) -> Dict[str, Any]:
        """
        User-facing summary without exposing hidden reasoning.
        Safe to include in API responses.
        """
        return {
            "mode": self.think_mode.value,
            "latency_ms": self.total_latency_ms,
            "tools_used": [tc.tool_name for tc in self.tool_calls],
            "verified": self.verification_passed,
            "status": self.status,
        }

    def format_log_line(self) -> str:
        """Format a single-line structured log entry."""
        return (
            f"[ThinkEngine] req={self.request_id} mode={self.think_mode.value} "
            f"intent={self.intent} complexity={self.complexity} "
            f"latency={self.total_latency_ms:.0f}ms tokens={self.total_tokens} "
            f"model_calls={self.model_calls} tool_calls={self.tool_calls_count} "
            f"retries={self.retries} replans={self.replan_count} "
            f"verified={self.verification_passed} status={self.status}"
        )


# ---------------------------------------------------------------------------
# Structured Logger (with redaction)
# ---------------------------------------------------------------------------


class ThinkLogger:
    """Structured logger with sensitive field redaction."""

    def __init__(self, redact_fields: Optional[List[str]] = None) -> None:
        self._redact = set(redact_fields or []) | _DEFAULT_REDACT_FIELDS

    def log_request_start(self, metrics: ThinkRequestMetrics) -> None:
        """Log the start of a request."""
        logger.info(
            "[ThinkEngine:START] req=%s mode=%s intent=%s",
            metrics.request_id, metrics.think_mode.value, metrics.intent,
        )

    def log_request_end(self, metrics: ThinkRequestMetrics) -> None:
        """Log the end of a request with full metrics."""
        logger.info("%s", metrics.format_log_line())

    def log_phase(self, metrics: ThinkRequestMetrics, phase: str, detail: str = "") -> None:
        """Log a phase transition."""
        logger.debug(
            "[ThinkEngine:PHASE] req=%s phase=%s %s",
            metrics.request_id, phase, detail,
        )

    def log_tool_call(self, metrics: ThinkRequestMetrics, tool_metric: ToolCallMetric) -> None:
        """Log a tool call completion."""
        logger.debug(
            "[ThinkEngine:TOOL] req=%s tool=%s success=%s duration=%.0fms",
            metrics.request_id, tool_metric.tool_name,
            tool_metric.success, tool_metric.duration_ms,
        )

    def log_replan(self, metrics: ThinkRequestMetrics, reason: str) -> None:
        """Log a re-planning event."""
        logger.info(
            "[ThinkEngine:REPLAN] req=%s replan_count=%d reason=%s",
            metrics.request_id, metrics.replan_count, reason,
        )

    def log_error(self, metrics: ThinkRequestMetrics, error: str) -> None:
        """Log an error event."""
        logger.warning(
            "[ThinkEngine:ERROR] req=%s error=%s",
            metrics.request_id, self._redact_string(error),
        )

    def _redact_string(self, text: str) -> str:
        """Redact known sensitive patterns from log text."""
        result = text
        for field_name in self._redact:
            # Simple pattern: field_name=value or "field_name": "value"
            import re
            result = re.sub(
                rf'({field_name})\s*[=:]\s*["\']?[^\s"\']+["\']?',
                rf'\1=***REDACTED***',
                result,
                flags=re.IGNORECASE,
            )
        return result

    def redact_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive fields from a dictionary."""
        redacted = {}
        for key, value in data.items():
            if key.lower() in self._redact:
                redacted[key] = "***REDACTED***"
            elif isinstance(value, dict):
                redacted[key] = self.redact_dict(value)
            elif isinstance(value, list):
                redacted[key] = [
                    self.redact_dict(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                redacted[key] = value
        return redacted
