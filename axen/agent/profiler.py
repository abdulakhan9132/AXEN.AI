"""
axen/agent/profiler.py
======================
High-precision latency profiling and instrumentation for AXEN requests.

Tracks end-to-end stage timings:
  1. request received
  2. conversation retrieval
  3. memory retrieval
  4. classification
  5. planning
  6. tool selection & execution
  7. prompt construction
  8. tokenization & model generation
  9. verification
  10. post-processing
  11. total response time
"""

from __future__ import annotations

import time
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("axen.profiler")


class RequestProfiler:
    """
    Per-request latency profiler.

    Records stage start/end timestamps and produces formatted [PERF] blocks.
    """

    def __init__(self, request_id: Optional[str] = None) -> None:
        self.request_id = request_id or f"req_{int(time.time()*1000)}"
        self._start_time = time.perf_counter()
        self._marks: Dict[str, float] = {"start": self._start_time}
        self._durations: Dict[str, float] = {}

    def mark(self, stage_name: str) -> None:
        """Record the point in time when a stage completes or begins."""
        self._marks[stage_name] = time.perf_counter()

    def record_duration(self, stage_name: str, duration_sec: float) -> None:
        """Explicitly record a duration for a specific stage."""
        self._durations[stage_name] = duration_sec

    def finish(self) -> Dict[str, float]:
        """Compute all stage durations relative to marks."""
        total = time.perf_counter() - self._start_time
        self._durations["total"] = round(total, 4)
        return dict(self._durations)

    def format_perf_block(self) -> str:
        """Format a human-readable [PERF] breakdown matching Section 5."""
        durations = self.finish()
        lines = [f"[PERF req={self.request_id}]"]
        order = [
            ("request", "request"),
            ("conversation", "conversation retrieval"),
            ("memory", "memory retrieval"),
            ("classification", "classification"),
            ("planning", "planning"),
            ("tool_selection", "tool selection"),
            ("tools", "tool execution"),
            ("prompt", "prompt construction"),
            ("tokenization", "tokenization"),
            ("generation", "model generation"),
            ("verification", "verification"),
            ("postprocess", "postprocess"),
            ("sse_first_event", "SSE first event"),
            ("sse_first_token", "SSE first token"),
            ("total", "total"),
        ]
        for key, label in order:
            if key in durations:
                lines.append(f"{label}: {durations[key]:.4f}s")
        return "\n".join(lines)
