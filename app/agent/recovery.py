from __future__ import annotations

from typing import Any, Dict, List


class RecoveryManager:
    """Handles retry, alternative paths, and replanning after failures."""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    def recover(self, *, error: str, step: Dict[str, Any], attempts: int) -> Dict[str, Any]:
        action = {
            "status": "replan",
            "retryable": attempts < self.max_retries,
            "error": error,
            "step": step,
        }
        if "permission" in error.lower() or "denied" in error.lower():
            action["status"] = "waiting_for_user"
        elif "not found" in error.lower() or "unknown tool" in error.lower():
            action["status"] = "replan"
        elif attempts >= self.max_retries:
            action["status"] = "failed"
        return action
