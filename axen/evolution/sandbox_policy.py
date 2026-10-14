"""
axen/evolution/sandbox_policy.py
================================
Immutable security policy for sandbox experiment execution.

Enforces:
1. Deny-list of critical production files that cannot be targeted or accessed
2. Network access blocking
3. Execution timeout ceilings
4. Per-process memory limits
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List, Set


@dataclass(frozen=True)
class SandboxPolicy:
    """
    Immutable policy configuration governing sandboxed evaluation.
    """

    timeout_seconds: float = 10.0
    max_memory_mb: int = 512
    block_network: bool = True
    max_concurrent_sandboxes: int = 3

    # Deny-list: Modules and resources that sandbox cannot touch or overwrite
    DENIED_PATTERNS: Set[str] = field(
        default_factory=lambda: {
            "axen/evolution/sandbox_policy.py",
            "axen/evolution/approval.py",
            "axen/evolution/permissions.py",
            "app/security/permissions.py",
            "axen/agent/loop.py",
            "checkpoints/foundation/",
        }
    )

    def is_path_permitted(self, file_path: str) -> bool:
        """
        Verify that a path does not intersect with any denied resources.
        """
        norm_path = file_path.replace("\\", "/").lower()
        for denied in self.DENIED_PATTERNS:
            if denied.lower() in norm_path:
                return False
        return True

    def validate_execution_request(self, target_file: str) -> None:
        """
        Raise PermissionError if target_file is within the denied perimeter.
        """
        if not self.is_path_permitted(target_file):
            raise PermissionError(
                f"Access denied: '{target_file}' is in the sandbox immutable deny-list."
            )
