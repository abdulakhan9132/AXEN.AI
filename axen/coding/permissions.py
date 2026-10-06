"""
axen/coding/permissions.py
===========================
User permission levels and security policy enforcement for the AXEN Coding Agent:
- READ_ONLY: inspect, search, and read workspace files
- PROPOSE: inspect, read, and propose diffs (edits require confirmation)
- EDIT: read, propose, and apply file edits within approved workspace
- EXECUTE: read, propose, apply edits, and execute validation/test commands
- FULL_WORKSPACE: unrestricted autonomous operations within the workspace boundary
"""

from __future__ import annotations

from enum import Enum
import logging
from typing import Optional

logger = logging.getLogger("axen.coding.permissions")


class PermissionLevel(str, Enum):
    READ_ONLY = "READ_ONLY"
    PROPOSE = "PROPOSE"
    EDIT = "EDIT"
    EXECUTE = "EXECUTE"
    FULL_WORKSPACE = "FULL_WORKSPACE"


class OperationType(str, Enum):
    READ = "READ"
    SEARCH = "SEARCH"
    INSPECT = "INSPECT"
    PROPOSE_PATCH = "PROPOSE_PATCH"
    APPLY_PATCH = "APPLY_PATCH"
    WRITE_FILE = "WRITE_FILE"
    DELETE_FILE = "DELETE_FILE"
    EXECUTE_COMMAND = "EXECUTE_COMMAND"
    EXECUTE_TESTS = "EXECUTE_TESTS"
    GIT_READ = "GIT_READ"
    GIT_WRITE = "GIT_WRITE"


class PermissionPolicy:
    """Enforces permissions on operations performed by the Coding Agent."""

    def __init__(self, level: PermissionLevel = PermissionLevel.PROPOSE) -> None:
        self.level = level

    def is_allowed(self, operation: OperationType) -> bool:
        """Check if the given operation is allowed under the current permission level."""
        # Read-only operations are always permitted inside workspace
        if operation in (
            OperationType.READ,
            OperationType.SEARCH,
            OperationType.INSPECT,
            OperationType.PROPOSE_PATCH,
            OperationType.GIT_READ,
        ):
            return True

        if self.level == PermissionLevel.READ_ONLY:
            return False

        if self.level == PermissionLevel.PROPOSE:
            # Propose can only inspect, read, and prepare diffs; cannot apply or execute
            return False

        if self.level == PermissionLevel.EDIT:
            # Edit can apply patches and write files, but cannot execute shell commands
            return operation in (
                OperationType.APPLY_PATCH,
                OperationType.WRITE_FILE,
            )

        if self.level == PermissionLevel.EXECUTE:
            # Execute can edit and run test/validation commands
            return operation in (
                OperationType.APPLY_PATCH,
                OperationType.WRITE_FILE,
                OperationType.EXECUTE_COMMAND,
                OperationType.EXECUTE_TESTS,
            )

        if self.level == PermissionLevel.FULL_WORKSPACE:
            # Full workspace allows everything except destructive git / destructive deletion
            if operation == OperationType.GIT_WRITE:
                return False  # Git commits/pushes require explicit user instruction
            return True

        return False

    def require(self, operation: OperationType, context: Optional[str] = None) -> None:
        """Raise PermissionError if the operation is not permitted."""
        if not self.is_allowed(operation):
            ctx_msg = f" for '{context}'" if context else ""
            raise PermissionError(
                f"Operation '{operation.value}'{ctx_msg} is not permitted under "
                f"current permission level '{self.level.value}'."
            )
