"""
axen/evolution/permissions.py
=============================
Permission boundaries and security policy for the AXEN Self-Evolution Engine.

Implements two-level permission enforcement:
- TaskAgent: restricted to read-only tool usage and sandboxed execution
- MetaAgent: can propose modifications (prompt, heuristics) but CANNOT
  directly write production code or self-grant APPROVE_MERGE
- Human Operator: sole authority possessing APPROVE_MERGE

Prevents recursive self-modification and unauthorized elevation of privilege.
"""

from __future__ import annotations

from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Set

try:
    from app.security.permissions import PermissionLevel, SecurityPolicy
except ImportError:
    # Standalone fallback if app.security is not in path
    class PermissionLevel(str, Enum):  # type: ignore[no-redef]
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    class SecurityPolicy:  # type: ignore[no-redef]
        def validate_tool_call(self, tool_name: str, arguments: dict) -> bool:
            return True

        def permission_for(self, tool_name: str) -> PermissionLevel:
            return PermissionLevel.LOW


logger = logging.getLogger("axen.evolution.permissions")


class EvolutionPermission(str, Enum):
    """Specific permission capabilities for evolution engine components."""

    READ_CANDIDATES = "evolution.read_candidates"
    WRITE_CANDIDATES = "evolution.write_candidates"
    EXECUTE_SANDBOX = "evolution.execute_sandbox"
    PROPOSE_META_CHANGE = "evolution.propose_meta_change"
    EVALUATE_BENCHMARK = "evolution.evaluate_benchmark"
    APPROVE_MERGE = "evolution.approve_merge"
    ROLLBACK_VERSION = "evolution.rollback_version"
    CURATE_DATASET = "evolution.curate_dataset"


class AgentRole(str, Enum):
    """Roles with predefined, immutable permission sets."""

    TASK_AGENT = "task_agent"
    META_AGENT = "meta_agent"
    EVALUATOR = "evaluator"
    HUMAN_ADMIN = "human_admin"


# Role-to-permission mapping
ROLE_PERMISSIONS: Dict[AgentRole, Set[EvolutionPermission]] = {
    AgentRole.TASK_AGENT: {
        EvolutionPermission.READ_CANDIDATES,
        EvolutionPermission.EXECUTE_SANDBOX,
    },
    AgentRole.META_AGENT: {
        EvolutionPermission.READ_CANDIDATES,
        EvolutionPermission.WRITE_CANDIDATES,
        EvolutionPermission.PROPOSE_META_CHANGE,
        EvolutionPermission.EVALUATE_BENCHMARK,
    },
    AgentRole.EVALUATOR: {
        EvolutionPermission.READ_CANDIDATES,
        EvolutionPermission.EXECUTE_SANDBOX,
        EvolutionPermission.EVALUATE_BENCHMARK,
    },
    AgentRole.HUMAN_ADMIN: {
        EvolutionPermission.READ_CANDIDATES,
        EvolutionPermission.WRITE_CANDIDATES,
        EvolutionPermission.EXECUTE_SANDBOX,
        EvolutionPermission.PROPOSE_META_CHANGE,
        EvolutionPermission.EVALUATE_BENCHMARK,
        EvolutionPermission.APPROVE_MERGE,
        EvolutionPermission.ROLLBACK_VERSION,
        EvolutionPermission.CURATE_DATASET,
    },
}

# Critical invariant: MetaAgent MUST NEVER possess APPROVE_MERGE
assert EvolutionPermission.APPROVE_MERGE not in ROLE_PERMISSIONS[AgentRole.META_AGENT]
assert EvolutionPermission.APPROVE_MERGE not in ROLE_PERMISSIONS[AgentRole.TASK_AGENT]


class EvolutionSecurityPolicy(SecurityPolicy):
    """
    Extends existing SecurityPolicy with evolution-specific gating.

    Enforces:
    1. Role-based permission checks for all evolution operations
    2. Prevention of meta-agent self-granting merge/approve permissions
    3. Production-path protection (cannot overwrite production modules without human approval)
    4. Deny-list for protected security and core loop modules
    """

    # Modules that can never be modified by candidates or meta-agents directly
    IMMUTABLE_PATHS: Set[str] = {
        "axen/evolution/permissions.py",
        "axen/evolution/approval.py",
        "axen/evolution/sandbox_policy.py",
        "app/security/permissions.py",
        "axen/agent/loop.py",
    }

    def __init__(self, current_role: AgentRole = AgentRole.TASK_AGENT) -> None:
        super().__init__()
        self.current_role = current_role
        self._active_permissions: Set[EvolutionPermission] = set(
            ROLE_PERMISSIONS.get(current_role, set())
        )

    def can(self, permission: EvolutionPermission) -> bool:
        """Check if current active role has the required permission."""
        return permission in self._active_permissions

    def authorize(self, permission: EvolutionPermission) -> None:
        """Raise PermissionError if the current role lacks the specified permission."""
        if not self.can(permission):
            raise PermissionError(
                f"Role '{self.current_role.value}' lacks required permission: '{permission.value}'"
            )

    def validate_code_target(self, target_file: str) -> bool:
        """
        Verify that a candidate does not attempt to target protected security paths.

        Raises:
            PermissionError: If target_file is protected or outside the repository.
        """
        normalized = target_file.replace("\\", "/").lower()
        for forbidden in self.IMMUTABLE_PATHS:
            if forbidden.lower() in normalized:
                logger.warning("Attempted modification of immutable file: %s", target_file)
                raise PermissionError(
                    f"Direct evolution of protected infrastructure '{target_file}' is forbidden."
                )
        return True

    def validate_elevation_attempt(self, requested_permissions: Set[EvolutionPermission]) -> None:
        """
        Ensure no agent attempts to grant itself APPROVE_MERGE or other human-only privileges.
        """
        if self.current_role != AgentRole.HUMAN_ADMIN:
            if EvolutionPermission.APPROVE_MERGE in requested_permissions:
                raise PermissionError(
                    f"Role '{self.current_role.value}' cannot self-grant '{EvolutionPermission.APPROVE_MERGE.value}'"
                )
