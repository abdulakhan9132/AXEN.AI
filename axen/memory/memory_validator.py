"""
axen/memory/memory_validator.py
===============================
Validation, deduplication, and conflict resolution layer for AXEN Memory.

Enforces:
1. Safety policy validation (rejects credentials, secrets, bad lengths).
2. Confidence & importance thresholds (rejects speculative/casual facts from auto-persisting).
3. Deduplication (avoids creating duplicate active memories for the same key/value).
4. Conflict resolution (deactivates obsolete contradictory memories when new ones arrive).
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from .memory_models import ImportanceLevel, MemoryItem, MemoryType
from .memory_policy import MemoryPolicy
from .memory_store import SQLiteMemoryStore


class MemoryValidator:
    """Validates candidate memories and resolves duplicates and conflicts."""

    MIN_AUTO_CONFIDENCE = 0.70  # Casual statements with lower confidence are ignored

    def __init__(self, store: SQLiteMemoryStore) -> None:
        self.store = store

    def validate_candidate(self, candidate: MemoryItem) -> Tuple[bool, Optional[str]]:
        """
        Validate whether a candidate memory should be stored.
        Returns (is_valid, rejection_reason).
        """
        # 1. Safety check
        safe, reason = MemoryPolicy.validate_for_storage(candidate.key, candidate.value)
        if not safe:
            return False, reason

        # 2. Confidence check for extracted/implicit memories
        if candidate.source == "extracted" and candidate.confidence < self.MIN_AUTO_CONFIDENCE:
            return False, f"Confidence too low ({candidate.confidence:.2f} < {self.MIN_AUTO_CONFIDENCE}) for automatic persistence."

        return True, None

    def check_duplicate_or_conflict(
        self,
        candidate: MemoryItem,
    ) -> Tuple[str, Optional[MemoryItem]]:
        """
        Check existing active memories for the same user, key, and scope.

        Returns:
            ("NEW", None)               -> No existing memory for this key & scope
            ("DUPLICATE", existing)     -> Existing memory has identical value (no write needed)
            ("CONFLICT", existing)      -> Existing memory has different value (needs conflict resolution)
        """
        existing = self.store.get_by_key(
            candidate.key,
            user_id=candidate.user_id,
            active_only=True,
            scope=candidate.scope,
            project_id=candidate.project_id,
        )
        if not existing:
            return "NEW", None

        # Check value equivalence
        if self._values_equivalent(existing.value, candidate.value):
            return "DUPLICATE", existing

        return "CONFLICT", existing

    def resolve_and_save(self, candidate: MemoryItem) -> Tuple[bool, str, Optional[int]]:
        """
        Validate, deduplicate, resolve conflict, and persist memory.

        Returns:
            (success: bool, status_message: str, memory_id: Optional[int])
        """
        # 1. Validate
        valid, reason = self.validate_candidate(candidate)
        if not valid:
            return False, reason or "Validation failed.", None

        # 2. Duplicate / Conflict Check
        status, existing = self.check_duplicate_or_conflict(candidate)

        if status == "DUPLICATE":
            assert existing is not None
            # Update last_used_at on the duplicate
            if existing.id is not None:
                self.store.mark_used(existing.id)
            return True, f"Memory already exists: '{candidate.key}' = '{existing.value}'.", existing.id

        if status == "CONFLICT":
            assert existing is not None
            # Deactivate the old conflicting memory scoped to the same project/scope
            self.store.deactivate_key(
                candidate.key,
                user_id=candidate.user_id,
                scope=candidate.scope,
                project_id=candidate.project_id,
            )
            new_id = self.store.add(candidate)
            return True, f"Updated '{candidate.key}' from '{existing.value}' to '{candidate.value}'.", new_id

        # New memory
        new_id = self.store.add(candidate)
        return True, f"Saved new memory: '{candidate.key}' = '{candidate.value}'.", new_id

    @classmethod
    def _values_equivalent(cls, val1: str, val2: str) -> bool:
        """Compare values with normalization for semantic equivalence."""
        c1 = cls._normalize_val(val1)
        c2 = cls._normalize_val(val2)
        if c1 == c2:
            return True

        # Semantic aliases for UI theme
        dark_set = {"dark", "dark ui", "dark mode", "dark interface", "dark theme"}
        light_set = {"light", "light ui", "light mode", "light interface", "light theme"}
        if c1 in dark_set and c2 in dark_set:
            return True
        if c1 in light_set and c2 in light_set:
            return True

        return False

    @staticmethod
    def _normalize_val(val: str) -> str:
        s = val.strip().lower()
        s = re.sub(r"^(?:i\s+prefer\s+|i\s+like\s+|my\s+preference\s+is\s+)", "", s)
        s = re.sub(r"[\s_\-]+", " ", s)
        return s.strip(".,! ")
