"""
axen/memory/memory_models.py
============================
Core data models and schemas for AXEN Phase 3 Memory System.

Defines:
- MemoryItem: typed persistent memory entity
- MemoryType: controlled set of memory categories
- ImportanceLevel & ConfidenceLevel
- ScoredMemory: retrieval result with relevance score and explanation
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class MemoryScope(str, Enum):
    GLOBAL = "global"
    PROJECT = "project"
    CONVERSATION = "conversation"


class MemoryType(str, Enum):
    IDENTITY = "identity"
    PREFERENCE = "preference"
    GOAL = "goal"
    PROJECT = "project"
    PROJECT_FACT = "project_fact"
    USER_FACT = "user_fact"
    SKILL = "skill"
    HABIT = "habit"
    INSTRUCTION = "instruction"
    FACT = "fact"
    RELATIONSHIP = "relationship"
    TECHNICAL_FACT = "technical_fact"
    DECISION = "decision"
    CONSTRAINT = "constraint"
    WORKFLOW = "workflow"


class ImportanceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MemoryItem:
    """
    Structured representation of a single long-term memory unit.
    Mapped 1:1 to the SQLite storage table.
    """
    key: str
    value: str
    memory_type: str = MemoryType.FACT.value
    importance: str = ImportanceLevel.MEDIUM.value
    confidence: float = 1.0
    source: str = "explicit_user"  # "explicit_user" | "conversation" | "project" | "imported" | "system"
    user_id: str = "default_user"
    scope: str = MemoryScope.GLOBAL.value  # "global" | "project" | "conversation"
    scope_id: Optional[str] = None
    project_id: Optional[str] = None
    conversation_id: Optional[str] = None
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    last_used_at: Optional[str] = None
    is_active: bool = True
    id: Optional[int] = None

    def __post_init__(self) -> None:
        if isinstance(self.scope, MemoryScope):
            self.scope = self.scope.value
        if isinstance(self.memory_type, MemoryType):
            self.memory_type = self.memory_type.value
        if isinstance(self.importance, ImportanceLevel):
            self.importance = self.importance.value
        self.key = str(self.key).strip().lower()
        self.value = str(self.value).strip()
        # If project_id is given but scope is global, align scope to project
        if self.project_id and (not self.scope or self.scope == MemoryScope.GLOBAL.value):
            self.scope = MemoryScope.PROJECT.value
        if self.project_id and not self.scope_id:
            self.scope_id = self.project_id

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable dictionary."""
        d = asdict(self)
        return d

    @classmethod
    def from_row(cls, row: dict[str, Any] | tuple) -> "MemoryItem":
        """Construct from SQLite row dictionary or tuple."""
        if isinstance(row, dict):
            return cls(
                id=row.get("id"),
                user_id=row.get("user_id", "default_user"),
                memory_type=row.get("memory_type", MemoryType.FACT.value),
                key=row.get("key", ""),
                value=row.get("value", ""),
                importance=row.get("importance", ImportanceLevel.MEDIUM.value),
                confidence=float(row.get("confidence", 1.0)),
                source=row.get("source", "explicit_user"),
                scope=row.get("scope", MemoryScope.GLOBAL.value),
                scope_id=row.get("scope_id"),
                project_id=row.get("project_id"),
                conversation_id=row.get("conversation_id"),
                created_at=row.get("created_at", _now_iso()),
                updated_at=row.get("updated_at", _now_iso()),
                last_used_at=row.get("last_used_at"),
                is_active=bool(row.get("is_active", 1)),
            )
        # Tuple handling if row is positional tuple
        # Detect tuple length: legacy has 13 columns, new has 16 columns
        if len(row) >= 16:
            return cls(
                id=row[0],
                user_id=row[1],
                memory_type=row[2],
                key=row[3],
                value=row[4],
                importance=row[5],
                confidence=float(row[6]),
                source=row[7],
                scope=row[8],
                scope_id=row[9],
                project_id=row[10],
                conversation_id=row[11],
                created_at=row[12],
                updated_at=row[13],
                last_used_at=row[14],
                is_active=bool(row[15]),
            )
        return cls(
            id=row[0],
            user_id=row[1],
            memory_type=row[2],
            key=row[3],
            value=row[4],
            importance=row[5],
            confidence=float(row[6]),
            source=row[7],
            conversation_id=row[8],
            created_at=row[9],
            updated_at=row[10],
            last_used_at=row[11],
            is_active=bool(row[12]),
        )


@dataclass
class ScoredMemory:
    """A memory candidate scored for relevance to a user query."""
    memory: MemoryItem
    score: float
    reason: str
    selected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.memory.key,
            "value": self.memory.value,
            "type": self.memory.memory_type,
            "importance": self.memory.importance,
            "score": round(self.score, 3),
            "reason": self.reason,
            "selected": self.selected,
        }

    @property
    def key(self) -> str:
        return self.memory.key

    @property
    def value(self) -> str:
        return self.memory.value

    @property
    def memory_type(self) -> str:
        return self.memory.memory_type

    @property
    def relevance_score(self) -> float:
        return self.score
