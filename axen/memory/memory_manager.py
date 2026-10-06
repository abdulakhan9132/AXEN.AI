"""
axen/memory/memory_manager.py
=============================
Unified Facade and Orchestrator for AXEN Phase 3 Memory System.

Coordinates:
- SQLite persistent storage
- Task-aware relevance retrieval
- Explicit command execution (remember, update, forget, recall)
- Conversational extraction and validation
- Deduplication and conflict resolution
- JSON export / import
- Structured debug reporting
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

from .memory_extractor import MemoryExtractor
from .memory_models import ConfidenceLevel, ImportanceLevel, MemoryItem, MemoryType, ScoredMemory
from .memory_policy import MemoryPolicy
from .memory_retriever import MemoryRetriever
from .memory_store import DEFAULT_DB_PATH, SQLiteMemoryStore
from .memory_validator import MemoryValidator

logger = logging.getLogger("axen.memory.manager")


class ExplicitCommandResult(str):
    """String subclass that also behaves like a dict with ['handled'] and ['message']."""
    def __new__(cls, message: str, handled: bool = True, action: Optional[str] = None):
        obj = str.__new__(cls, message)
        obj.handled = handled
        obj.message = message
        obj.action = action
        return obj

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            if item == "handled":
                return self.handled
            if item == "message":
                return self.message
            if item == "action":
                return self.action
            raise KeyError(item)
        return super().__getitem__(item)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


class MemoryManager:
    """Central manager for persistent, private, local user memories."""

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self.store = SQLiteMemoryStore(db_path or DEFAULT_DB_PATH)
        self.validator = MemoryValidator(self.store)
        self.retriever = MemoryRetriever(self.store)
        self.extractor = MemoryExtractor()

    def close(self) -> None:
        """Close DB connections / clean up resources."""
        pass

    # ------------------------------------------------------------------
    # Explicit User Commands
    # ------------------------------------------------------------------

    def handle_explicit_command(
        self,
        text: str,
        user_id: str = "default_user",
        conversation_id: Optional[str] = None,
        scope: str = "global",
        project_id: Optional[str] = None,
    ) -> Optional[ExplicitCommandResult]:
        """
        Check and execute explicit memory directives:
        - "Remember that I prefer dark UI."
        - "Remember my name is Alex."
        - "Forget that I prefer dark UI."
        - "Update my preference to light UI."
        - "What do you remember about me?"

        Returns an ExplicitCommandResult (str + dict interface) if handled, or None.
        """
        cmd = self.extractor.parse_explicit_command(text)
        if not cmd:
            return None

        action = cmd.get("action")
        eff_scope = "project" if project_id else scope

        if action == "recall":
            memories = self.store.list_active(user_id=user_id, scope=eff_scope, project_id=project_id)
            if not memories:
                scope_label = " in this project" if project_id else " about you"
                return ExplicitCommandResult(f"I don't have any saved notes{scope_label} yet.", handled=True, action="recall")
            lines = [f"- **{m.key}** ({m.memory_type}): {m.value}" for m in memories[:15]]
            header = f"Here is what I remember about this project:\n" if project_id else "Here is what I remember about you:\n"
            return ExplicitCommandResult(header + "\n".join(lines), handled=True, action="recall")

        if action == "forget":
            target = cmd.get("target", "")
            key = cmd.get("key", "")
            # Try by key first
            count = self.store.deactivate_key(key, user_id=user_id, scope=eff_scope, project_id=project_id)
            if count == 0 and target:
                # Search by text match if exact key didn't match
                matches = self.store.search_text(target, user_id=user_id, scope=eff_scope, project_id=project_id)
                for m in matches:
                    if m.id:
                        self.store.delete_by_id(m.id, hard=False)
                        count += 1
            if count > 0:
                return ExplicitCommandResult(f"Got it — I've forgotten that: \"{target or key}\"", handled=True, action="forget")
            return ExplicitCommandResult(f"I didn't find any active memory matching \"{target or key}\".", handled=True, action="forget")

        if action in ("remember", "update"):
            key = cmd["key"]
            val = cmd["value"]
            mtype = cmd.get("memory_type", MemoryType.FACT.value)
            importance = cmd.get("importance", ImportanceLevel.HIGH.value)

            candidate = MemoryItem(
                key=key,
                value=val,
                memory_type=mtype,
                importance=importance,
                confidence=1.0,
                source="explicit_user",
                user_id=user_id,
                scope=eff_scope,
                scope_id=project_id,
                project_id=project_id,
                conversation_id=conversation_id,
            )
            ok, msg, _ = self.validator.resolve_and_save(candidate)
            if ok:
                display_fact = cmd.get("fact") or val
                return ExplicitCommandResult(f"Got it — I've saved that: \"{display_fact}\"", handled=True, action=action)
            return ExplicitCommandResult(f"Could not save memory: {msg}", handled=True, action=action)

        return None

    # ------------------------------------------------------------------
    # Core Memory APIs
    # ------------------------------------------------------------------

    def remember(
        self,
        key: Any = None,
        value: Any = None,
        category: str = "general",
        importance: Any = ImportanceLevel.MEDIUM,
        confidence: float = 1.0,
        user_id: str = "default_user",
        conversation_id: Optional[str] = None,
        memory_type: Optional[Any] = None,
        scope: str = "global",
        scope_id: Optional[str] = None,
        project_id: Optional[str] = None,
        **kwargs,
    ) -> Optional[MemoryItem]:
        """
        Store a fact under the given key with validation and conflict resolution.
        Supports both positional and keyword invocations, and scope tagging.
        """
        # Determine memory_type / category
        mtype = None
        if isinstance(key, MemoryType):
            mtype = key.value
            actual_key = str(value)
            actual_val = str(category)
        elif memory_type is not None:
            mtype = memory_type.value if hasattr(memory_type, "value") else str(memory_type)
            actual_key = str(key)
            actual_val = str(value)
        elif kwargs.get("memory_type") is not None:
            m_kw = kwargs.get("memory_type")
            mtype = m_kw.value if hasattr(m_kw, "value") else str(m_kw)
            actual_key = str(key)
            actual_val = str(value)
        else:
            mtype = category if category in [mt.value for mt in MemoryType] else MemoryType.FACT.value
            actual_key = str(key)
            actual_val = str(value)

        # Normalize importance
        imp = importance.value if hasattr(importance, "value") else str(importance)

        # Normalize confidence
        CONFIDENCE_MAP = {
            ConfidenceLevel.LOW: 0.3,
            "low": 0.3,
            ConfidenceLevel.MEDIUM: 0.6,
            "medium": 0.6,
            ConfidenceLevel.HIGH: 0.95,
            "high": 0.95,
        }
        if confidence in CONFIDENCE_MAP:
            conf_val = CONFIDENCE_MAP[confidence]
        else:
            try:
                conf_val = float(confidence)
            except (ValueError, TypeError):
                conf_val = 1.0

        # Validate security policy
        is_safe, reason = MemoryPolicy.validate_for_storage(actual_key, actual_val)
        if not is_safe:
            logger.warning(f"Memory blocked by policy: {reason}")
            return None

        eff_scope = "project" if project_id else scope
        eff_scope_id = scope_id or project_id

        candidate = MemoryItem(
            key=actual_key,
            value=actual_val,
            memory_type=mtype,
            importance=imp,
            confidence=conf_val,
            source=kwargs.get("source", "explicit_user"),
            user_id=user_id,
            scope=eff_scope,
            scope_id=eff_scope_id,
            project_id=project_id,
            conversation_id=conversation_id,
        )
        ok, msg, mem_id = self.validator.resolve_and_save(candidate)
        if ok:
            if mem_id and candidate.id is None:
                candidate.id = mem_id
            return candidate
        logger.debug(f"Memory validation failed: {msg}")
        return None

    def recall(
        self,
        key: str,
        default: Any = None,
        user_id: str = "default_user",
        scope: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> Any:
        """Retrieve the value of an active memory by key and optional scope."""
        item = self.store.get_by_key(key, user_id=user_id, active_only=True, scope=scope, project_id=project_id)
        if item:
            return item.value
        return default

    def forget(
        self,
        key: str,
        user_id: str = "default_user",
        scope: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> bool:
        """Deactivate all memories under a key with optional scope filtering."""
        count = self.store.deactivate_key(key, user_id=user_id, scope=scope, project_id=project_id)
        return count > 0

    def delete(
        self,
        key: str,
        user_id: str = "default_user",
        scope: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> bool:
        """Alias for forget."""
        return self.forget(key, user_id=user_id, scope=scope, project_id=project_id)

    def list_memories(
        self,
        user_id: Optional[str] = None,
        is_active: Optional[bool] = None,
        scope: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> List[MemoryItem]:
        """List memories matching user, active status, scope, and project_id."""
        all_mems = self.store.all_memories(user_id=user_id, scope=scope, project_id=project_id, include_inactive=True)
        if is_active is True:
            return [m for m in all_mems if m.is_active]
        elif is_active is False:
            return [m for m in all_mems if not m.is_active]
        return all_mems

    def is_retrieval_needed(self, query: str) -> bool:
        """Task-aware check to determine if memory retrieval is appropriate."""
        return self.retriever.is_task_relevant(query)

    def retrieve(
        self,
        query: str,
        user_id: str = "default_user",
        top_k: int = 3,
        task_type: str = "conversation",
        conversation_id: Optional[str] = None,
        scope: Optional[str] = None,
        project_id: Optional[str] = None,
        include_global: bool = True,
    ) -> List[ScoredMemory]:
        """Retrieve relevant scored memories exceeding relevance threshold."""
        scored = self.retriever.retrieve_relevant(
            query,
            user_id=user_id,
            top_k=top_k,
            task_type=task_type,
            scope=scope,
            project_id=project_id,
            include_global=include_global,
        )
        return [sm for sm in scored if sm.selected]

    def format_context_block(self, memories: List[Any]) -> str:
        """Format relevant memories for injection into system prompt."""
        if not memories:
            return ""
        items = []
        for m in memories:
            if hasattr(m, "memory"):
                items.append(f"- {m.memory.key}: {m.memory.value}")
            elif hasattr(m, "key") and hasattr(m, "value"):
                items.append(f"- {m.key}: {m.value}")
            elif isinstance(m, dict):
                items.append(f"- {m.get('key')}: {m.get('value')}")
        if not items:
            return ""
        return (
            "RELEVANT USER MEMORY:\n"
            "(Note: Current user instructions always supersede saved memories.)\n"
            + "\n".join(items)
        )

    def process_message_for_memory(
        self,
        message: str,
        user_id: str = "default_user",
        conversation_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> List[MemoryItem]:
        """Extract and validate candidate memories from conversational messages."""
        item = self.process_turn_extraction(message, user_id=user_id, conversation_id=conversation_id, project_id=project_id)
        return [item] if item else []

    def retrieve_relevant(
        self,
        query: str,
        user_id: str = "default_user",
        top_k: int = 3,
        task_type: str = "conversation",
        scope: Optional[str] = None,
        project_id: Optional[str] = None,
        include_global: bool = True,
    ) -> List[dict[str, Any]]:
        """
        Retrieve relevant memories formatted as dictionaries (compatible with legacy callers).
        """
        scored = self.retriever.retrieve_relevant(
            query,
            user_id=user_id,
            top_k=top_k,
            task_type=task_type,
            scope=scope,
            project_id=project_id,
            include_global=include_global,
        )
        return [{"key": sm.memory.key, "value": sm.memory.value, "category": sm.memory.memory_type} for sm in scored]

    def retrieve_scored(
        self,
        query: str,
        user_id: str = "default_user",
        top_k: int = 3,
        task_type: str = "conversation",
        scope: Optional[str] = None,
        project_id: Optional[str] = None,
        include_global: bool = True,
    ) -> List[ScoredMemory]:
        """Retrieve relevant memories with full scoring and reasoning."""
        return self.retriever.retrieve_relevant(
            query,
            user_id=user_id,
            top_k=top_k,
            task_type=task_type,
            scope=scope,
            project_id=project_id,
            include_global=include_global,
        )

    def search(
        self,
        query: str,
        user_id: str = "default_user",
        scope: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> List[dict[str, Any]]:
        """Search memories containing query substring."""
        items = self.store.search_text(query, user_id=user_id, scope=scope, project_id=project_id, active_only=True)
        return [{"key": m.key, "value": m.value, "category": m.memory_type, "timestamp": m.updated_at} for m in items]

    def all_facts(
        self,
        user_id: str = "default_user",
        scope: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> dict[str, str]:
        """Return a mapping of all active facts (key -> value)."""
        active = self.store.list_active(user_id=user_id, scope=scope, project_id=project_id)
        return {m.key: m.value for m in active}

    def clear(
        self,
        user_id: Optional[str] = None,
        scope: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> None:
        """Clear memories."""
        self.store.clear(user_id=user_id, scope=scope, project_id=project_id)

    def __len__(self) -> int:
        return self.store.count(active_only=True)

    # ------------------------------------------------------------------
    # Automatic Extraction from Conversational Turns
    # ------------------------------------------------------------------

    def process_turn_extraction(
        self,
        user_message: str,
        user_id: str = "default_user",
        conversation_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> Optional[MemoryItem]:
        """
        Extract and validate candidate memories from conversational messages.
        Casual / speculative statements are discarded or kept out of permanent memory.
        """
        candidate = self.extractor.extract_implicit_candidate(user_message)
        if not candidate:
            return None

        candidate.user_id = user_id
        candidate.conversation_id = conversation_id
        if project_id:
            candidate.scope = "project"
            candidate.scope_id = project_id
            candidate.project_id = project_id

        # Validate & Save
        ok, msg, mem_id = self.validator.resolve_and_save(candidate)
        if ok and mem_id:
            logger.info("Auto-extracted memory saved: key=%s value=%s", candidate.key, candidate.value)
            return candidate
        else:
            logger.debug("Candidate memory rejected: %s", msg)
            return None
            logger.debug("Candidate memory rejected: %s", msg)
            return None

    # ------------------------------------------------------------------
    # Backup & Export / Import (Step 24)
    # ------------------------------------------------------------------

    def export_memories(self, user_id: str = "default_user") -> dict[str, Any]:
        """Export all user memories to a clean JSON structure."""
        memories = self.store.all_memories(user_id=user_id, include_inactive=True)
        # Filter out any secrets just in case
        clean_list = []
        for m in memories:
            if not MemoryPolicy.is_secret(f"{m.key} {m.value}"):
                clean_list.append(m.to_dict())

        return {
            "memory_version": 1,
            "user_id": user_id,
            "total_memories": len(clean_list),
            "memories": clean_list,
        }

    def import_memories(self, data: dict[str, Any], user_id: str = "default_user") -> int:
        """Import memories from exported JSON dictionary."""
        mem_list = data.get("memories", [])
        imported_count = 0
        for m_data in mem_list:
            key = m_data.get("key", "")
            val = m_data.get("value", "")
            if not key or not val or MemoryPolicy.is_secret(f"{key} {val}"):
                continue

            item = MemoryItem(
                key=key,
                value=val,
                memory_type=m_data.get("memory_type", MemoryType.FACT.value),
                importance=m_data.get("importance", ImportanceLevel.MEDIUM.value),
                confidence=float(m_data.get("confidence", 1.0)),
                source="imported",
                user_id=user_id,
            )
            ok, _, _ = self.validator.resolve_and_save(item)
            if ok:
                imported_count += 1
        return imported_count

    # ------------------------------------------------------------------
    # Memory Debug Block (Step 20)
    # ------------------------------------------------------------------

    def format_debug_block(self, query: str, scored_memories: List[ScoredMemory]) -> str:
        """
        Format clean structured memory debugging block:
        === MEMORY DEBUG ===
        query: ...
        retrieval: enabled
        candidates: 4
        selected: 1
        memory: preference / ui_theme / dark
        relevance: 0.94
        reason: high task relevance
        """
        retrieval_status = "enabled" if self.retriever.is_task_relevant(query) else "disabled"
        selected = [sm for sm in scored_memories if sm.selected]

        lines = [
            "=== MEMORY DEBUG ===",
            f"query: {query[:80]}",
            f"retrieval: {retrieval_status}",
            f"candidates: {len(scored_memories)}",
            f"selected: {len(selected)}",
        ]

        if selected:
            top = selected[0]
            lines.extend([
                f"memory: {top.memory.memory_type} / {top.memory.key} / {top.memory.value}",
                f"relevance: {top.score:.2f}",
                f"reason: {top.reason}",
            ])
        else:
            lines.extend([
                "memory: none",
                "relevance: 0.00",
                "reason: no relevant memories exceeded threshold",
            ])

        return "\n".join(lines)
