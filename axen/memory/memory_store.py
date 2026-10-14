"""
axen/memory/memory_store.py
===========================
Thread-safe SQLite storage engine for AXEN Phase 3 Memory System.

Features:
- Structured schema supporting typed memories, confidence, importance, active flags.
- WAL mode for concurrent reads & crash recovery.
- Safe connection management and transaction isolation.
- Soft-deletion for conflict preservation and user reversibility.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from .memory_models import ImportanceLevel, MemoryItem, MemoryType

logger = logging.getLogger("axen.memory.store")

DEFAULT_DB_PATH = Path("data/axen_memory.db")


class SQLiteMemoryStore:
    """Thread-safe persistent SQLite storage for AXEN memories."""

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._memory_connection: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        # A :memory: database belongs to its connection. All public operations
        # already hold _lock, so reuse one connection for this store only.
        if self._memory_connection is not None:
            return self._memory_connection
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=15.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        if str(self.db_path) == ":memory:":
            self._memory_connection = conn
        # Enable Write-Ahead Logging for concurrency & crash safety
        if str(self.db_path) != ":memory:":
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
            except Exception as e:
                logger.debug("Could not set WAL pragma: %s", e)
        return conn

    def _init_db(self) -> None:
        """Initialize SQLite tables, columns, and indexes."""
        with self._lock, self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL DEFAULT 'default_user',
                    memory_type TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    importance TEXT NOT NULL DEFAULT 'medium',
                    confidence REAL NOT NULL DEFAULT 1.0,
                    source TEXT NOT NULL DEFAULT 'explicit_user',
                    scope TEXT NOT NULL DEFAULT 'global',
                    scope_id TEXT,
                    project_id TEXT,
                    conversation_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_used_at TEXT,
                    is_active INTEGER NOT NULL DEFAULT 1
                );
            """)
            # Check and run automatic migration for existing databases missing scope / scope_id / project_id
            cursor = conn.execute("PRAGMA table_info(memories);")
            existing_cols = {row["name"] for row in cursor.fetchall()}
            if "scope" not in existing_cols:
                try:
                    conn.execute("ALTER TABLE memories ADD COLUMN scope TEXT NOT NULL DEFAULT 'global';")
                except Exception as e:
                    logger.debug("Migration scope column exists or failed: %s", e)
            if "scope_id" not in existing_cols:
                try:
                    conn.execute("ALTER TABLE memories ADD COLUMN scope_id TEXT;")
                except Exception as e:
                    logger.debug("Migration scope_id column exists or failed: %s", e)
            if "project_id" not in existing_cols:
                try:
                    conn.execute("ALTER TABLE memories ADD COLUMN project_id TEXT;")
                except Exception as e:
                    logger.debug("Migration project_id column exists or failed: %s", e)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_user_active ON memories (user_id, is_active);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_key ON memories (key);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_type ON memories (memory_type);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_scope ON memories (scope, scope_id, is_active);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_project ON memories (project_id, is_active);")
            conn.commit()

    # ------------------------------------------------------------------
    # CRUD Operations
    # ------------------------------------------------------------------

    def add(self, item: MemoryItem) -> int:
        """Insert a new memory record and return its row ID."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO memories (
                    user_id, memory_type, key, value, importance, confidence,
                    source, scope, scope_id, project_id, conversation_id,
                    created_at, updated_at, last_used_at, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    item.user_id,
                    item.memory_type,
                    item.key,
                    item.value,
                    item.importance,
                    item.confidence,
                    item.source,
                    item.scope or "global",
                    item.scope_id or item.project_id,
                    item.project_id,
                    item.conversation_id,
                    item.created_at,
                    item.updated_at,
                    item.last_used_at,
                    1 if item.is_active else 0,
                ),
            )
            conn.commit()
            item.id = cursor.lastrowid
            return item.id

    def get_by_key(
        self,
        key: str,
        user_id: str = "default_user",
        active_only: bool = True,
        scope: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> Optional[MemoryItem]:
        """Retrieve most recent memory for a key with optional scope filtering."""
        clean_key = key.strip().lower()
        query = "SELECT * FROM memories WHERE user_id = ? AND key = ?"
        params: list[Any] = [user_id, clean_key]
        if scope:
            query += " AND scope = ?"
            params.append(scope)
        if project_id:
            query += " AND (project_id = ? OR scope_id = ?)"
            params.extend([project_id, project_id])
        if active_only:
            query += " AND is_active = 1"
        query += " ORDER BY id DESC LIMIT 1"

        with self._lock, self._get_connection() as conn:
            row = conn.execute(query, params).fetchone()
            if row:
                return MemoryItem.from_row(dict(row))
            return None

    def get_by_id(self, item_id: int) -> Optional[MemoryItem]:
        """Retrieve a specific memory by primary ID."""
        with self._lock, self._get_connection() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (item_id,)).fetchone()
            if row:
                return MemoryItem.from_row(dict(row))
            return None

    def deactivate_key(
        self,
        key: str,
        user_id: str = "default_user",
        scope: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> int:
        """Soft-deactivate all active records for a given key, optionally filtered by scope."""
        clean_key = key.strip().lower()
        now_str = datetime.now(timezone.utc).isoformat()
        query = "UPDATE memories SET is_active = 0, updated_at = ? WHERE user_id = ? AND key = ? AND is_active = 1"
        params: list[Any] = [now_str, user_id, clean_key]
        if scope:
            query += " AND scope = ?"
            params.append(scope)
        if project_id:
            query += " AND (project_id = ? OR scope_id = ?)"
            params.extend([project_id, project_id])

        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.rowcount

    def update_value(self, item_id: int, new_value: str, confidence: Optional[float] = None) -> bool:
        """Update the value and updated_at timestamp of an existing memory."""
        now_str = datetime.now(timezone.utc).isoformat()
        with self._lock, self._get_connection() as conn:
            if confidence is not None:
                cursor = conn.execute(
                    "UPDATE memories SET value = ?, confidence = ?, updated_at = ? WHERE id = ?;",
                    (new_value.strip(), confidence, now_str, item_id),
                )
            else:
                cursor = conn.execute(
                    "UPDATE memories SET value = ?, updated_at = ? WHERE id = ?;",
                    (new_value.strip(), now_str, item_id),
                )
            conn.commit()
            return cursor.rowcount > 0

    def delete_by_id(self, item_id: int, hard: bool = False) -> bool:
        """Delete or soft-deactivate by ID."""
        now_str = datetime.now(timezone.utc).isoformat()
        with self._lock, self._get_connection() as conn:
            if hard:
                cursor = conn.execute("DELETE FROM memories WHERE id = ?;", (item_id,))
            else:
                cursor = conn.execute(
                    "UPDATE memories SET is_active = 0, updated_at = ? WHERE id = ?;",
                    (now_str, item_id),
                )
            conn.commit()
            return cursor.rowcount > 0

    def list_active(
        self,
        user_id: str = "default_user",
        memory_type: Optional[str] = None,
        scope: Optional[str] = None,
        project_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[MemoryItem]:
        """List active memories for user, optionally filtered by type, scope, and project_id."""
        query = "SELECT * FROM memories WHERE user_id = ? AND is_active = 1"
        params: list[Any] = [user_id]
        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type)
        if scope:
            query += " AND scope = ?"
            params.append(scope)
        if project_id:
            query += " AND (project_id = ? OR scope_id = ?)"
            params.extend([project_id, project_id])
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._lock, self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [MemoryItem.from_row(dict(r)) for r in rows]

    def all_memories(
        self,
        user_id: Optional[str] = None,
        scope: Optional[str] = None,
        project_id: Optional[str] = None,
        include_inactive: bool = False,
    ) -> List[MemoryItem]:
        """Fetch all memories, optionally filtered by user, scope, and project."""
        query = "SELECT * FROM memories WHERE 1=1"
        params: list[Any] = []
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        if scope:
            query += " AND scope = ?"
            params.append(scope)
        if project_id:
            query += " AND (project_id = ? OR scope_id = ?)"
            params.extend([project_id, project_id])
        if not include_inactive:
            query += " AND is_active = 1"
        query += " ORDER BY id ASC"

        with self._lock, self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [MemoryItem.from_row(dict(r)) for r in rows]

    def search_text(
        self,
        query: str,
        user_id: str = "default_user",
        scope: Optional[str] = None,
        project_id: Optional[str] = None,
        active_only: bool = True,
    ) -> List[MemoryItem]:
        """Search active memories by substring in key or value with scope filtering."""
        pattern = f"%{query.strip().lower()}%"
        sql = "SELECT * FROM memories WHERE user_id = ? AND (LOWER(key) LIKE ? OR LOWER(value) LIKE ?)"
        params: list[Any] = [user_id, pattern, pattern]
        if scope:
            sql += " AND scope = ?"
            params.append(scope)
        if project_id:
            sql += " AND (project_id = ? OR scope_id = ?)"
            params.extend([project_id, project_id])
        if active_only:
            sql += " AND is_active = 1"
        sql += " ORDER BY id DESC LIMIT 50"

        with self._lock, self._get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [MemoryItem.from_row(dict(r)) for r in rows]

    def mark_used(self, item_id: int) -> None:
        """Update last_used_at timestamp when a memory is retrieved."""
        now_str = datetime.now(timezone.utc).isoformat()
        with self._lock, self._get_connection() as conn:
            conn.execute("UPDATE memories SET last_used_at = ? WHERE id = ?;", (now_str, item_id))
            conn.commit()

    def count(
        self,
        user_id: str = "default_user",
        scope: Optional[str] = None,
        project_id: Optional[str] = None,
        active_only: bool = True,
    ) -> int:
        """Count memories for a user and scope."""
        sql = "SELECT COUNT(*) FROM memories WHERE user_id = ?"
        params: list[Any] = [user_id]
        if scope:
            sql += " AND scope = ?"
            params.append(scope)
        if project_id:
            sql += " AND (project_id = ? OR scope_id = ?)"
            params.extend([project_id, project_id])
        if active_only:
            sql += " AND is_active = 1"

        with self._lock, self._get_connection() as conn:
            row = conn.execute(sql, params).fetchone()
            return int(row[0]) if row else 0

    def clear(
        self,
        user_id: Optional[str] = None,
        scope: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> int:
        """Hard delete memories, optionally filtered by user_id, scope, or project_id."""
        with self._lock, self._get_connection() as conn:
            conditions = []
            params: list[Any] = []
            if user_id:
                conditions.append("user_id = ?")
                params.append(user_id)
            if scope:
                conditions.append("scope = ?")
                params.append(scope)
            if project_id:
                conditions.append("(project_id = ? OR scope_id = ?)")
                params.extend([project_id, project_id])

            if conditions:
                sql = f"DELETE FROM memories WHERE {' AND '.join(conditions)};"
                cursor = conn.execute(sql, params)
            else:
                cursor = conn.execute("DELETE FROM memories;")
            conn.commit()
            return cursor.rowcount


# Alias for convenience
MemoryStore = SQLiteMemoryStore
