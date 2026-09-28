from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


class MemoryStore:
    """Persistent SQLite memory for episodic, semantic, and task history data."""

    def __init__(self, db_path: str | Path = "data/agent_memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_db()

    def _initialize_db(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    category TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}'
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def add_memory(
        self,
        *,
        content: str,
        source: str,
        category: str,
        confidence: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        memory_id = f"mem-{abs(hash((content, source, category, confidence, json.dumps(metadata or {})))) % 100000000}"
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute(
                "INSERT OR REPLACE INTO memory (id, content, source, timestamp, confidence, category, metadata) VALUES (?, ?, ?, datetime('now'), ?, ?, ?)",
                (memory_id, content, source, confidence, category, json.dumps(metadata or {})),
            )
            conn.commit()
        finally:
            conn.close()
        return memory_id

    def search(self, query: str, category: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        sql = "SELECT id, content, source, timestamp, confidence, category, metadata FROM memory WHERE content LIKE ?"
        params: List[Any] = [f"%{query}%"]
        if category:
            sql += " AND category = ?"
            params.append(category)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        conn = sqlite3.connect(str(self.db_path))
        try:
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()
        return [
            {
                "id": row[0],
                "content": row[1],
                "source": row[2],
                "timestamp": row[3],
                "confidence": row[4],
                "category": row[5],
                "metadata": json.loads(row[6] or "{}"),
            }
            for row in rows
        ]

    def list_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(str(self.db_path))
        try:
            rows = conn.execute(
                "SELECT id, content, source, timestamp, confidence, category, metadata FROM memory ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        finally:
            conn.close()
        return [
            {
                "id": row[0],
                "content": row[1],
                "source": row[2],
                "timestamp": row[3],
                "confidence": row[4],
                "category": row[5],
                "metadata": json.loads(row[6] or "{}"),
            }
            for row in rows
        ]

    def category(self, category: str, limit: int = 10) -> List[Dict[str, Any]]:
        return self.search("", category=category, limit=limit)
