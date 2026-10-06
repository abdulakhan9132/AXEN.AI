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

    def remember(self, content: str, category: str = "working_memory", source: str = "user") -> str:
        return self.add_memory(content=content, source=source, category=category)

    def _sim(self, a: str, b: str) -> float:
        import re
        words_a = set(re.findall(r"[a-z0-9']+", a.lower()))
        words_b = set(re.findall(r"[a-z0-9']+", b.lower()))
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / ((len(words_a) * len(words_b)) ** 0.5)

    def recall(self, query: str, min_confidence: float = 0.3, limit: int = 3) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(str(self.db_path))
        try:
            # We fetch all memories (or a large subset) to compute similarity in Python
            rows = conn.execute("SELECT id, content, source, timestamp, confidence, category, metadata FROM memory").fetchall()
        finally:
            conn.close()
        
        scored = []
        for row in rows:
            score = self._sim(query, row[1])
            if score >= min_confidence:
                scored.append({
                    "id": row[0],
                    "content": row[1],
                    "source": row[2],
                    "timestamp": row[3],
                    "confidence": row[4],
                    "category": row[5],
                    "metadata": json.loads(row[6] or "{}"),
                    "score": score
                })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def update(self, memory_id: str, content: Optional[str] = None, confidence: Optional[float] = None) -> bool:
        updates = []
        params = []
        if content is not None:
            updates.append("content = ?")
            params.append(content)
        if confidence is not None:
            updates.append("confidence = ?")
            params.append(confidence)
        if not updates:
            return False
        
        sql = f"UPDATE memory SET {', '.join(updates)} WHERE id = ?"
        params.append(memory_id)
        
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.execute(sql, params)
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def delete(self, memory_id: str) -> bool:
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.execute("DELETE FROM memory WHERE id = ?", (memory_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def clear(self) -> bool:
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("DELETE FROM memory")
            conn.commit()
            return True
        finally:
            conn.close()

    def stats(self) -> Dict[str, int]:
        conn = sqlite3.connect(str(self.db_path))
        try:
            working = conn.execute("SELECT COUNT(*) FROM memory WHERE category='working_memory'").fetchone()[0]
            episodic = conn.execute("SELECT COUNT(*) FROM memory WHERE category='episodic_memory'").fetchone()[0]
            total = conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
        finally:
            conn.close()
        return {"facts": working, "exchanges": episodic, "total": total}

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        store = MemoryStore()
        if cmd == "stats":
            print(json.dumps(store.stats()))
        elif cmd == "remember" and len(sys.argv) > 2:
            fact = sys.argv[2]
            store.remember(fact, category="working_memory")
            print(json.dumps({"success": True}))
        elif cmd == "forget" and len(sys.argv) > 2:
            query = sys.argv[2]
            matches = store.recall(query, limit=10)
            count = 0
            for m in matches:
                store.delete(m["id"])
                count += 1
            print(json.dumps({"removed": count}))
        elif cmd == "recall" and len(sys.argv) > 2:
            query = sys.argv[2]
            matches = store.recall(query)
            print(json.dumps([m["content"] for m in matches]))
        elif cmd == "addExchange" and len(sys.argv) > 3:
            user = sys.argv[2]
            axen = sys.argv[3]
            good = sys.argv[4] == "true" if len(sys.argv) > 4 else False
            store.add_memory(content=json.dumps({"user": user, "axen": axen, "good": good}), source="exchange", category="episodic_memory")
            print(json.dumps({"success": True}))
        elif cmd == "recallExchange" and len(sys.argv) > 2:
            query = sys.argv[2]
            conn = sqlite3.connect(str(store.db_path))
            try:
                rows = conn.execute("SELECT content FROM memory WHERE category='episodic_memory'").fetchall()
            finally:
                conn.close()
            scored = []
            for row in rows:
                try:
                    obj = json.loads(row[0])
                    score = store._sim(query, obj.get("user", ""))
                    if score >= 0.3 and obj.get("good"):
                        scored.append({"axen": obj.get("axen"), "score": score})
                except: pass
            scored.sort(key=lambda x: x["score"], reverse=True)
            print(json.dumps([m["axen"] for m in scored[:3]]))
        elif cmd == "markGood" and len(sys.argv) > 3:
            user = sys.argv[2]
            axen = sys.argv[3]
            good = sys.argv[4] == "true" if len(sys.argv) > 4 else True
            conn = sqlite3.connect(str(store.db_path))
            try:
                rows = conn.execute("SELECT id, content FROM memory WHERE category='episodic_memory' ORDER BY timestamp DESC LIMIT 50").fetchall()
                for row in rows:
                    try:
                        obj = json.loads(row[1])
                        if obj.get("user") == user and obj.get("axen") == axen:
                            obj["good"] = good
                            conn.execute("UPDATE memory SET content=? WHERE id=?", (json.dumps(obj), row[0]))
                            conn.commit()
                            break
                    except: pass
            finally:
                conn.close()
            print(json.dumps({"success": True}))
