"""
axen/memory/long_term_memory.py
================================
Long-term persistent memory for AXEN (Phase 3).

Stores user-approved facts, preferences, and notes with:
- SQLite persistent storage engine (data/axen_memory.db)
- Task-aware relevance retrieval
- Deduplication and conflict resolution
- Backward-compatible JSON persistence and dictionary interface

Storage formats:
- Primary: data/axen_memory.db (SQLite)
- Secondary/Export: data/axen_long_term.json
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, List, Optional

from .memory_manager import MemoryManager

logger = logging.getLogger("axen.memory.long_term")


class LongTermMemory:
    """
    User-controlled persistent knowledge store backed by SQLite and MemoryManager.
    Maintains 100% backward-compatible API for existing Phase 1 & 2 tests.
    """

    DEFAULT_PATH = Path("data/axen_long_term.json")

    def __init__(
        self,
        path: Optional[Path | str] = None,
        db_path: Optional[Path | str] = None,
    ) -> None:
        self.path = Path(path) if path else self.DEFAULT_PATH
        if self.path and str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)

        if db_path:
            self.db_path = Path(db_path)
        elif self.path and str(self.path).endswith(".db"):
            self.db_path = self.path
        elif self.path and str(self.path) != ":memory:":
            self.db_path = self.path.parent / "axen_memory.db"
        else:
            self.db_path = Path("data/axen_memory.db")

        self.manager = MemoryManager(db_path=self.db_path)
        self._store: dict[str, dict[str, Any]] = {}
        self._load()
        self._sync_store_to_manager()

    def _sync_store_to_manager(self) -> None:
        """Seed SQLite store from JSON file on first boot if present."""
        for k, v in self._store.items():
            if not self.manager.store.get_by_key(k, active_only=True):
                val = v.get("value", "")
                cat = v.get("category", "general")
                self.manager.remember(k, val, category=cat)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def remember(
        self,
        key: str,
        value: Any,
        category: str = "general",
        importance: str = "medium",
        confidence: float = 1.0,
    ) -> None:
        """
        Store a fact under the given key.
        Saves to SQLite store and mirrors to in-memory/JSON store.
        """
        # Flexible argument detection: if first arg is category name like 'identity', 'preference'
        KNOWN_CATEGORIES = {"identity", "preference", "goal", "project", "skill", "habit", "instruction", "fact", "relationship"}
        if str(key).lower() in KNOWN_CATEGORIES and category not in ("general", ""):
            cat = str(key).lower()
            clean_key = str(value).strip().lower()
            clean_val = str(category).strip()
        else:
            cat = category
            clean_key = str(key).strip().lower()
            clean_val = str(value).strip()

        # Save to SQLite through manager
        self.manager.remember(
            clean_key,
            clean_val,
            category=cat,
            importance=importance,
            confidence=confidence,
        )

        self._store[clean_key] = {
            "value": clean_val,
            "category": cat,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._persist()

    def recall(self, key: str, default: Any = None) -> Any:
        """Retrieve a stored fact by key. Returns default if not found."""
        clean_key = str(key).strip().lower()
        val = self.manager.recall(clean_key)
        if val is not None:
            return val
        entry = self._store.get(clean_key)
        if entry is not None:
            return entry["value"]
        return default

    def forget(self, key: str) -> bool:
        """Remove a stored fact. Returns True if it existed."""
        clean_key = str(key).strip().lower()
        ok_mgr = self.manager.forget(clean_key)
        ok_store = False
        if clean_key in self._store:
            del self._store[clean_key]
            self._persist()
            ok_store = True
        return ok_mgr or ok_store

    def search(self, query: str) -> list[dict[str, Any]]:
        """Return all entries whose key or value contains the query string."""
        results = self.manager.search(query)
        if results:
            return results

        # Fallback to local store
        q = query.lower()
        res = []
        for key, entry in self._store.items():
            val_str = str(entry.get("value", "")).lower()
            if q in key.lower() or q in val_str:
                res.append({"key": key, **entry})
        return res

    def retrieve_relevant(
        self,
        query: str,
        top_k: int = 3,
        task_type: str = "conversation",
    ) -> list[dict[str, Any]]:
        """
        Relevance-based retrieval for long-term memory.
        Task-aware: filters out greetings, pure arithmetic, binary conversion, etc.
        """
        mgr_results = self.manager.retrieve_relevant(query, top_k=top_k, task_type=task_type)
        if mgr_results:
            return mgr_results

        # Fallback keyword algorithm on self._store if SQLite had no hits
        clean_q = query.strip().lower()
        if re.match(
            r"^(?:hello|hi|hey|greetings|good\s+(?:morning|afternoon|evening|day)|howdy|sup)(?:[\s,]+(?:there|axen|friend|everyone|everybody))?[\s\.\!\?]*$|"
            r"^(?:how\s+are\s+you(?:\s+doing|\s+today)?|what'?s\s+up(?:\s+axen|\s+there)?|how's\s+it\s+going)[\s\.\!\?]*$",
            clean_q,
            re.I,
        ):
            return []

        stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "do", "does", "did",
            "you", "your", "my", "me", "i", "what", "which", "who", "whom",
            "this", "that", "these", "those", "have", "has", "had", "for", "to",
            "in", "on", "at", "by", "from", "and", "or", "but", "about", "with",
            "can", "could", "would", "should", "will", "tell", "show", "give", "please",
            "hi", "hello", "hey", "howdy", "sup", "greetings", "good", "morning",
            "afternoon", "evening", "thanks", "thank", "ok", "okay", "yes", "no",
        }
        words = re.findall(r"[a-zA-Z0-9]+", clean_q)
        salient = [w for w in words if len(w) > 1 and w not in stopwords]
        if not salient:
            return []

        DOMAIN_ASSOCIATIONS = {
            "dashboard": {"ui", "interface", "design", "theme", "dark", "light", "minimal", "frontend"},
            "website": {"ui", "design", "html", "css", "web", "frontend"},
            "portfolio": {"website", "ui", "projects", "resume"},
            "app": {"software", "ui", "application"},
            "ui": {"dashboard", "interface", "design", "theme"},
        }
        expanded_terms = set(salient)
        for term in salient:
            if term in DOMAIN_ASSOCIATIONS:
                expanded_terms.update(DOMAIN_ASSOCIATIONS[term])

        scored_results: list[tuple[float, dict[str, Any]]] = []
        for key, entry in self._store.items():
            key_words = set(re.findall(r"[a-zA-Z0-9]+", key.lower()))
            val_str = str(entry.get("value", "")).lower()
            val_words = set(re.findall(r"[a-zA-Z0-9]+", val_str))

            score = 0.0
            for term in expanded_terms:
                is_direct = term in salient
                weight = 1.0 if is_direct else 0.6
                if term in key_words:
                    score += 2.0 * weight
                elif len(term) >= 4 and any(kw.startswith(term) for kw in key_words):
                    score += 1.0 * weight

                if term in val_words:
                    score += 1.5 * weight
                elif len(term) >= 4 and any(vw.startswith(term) for vw in val_words):
                    score += 0.8 * weight

            if score > 0:
                scored_results.append((score, {"key": key, **entry}))

        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored_results[:top_k]]

    def all_facts(self) -> dict[str, Any]:
        """Return a summary of all active stored facts (key -> value)."""
        facts = self.manager.all_facts()
        if facts:
            return facts
        return {k: v["value"] for k, v in self._store.items()}

    def save(self) -> None:
        """Explicitly persist memory to disk."""
        self._persist()

    def clear(self) -> None:
        """Remove all stored facts and save."""
        self.manager.clear()
        self._store.clear()
        self._persist()

    def __len__(self) -> int:
        mgr_len = len(self.manager)
        if mgr_len > 0:
            return mgr_len
        return len(self._store)

    def __repr__(self) -> str:
        return f"LongTermMemory(facts={len(self)}, path={self.path!r}, db={self.db_path!r})"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load memory from JSON disk file if exists."""
        if self.path and self.path.exists() and str(self.path).endswith(".json"):
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._store = data
            except (json.JSONDecodeError, OSError):
                self._store = {}

    def _persist(self) -> None:
        """Write memory to disk as JSON if configured."""
        if self.path and str(self.path).endswith(".json"):
            try:
                self.path.write_text(
                    json.dumps(self._store, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except OSError:
                pass
