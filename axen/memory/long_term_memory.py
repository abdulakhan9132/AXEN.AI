"""
axen/memory/long_term_memory.py
================================
Long-term persistent memory for AXEN.

Stores user-approved facts, preferences, and notes in a JSON file.
Memory is only saved explicitly — AXEN does not automatically
store sensitive information.

Storage format: data/axen_long_term.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional


class LongTermMemory:
    """
    User-controlled persistent knowledge store.

    Facts are stored as key-value pairs with metadata.
    The backing file is only written when save() is called
    or remember() is used.

    Example usage:
        mem = LongTermMemory()
        mem.remember("preferred_language", "Python")
        mem.remember("user_name", "Hamza")
        value = mem.recall("preferred_language")  # "Python"
        mem.save()
    """

    DEFAULT_PATH = Path("data/axen_long_term.json")

    def __init__(self, path: Optional[Path | str] = None) -> None:
        self.path = Path(path) if path else self.DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._store: dict[str, dict[str, Any]] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def remember(self, key: str, value: Any, category: str = "general") -> None:
        """
        Store a fact under the given key.
        Overwrites any existing value for that key.
        """
        self._store[key] = {
            "value": value,
            "category": category,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._persist()

    def recall(self, key: str, default: Any = None) -> Any:
        """Retrieve a stored fact by key. Returns default if not found."""
        entry = self._store.get(key)
        if entry is None:
            return default
        return entry["value"]

    def forget(self, key: str) -> bool:
        """Remove a stored fact. Returns True if it existed."""
        if key in self._store:
            del self._store[key]
            self._persist()
            return True
        return False

    def search(self, query: str) -> list[dict[str, Any]]:
        """Return all entries whose key or value contains the query string."""
        q = query.lower()
        results = []
        for key, entry in self._store.items():
            val_str = str(entry.get("value", "")).lower()
            if q in key.lower() or q in val_str:
                results.append({"key": key, **entry})
        return results

    def all_facts(self) -> dict[str, Any]:
        """Return a summary of all stored facts (key → value)."""
        return {k: v["value"] for k, v in self._store.items()}

    def save(self) -> None:
        """Explicitly persist memory to disk."""
        self._persist()

    def clear(self) -> None:
        """Remove all stored facts and save."""
        self._store.clear()
        self._persist()

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"LongTermMemory(facts={len(self._store)}, path={self.path!r})"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load memory from disk if the file exists."""
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._store = data
            except (json.JSONDecodeError, OSError):
                self._store = {}

    def _persist(self) -> None:
        """Write memory to disk as JSON."""
        try:
            self.path.write_text(
                json.dumps(self._store, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass  # Non-fatal — memory still works in-session
