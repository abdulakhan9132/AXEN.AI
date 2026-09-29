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

    def retrieve_relevant(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """
        Relevance-based retrieval for long-term memory.
        
        Extracts salient query terms and domain associations, scores stored facts
        by keyword/token overlap, and returns only relevant memories.
        Does NOT return unrelated memories for greetings or simple conversation.
        """
        import re

        clean_q = query.strip().lower()

        # Greetings and conversational pleasantries must NEVER retrieve long-term memory
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

        # Domain associations for common conceptual overlaps
        DOMAIN_ASSOCIATIONS = {
            "dashboard": {"ui", "interface", "design", "theme", "dark", "light", "minimal", "frontend"},
            "website": {"ui", "design", "html", "css", "web", "frontend"},
            "portfolio": {"website", "ui", "projects", "resume"},
            "app": {"software", "ui", "application"},
            "ui": {"dashboard", "interface", "design", "theme"},
        }

        # Expand salient terms with domain associations
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

                # Direct match in key words or prefix match for terms >= 4 chars
                if term in key_words:
                    score += 2.0 * weight
                elif len(term) >= 4 and any(kw.startswith(term) for kw in key_words):
                    score += 1.0 * weight

                # Direct match in value words or prefix match for terms >= 4 chars
                # NEVER do generic substring matching (e.g. 'hi' in 'chip')
                if term in val_words:
                    score += 1.5 * weight
                elif len(term) >= 4 and any(vw.startswith(term) for vw in val_words):
                    score += 0.8 * weight

            if score > 0:
                scored_results.append((score, {"key": key, **entry}))

        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored_results[:top_k]]



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
