"""
axen/memory/memory_retriever.py
===============================
Task-aware relevance filtering and scoring for AXEN Memory Retrieval.

Implements:
1. Task-awareness: Bypasses memory for greetings, calculations, binary conversions, raw code runs.
2. Transparent relevance scoring: Term frequency, key matching, category bonus, importance weighting.
3. Rejection threshold: Drops weak coincidental matches.
4. Clean explanation of why a memory was selected.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

from .memory_models import MemoryItem, MemoryType, ScoredMemory
from .memory_store import SQLiteMemoryStore


class MemoryRetriever:
    """Retrieves and scores relevant memories for user requests."""

    RELEVANCE_THRESHOLD = 1.4  # Minimum score required to be selected

    _STOPWORDS = frozenset({
        "a", "an", "the", "is", "am", "are", "was", "were", "do", "does", "did",
        "you", "your", "my", "me", "i", "what", "which", "who", "whom",
        "this", "that", "these", "those", "have", "has", "had", "for", "to",
        "in", "on", "at", "by", "from", "and", "or", "but", "about", "with",
        "can", "could", "would", "should", "will", "tell", "show", "give", "please",
        "hi", "hello", "hey", "howdy", "sup", "greetings", "good", "morning",
        "afternoon", "evening", "thanks", "thank", "ok", "okay", "yes", "no",
    })

    # Domain synonyms to bridge queries and keys
    _DOMAIN_SYNONYMS = {
        "ui": {"theme", "mode", "dark", "light", "interface", "design", "style", "dashboard", "frontend"},
        "theme": {"ui", "mode", "dark", "light", "interface", "color", "style"},
        "preference": {"like", "prefer", "favorite", "choice", "style", "theme"},
        "prefer": {"preference", "like", "favorite", "choice", "theme", "mode"},
        "project": {"building", "working", "work", "app", "axen", "code", "development", "task"},
        "working": {"project", "work", "building", "task", "development", "app"},
        "work": {"project", "working", "building", "task", "development"},
        "building": {"project", "working", "work", "app", "development"},
        "name": {"identity", "called", "who", "user"},
        "language": {"python", "javascript", "rust", "c++", "coding", "code"},
    }

    def __init__(self, store: SQLiteMemoryStore) -> None:
        self.store = store

    def is_task_relevant(self, query: str, task_type: str = "conversation") -> bool:
        """
        Determine whether the current request could genuinely benefit from long-term memory.
        Bypasses retrieval for pure deterministic tasks and greetings.
        """
        clean = query.strip().lower()

        # 1. Greetings and pleasantries -> NEVER retrieve memory
        if re.match(
            r"^(?:hello|hi|hey|greetings|good\s+(?:morning|afternoon|evening|day)|howdy|sup)(?:[\s,]+(?:there|axen|friend|everyone|everybody))?[\s\.\!\?]*$|"
            r"^(?:how\s+are\s+you(?:\s+doing|\s+today)?|what'?s\s+up(?:\s+axen|\s+there)?|how's\s+it\s+going)[\s\.\!\?]*$",
            clean,
            re.I,
        ):
            return False

        # 2. Pure deterministic arithmetic calculations -> NO memory needed
        if task_type == "calculation" or (
            re.match(r"^(?:what\s+is|what's|calculate|compute|solve)?\s*[\d\s\+\-\*\/\(\)\^\.%=]+[?\.]*$", clean, re.I)
            and not any(term in clean for term in ["remember", "prefer", "my", "name", "project"])
        ):
            return False

        # 3. Pure computer systems conversions -> NO memory needed unless personal
        if task_type == "computer_systems" and not any(term in clean for term in ["my", "remember", "prefer", "project"]):
            return False

        # 4. Pure symbolic math -> NO memory needed unless personal
        if task_type == "math" and not any(term in clean for term in ["my", "remember", "prefer", "project"]):
            return False

        # 5. Generic python execution request without memory context
        if task_type == "code_execution" and re.match(r"^run(?:\s+this)?\s+python:?\s*print", clean):
            return False

        return True

    def retrieve_relevant(
        self,
        query: str,
        user_id: str = "default_user",
        top_k: int = 3,
        task_type: str = "conversation",
        scope: Optional[str] = None,
        project_id: Optional[str] = None,
        include_global: bool = True,
    ) -> List[ScoredMemory]:
        """
        Retrieve and score active memories relevant to query.
        Returns top_k ScoredMemory instances exceeding RELEVANCE_THRESHOLD.
        Respects strict project isolation:
        - If project_id is given, returns project-scoped memories for that project
          and (if include_global=True) global memories.
        - NEVER returns project memories belonging to other projects.
        """
        if not self.is_task_relevant(query, task_type=task_type):
            return []

        all_candidates: List[MemoryItem] = []
        if project_id:
            # Load project-specific memories
            proj_memories = self.store.list_active(user_id=user_id, project_id=project_id, limit=200)
            all_candidates.extend(proj_memories)
            if include_global:
                global_memories = self.store.list_active(user_id=user_id, scope="global", limit=200)
                all_candidates.extend(global_memories)
        elif scope == "global":
            all_candidates = self.store.list_active(user_id=user_id, scope="global", limit=200)
        else:
            # Default retrieval: global memories plus any unrestricted memories without a foreign project_id
            all_mem = self.store.list_active(user_id=user_id, limit=200)
            all_candidates = [m for m in all_mem if not m.project_id or m.scope == "global"]

        if not all_candidates:
            return []

        salient_terms, expanded_terms = self._extract_terms(query)
        if not salient_terms:
            return []

        scored_candidates: List[ScoredMemory] = []
        seen_keys = set()

        for mem in all_candidates:
            # Deduplicate by key if global & project share the same key (project takes precedence)
            mem_unique_key = f"{mem.key}_{mem.scope}_{mem.project_id}"
            if mem_unique_key in seen_keys:
                continue
            seen_keys.add(mem_unique_key)

            score, reason = self._score_memory(mem, salient_terms, expanded_terms, query)
            if score > 0:
                scored_candidates.append(
                    ScoredMemory(
                        memory=mem,
                        score=score,
                        reason=reason,
                        selected=score >= self.RELEVANCE_THRESHOLD,
                    )
                )

        # Sort descending by score
        scored_candidates.sort(key=lambda sm: sm.score, reverse=True)

        selected = [sm for sm in scored_candidates if sm.selected][:top_k]

        # Update last_used_at on the selected memories
        for sm in selected:
            if sm.memory.id is not None:
                self.store.mark_used(sm.memory.id)

        return selected

    def _extract_terms(self, query: str) -> Tuple[set[str], set[str]]:
        """Extract salient keywords and expand them with semantic synonyms."""
        words = re.findall(r"[a-zA-Z0-9]+", query.lower())
        salient = {w for w in words if len(w) > 1 and w not in self._STOPWORDS}

        expanded = set(salient)
        for term in salient:
            if term in self._DOMAIN_SYNONYMS:
                expanded.update(self._DOMAIN_SYNONYMS[term])

        return salient, expanded

    def _score_memory(
        self,
        mem: MemoryItem,
        salient_terms: set[str],
        expanded_terms: set[str],
        query: str,
    ) -> Tuple[float, str]:
        """Score an individual MemoryItem against query terms."""
        key_words = set(re.findall(r"[a-zA-Z0-9]+", mem.key.lower()))
        val_words = set(re.findall(r"[a-zA-Z0-9]+", mem.value.lower()))
        matched_reasons = []

        raw_score = 0.0

        for term in expanded_terms:
            is_direct = term in salient_terms
            term_weight = 1.0 if is_direct else 0.65

            # Match in key (highest signal)
            if term in key_words:
                raw_score += 2.0 * term_weight
                matched_reasons.append(f"key:'{term}'")
            elif len(term) >= 4 and any(kw.startswith(term) for kw in key_words):
                raw_score += 1.2 * term_weight
                matched_reasons.append(f"key_prefix:'{term}'")

            # Match in value
            if term in val_words:
                raw_score += 1.5 * term_weight
                matched_reasons.append(f"val:'{term}'")
            elif len(term) >= 4 and any(vw.startswith(term) for vw in val_words):
                raw_score += 0.8 * term_weight
                matched_reasons.append(f"val_prefix:'{term}'")

        if raw_score <= 0.0:
            return 0.0, "No term overlap."

        # Importance multiplier
        imp_mult = 1.2 if mem.importance == "high" else (1.0 if mem.importance == "medium" else 0.8)
        raw_score *= imp_mult

        # Confidence weight
        raw_score *= max(0.5, mem.confidence)

        # Type relevance bonus (e.g. asking "what UI style do I prefer" boosts 'preference' memories)
        lower_q = query.lower()
        if mem.memory_type == MemoryType.PREFERENCE.value and any(w in lower_q for w in ["prefer", "preference", "favorite", "style", "theme", "ui"]):
            raw_score += 1.5
            matched_reasons.append("preference_match")
        elif mem.memory_type == MemoryType.IDENTITY.value and any(w in lower_q for w in ["name", "who am i", "my name", "identity"]):
            raw_score += 2.0
            matched_reasons.append("identity_match")
        elif mem.memory_type == MemoryType.PROJECT.value and any(w in lower_q for w in ["project", "building", "working on"]):
            raw_score += 1.5
            matched_reasons.append("project_match")

        reason_str = ", ".join(matched_reasons) if matched_reasons else "keyword match"
        return raw_score, reason_str
