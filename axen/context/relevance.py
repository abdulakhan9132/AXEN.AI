"""
axen/context/relevance.py
=========================
Fast-path classification and multi-factor relevance scoring for AXEN Context Engine.

Features:
- Simple Query Fast Path: Completely halts memory retrieval on greetings, basic math,
  conversions, or syntax execution.
- Scope Rules:
    * Global Memory: cross-project unless disabled.
    * Project Memory: available ONLY inside the matching project.
    * Conversation: available ONLY inside current conversation.
- Multi-factor scoring with configurable weights:
    score = (
        semantic_overlap * weight_overlap
        + scope_match * weight_scope
        + importance_factor * weight_importance
        + confidence_factor * weight_confidence
        + recency_factor * weight_recency
    )
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from axen.memory.memory_models import MemoryItem, ScoredMemory


@dataclass
class ScoredContextItem:
    """A contextual memory candidate with transparent score decomposition."""
    item: MemoryItem
    total_score: float
    relevance_overlap: float
    scope_bonus: float
    importance_bonus: float
    confidence_bonus: float
    reason: str
    selected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.item.key,
            "value": self.item.value,
            "scope": getattr(self.item, "scope", "global"),
            "project_id": getattr(self.item, "project_id", None),
            "type": self.item.memory_type,
            "score": round(self.total_score, 3),
            "reason": self.reason,
            "selected": self.selected,
        }


class RelevanceFilter:
    """Determines relevance of memories and whether queries qualify for fast-path bypass."""

    RELEVANCE_THRESHOLD = 1.35

    _GREETING_PATTERN = re.compile(
        r"^(?:hello|hi|hey|greetings|good\s+(?:morning|afternoon|evening|day)|howdy|sup)(?:[\s,]+(?:there|axen|friend|everyone|everybody))?[\s\.\!\?]*$|"
        r"^(?:how\s+are\s+you(?:\s+doing|\s+today)?|what'?s\s+up(?:\s+axen|\s+there)?|how's\s+it\s+going)[\s\.\!\?]*$",
        re.I,
    )

    _POLITENESS_PATTERN = re.compile(
        r"^(?:thanks|thank\s+you|thx|cheers|ok|okay|cool|great|bye|goodbye|see\s+ya)[\s\.\!\?]*$",
        re.I,
    )

    _SIMPLE_MATH_PATTERN = re.compile(
        r"^(?:what\s+is|what's|calculate|compute|solve)?\s*[\d\s\+\-\*\/\(\)\^\.%=]+[?\.]*$",
        re.I,
    )

    _STOPWORDS = frozenset({
        "a", "an", "the", "is", "am", "are", "was", "were", "do", "does", "did",
        "you", "your", "my", "me", "i", "what", "which", "who", "whom",
        "this", "that", "these", "those", "have", "has", "had", "for", "to",
        "in", "on", "at", "by", "from", "and", "or", "but", "about", "with",
        "can", "could", "would", "should", "will", "tell", "show", "give", "please",
        "hi", "hello", "hey", "howdy", "sup", "greetings", "good", "morning",
        "afternoon", "evening", "thanks", "thank", "ok", "okay", "yes", "no",
    })

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
        "model": {"tinyllama", "llm", "ai", "weights", "inference"},
    }

    def __init__(
        self,
        weight_overlap: float = 1.0,
        weight_scope: float = 0.5,
        weight_importance: float = 0.3,
        weight_confidence: float = 0.2,
    ) -> None:
        self.weight_overlap = weight_overlap
        self.weight_scope = weight_scope
        self.weight_importance = weight_importance
        self.weight_confidence = weight_confidence

    def is_fast_path(self, query: str, task_type: str = "conversation") -> bool:
        """
        Check if query is a simple greeting, thank you, or pure arithmetic calculation.
        These MUST bypass expensive memory retrieval to prevent prompt contamination.
        """
        clean = query.strip()
        lower = clean.lower()

        # Greetings & farewells
        if self._GREETING_PATTERN.match(clean) or self._POLITENESS_PATTERN.match(clean):
            return True

        # Simple arithmetic calculations without personal query keywords
        personal_terms = ("remember", "prefer", "my", "name", "project", "note", "save", "what do you know", "instructions")
        if self._SIMPLE_MATH_PATTERN.match(clean) and not any(pt in lower for pt in personal_terms):
            return True

        # Pure deterministic tasks
        if task_type in ("calculation", "computer_systems", "math") and not any(pt in lower for pt in personal_terms):
            return True

        return False

    def is_memory_needed(self, query: str, task_type: str = "conversation") -> bool:
        """Check if memory retrieval is appropriate for the query."""
        if self.is_fast_path(query, task_type=task_type):
            return False
        return True

    def extract_salient_terms(self, query: str) -> Tuple[Set[str], Set[str]]:
        """Extract salient keywords and expanded semantic synonyms."""
        words = re.findall(r"[a-zA-Z0-9]+", query.lower())
        salient = {w for w in words if len(w) > 1 and w not in self._STOPWORDS}

        expanded = set(salient)
        for term in salient:
            if term in self._DOMAIN_SYNONYMS:
                expanded.update(self._DOMAIN_SYNONYMS[term])

        return salient, expanded

    def score_memory(
        self,
        item: MemoryItem,
        salient_terms: Set[str],
        expanded_terms: Set[str],
        query: str,
        current_project_id: Optional[str] = None,
    ) -> ScoredContextItem:
        """
        Score a memory candidate against query terms and active scope.
        Strict scope isolation rules apply:
        - If item.project_id is set and does not match current_project_id -> score 0.0.
        """
        item_proj = getattr(item, "project_id", None)
        item_scope = getattr(item, "scope", "global")

        # Cross-project leakage barrier: NEVER score another project's memory
        if item_proj and current_project_id and item_proj != current_project_id:
            return ScoredContextItem(
                item=item,
                total_score=0.0,
                relevance_overlap=0.0,
                scope_bonus=0.0,
                importance_bonus=0.0,
                confidence_bonus=0.0,
                reason="Cross-project memory blocked",
                selected=False,
            )

        # If memory belongs to a project but we are currently in global scope without a project
        if item_proj and not current_project_id and item_scope == "project":
            return ScoredContextItem(
                item=item,
                total_score=0.0,
                relevance_overlap=0.0,
                scope_bonus=0.0,
                importance_bonus=0.0,
                confidence_bonus=0.0,
                reason="Project-scoped memory excluded from global context",
                selected=False,
            )

        key_words = set(re.findall(r"[a-zA-Z0-9]+", item.key.lower()))
        val_words = set(re.findall(r"[a-zA-Z0-9]+", item.value.lower()))
        matched_reasons = []

        overlap_score = 0.0
        for term in expanded_terms:
            is_direct = term in salient_terms
            term_weight = 1.0 if is_direct else 0.65

            if term in key_words:
                overlap_score += 2.0 * term_weight
                matched_reasons.append(f"key:'{term}'")
            elif len(term) >= 4 and any(kw.startswith(term) for kw in key_words):
                overlap_score += 1.2 * term_weight
                matched_reasons.append(f"key_prefix:'{term}'")

            if term in val_words:
                overlap_score += 1.5 * term_weight
                matched_reasons.append(f"val:'{term}'")
            elif len(term) >= 4 and any(vw.startswith(term) for vw in val_words):
                overlap_score += 0.8 * term_weight
                matched_reasons.append(f"val_prefix:'{term}'")

        # Direct phrase match in key or value
        clean_q = query.strip().lower()
        if item.key in clean_q or any(tok in clean_q for tok in item.key.split("_") if len(tok) > 3):
            overlap_score += 1.5
            matched_reasons.append("phrase_match")

        if overlap_score <= 0.0:
            # Special check for general "what do you know" or "what is my name"
            if item.memory_type == "identity" and any(w in clean_q for w in ["name", "who am i"]):
                overlap_score = 2.0
                matched_reasons.append("identity_inquiry")
            elif item.memory_type in ("preference", "instruction") and any(w in clean_q for w in ["prefer", "preference", "how should you answer"]):
                overlap_score = 2.0
                matched_reasons.append("preference_inquiry")
            elif item_proj and current_project_id == item_proj and any(w in clean_q for w in ["model", "project", "what model"]):
                overlap_score = 2.0
                matched_reasons.append("project_memory_inquiry")
            else:
                return ScoredContextItem(
                    item=item,
                    total_score=0.0,
                    relevance_overlap=0.0,
                    scope_bonus=0.0,
                    importance_bonus=0.0,
                    confidence_bonus=0.0,
                    reason="No term overlap",
                    selected=False,
                )

        # Scope bonus
        scope_bonus = 0.0
        if current_project_id and item_proj == current_project_id:
            scope_bonus = 0.5 * self.weight_scope
            matched_reasons.append("project_match")
        elif item_scope == "global":
            scope_bonus = 0.2 * self.weight_scope

        # Importance factor
        imp = getattr(item, "importance", "medium")
        imp_factor = 1.2 if imp == "high" else (1.0 if imp == "medium" else 0.8)
        importance_bonus = (imp_factor - 1.0) * self.weight_importance

        # Confidence factor
        conf = float(getattr(item, "confidence", 1.0))
        confidence_bonus = (conf - 0.5) * self.weight_confidence

        total = (overlap_score * self.weight_overlap) + scope_bonus + importance_bonus + confidence_bonus
        reason = ", ".join(matched_reasons) if matched_reasons else "keyword match"

        return ScoredContextItem(
            item=item,
            total_score=total,
            relevance_overlap=overlap_score,
            scope_bonus=scope_bonus,
            importance_bonus=importance_bonus,
            confidence_bonus=confidence_bonus,
            reason=reason,
            selected=total >= self.RELEVANCE_THRESHOLD,
        )
