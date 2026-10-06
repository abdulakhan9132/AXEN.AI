"""
axen/memory/memory_extractor.py
===============================
Extracts explicit commands and candidate memories from user input.

Distinguishes:
1. Explicit commands (high confidence, direct user instruction)
   - "Remember that I prefer dark UI."
   - "Remember my name is Alex."
   - "Forget that I prefer dark UI."
   - "Update my preference to light UI."
   - "What do you remember about me?"
2. Candidate extractions from conversational declarations
   - "My name is Alex."
   - "I prefer dark mode."
   - "I'm building AXEN AI."
   - Casual statements ("I might move to London someday") assigned LOW confidence.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Tuple

from .memory_models import ConfidenceLevel, ImportanceLevel, MemoryItem, MemoryType


class MemoryExtractor:
    """Extracts explicit memory commands and implicit candidate memories."""

    # Explicit memory commands
    _REMEMBER_EXPLICIT = re.compile(
        r"^(?:please\s+)?(?:remember\s+that|remember\s+this|save\s+that|note\s+that|keep\s+in\s+mind|save\s+to\s+memory)[:\s]+(.+)",
        re.I,
    )
    _REMEMBER_MY = re.compile(
        r"^(?:please\s+)?remember\s+(?:that\s+)?my\s+([a-zA-Z\s_]+?)\s+(?:is|are)\s+(.+)",
        re.I,
    )
    _UPDATE_EXPLICIT = re.compile(
        r"^(?:please\s+)?update\s+my\s+([a-zA-Z\s_]+?)\s+to\s+(.+)",
        re.I,
    )
    _FORGET_EXPLICIT = re.compile(
        r"^(?:please\s+)?(?:forget\s+that|forget\s+about|delete\s+memory|remove\s+memory)[:\s]+(.+)|^forget\s+my\s+([a-zA-Z\s_]+)",
        re.I,
    )
    _RECALL_EXPLICIT = re.compile(
        r"\b(?:what\s+do\s+you\s+remember|show\s+(?:all\s+)?(?:my\s+)?(?:saved\s+)?memories|recall\s+(?:notes|everything|all|memories)|list\s+(?:my\s+)?memories|recall\s+notes)\b",
        re.I,
    )

    # Implicit candidate declarations
    _NAME_DECLARATION = re.compile(
        r"^(?:my\s+name\s+is|i\s+am|i'm|call\s+me)\s+([A-Z][a-zA-Z0-9_\-]{1,25})(?:[\.,!].*)?$",
        re.I,
    )
    _PREFERENCE_DECLARATION = re.compile(
        r"\bi\s+(?:prefer|like|love|favor)\s+([a-zA-Z0-9\s\-]{2,30})\b",
        re.I,
    )
    _PROJECT_DECLARATION = re.compile(
        r"\b(?:i\s+am|i'm)\s+(?:building|working\s+on|developing|creating)\s+([a-zA-Z0-9\s\.\-_]{2,40})\b",
        re.I,
    )
    _SKILL_DECLARATION = re.compile(
        r"\bi\s+(?:use|code\s+in|program\s+in|develop\s+in|work\s+with)\s+([a-zA-Z0-9\+\#\.\s]{2,25})\b",
        re.I,
    )
    _CASUAL_SPECULATIVE = re.compile(
        r"\b(?:might|maybe|possibly|someday|could|perhaps|thinking\s+about)\b",
        re.I,
    )

    @classmethod
    def parse_explicit_command(cls, text: str) -> Optional[dict[str, Any]]:
        """
        Check if input is an explicit memory directive.
        Returns command dict: {"action": "remember"|"forget"|"update"|"recall", ...}
        """
        clean = text.strip()

        # 1. Recall command
        if cls._RECALL_EXPLICIT.search(clean):
            return {"action": "recall"}

        # 2. Update preference/fact: "Update my preference to light UI"
        m_upd = cls._UPDATE_EXPLICIT.match(clean)
        if m_upd:
            raw_key = m_upd.group(1).strip()
            raw_val = m_upd.group(2).strip().rstrip(".,!")
            key, mtype = cls._normalize_key_and_type(raw_key, raw_val)
            return {
                "action": "update",
                "key": key,
                "value": raw_val,
                "memory_type": mtype,
                "importance": ImportanceLevel.HIGH.value,
                "confidence": 1.0,
            }

        # 3. Forget command: "Forget that I prefer dark UI", "Forget my name"
        m_forg = cls._FORGET_EXPLICIT.match(clean)
        if m_forg:
            target = (m_forg.group(1) or m_forg.group(2) or "").strip().rstrip(".,!")
            key, _ = cls._normalize_key_and_type(target, target)
            return {
                "action": "forget",
                "target": target,
                "key": key,
            }

        # 4. Remember my [attr] is [val]: "Remember my name is Alex"
        m_my = cls._REMEMBER_MY.match(clean)
        if m_my:
            raw_key = m_my.group(1).strip()
            raw_val = m_my.group(2).strip().rstrip(".,!")
            key, mtype = cls._normalize_key_and_type(raw_key, raw_val)
            importance = ImportanceLevel.HIGH.value if mtype in ("identity", "instruction") else ImportanceLevel.MEDIUM.value
            return {
                "action": "remember",
                "key": key,
                "value": raw_val,
                "memory_type": mtype,
                "importance": importance,
                "confidence": 1.0,
            }

        # 5. Remember that [fact]: "Remember that I prefer dark UI."
        m_rem = cls._REMEMBER_EXPLICIT.match(clean)
        if m_rem:
            fact = m_rem.group(1).strip().rstrip(".,!")
            key, mtype, clean_val = cls._infer_key_from_fact(fact)
            return {
                "action": "remember",
                "fact": fact,
                "key": key,
                "value": clean_val,
                "memory_type": mtype,
                "importance": ImportanceLevel.HIGH.value if mtype in ("identity", "instruction", "preference") else ImportanceLevel.MEDIUM.value,
                "confidence": 1.0,
            }

        return None

    @classmethod
    def extract_implicit_candidate(cls, text: str) -> Optional[MemoryItem]:
        """
        Extract a potential memory item from a normal conversational statement.
        Applies confidence penalization for casual/speculative phrasing.
        """
        clean = text.strip()
        lower = clean.lower()

        # Check if statement is speculative or casual
        is_speculative = bool(cls._CASUAL_SPECULATIVE.search(clean))
        conf = 0.4 if is_speculative else 0.85

        # 1. Identity declaration: "My name is Alex"
        m_name = cls._NAME_DECLARATION.match(clean)
        if m_name:
            name_val = m_name.group(1).strip()
            # Exclude false positives like "I am happy" or "I am fine"
            if name_val.lower() not in ("happy", "sad", "fine", "good", "okay", "here", "ready", "listening", "trying"):
                return MemoryItem(
                    key="user_name",
                    value=name_val,
                    memory_type=MemoryType.IDENTITY.value,
                    importance=ImportanceLevel.HIGH.value,
                    confidence=0.95 if not is_speculative else 0.4,
                    source="extracted",
                )

        # 2. Preference declaration: "I prefer dark UI" / "I like dark mode"
        m_pref = cls._PREFERENCE_DECLARATION.search(clean)
        if m_pref and not any(neg in lower for neg in ["don't", "do not", "never", "hate"]):
            pref_val = m_pref.group(1).strip().rstrip(".,!")
            # Determine specific preference key
            key = "preference"
            if any(w in pref_val.lower() for w in ["dark", "light", "theme", "ui", "interface", "mode"]):
                key = "ui_theme"
                pref_val = "dark" if "dark" in pref_val.lower() else ("light" if "light" in pref_val.lower() else pref_val)
            elif any(w in pref_val.lower() for w in ["python", "javascript", "rust", "c++", "go"]):
                key = "preferred_language"

            return MemoryItem(
                key=key,
                value=pref_val,
                memory_type=MemoryType.PREFERENCE.value,
                importance=ImportanceLevel.MEDIUM.value if not is_speculative else ImportanceLevel.LOW.value,
                confidence=conf,
                source="extracted",
            )

        # 3. Project declaration: "I'm building AXEN AI"
        m_proj = cls._PROJECT_DECLARATION.search(clean)
        if m_proj:
            proj_val = m_proj.group(1).strip().rstrip(".,!")
            return MemoryItem(
                key="current_project",
                value=proj_val,
                memory_type=MemoryType.PROJECT.value,
                importance=ImportanceLevel.HIGH.value if not is_speculative else ImportanceLevel.LOW.value,
                confidence=conf,
                source="extracted",
            )

        # 4. Skill declaration: "I use Figma"
        m_skill = cls._SKILL_DECLARATION.search(clean)
        if m_skill:
            skill_val = m_skill.group(1).strip().rstrip(".,!")
            return MemoryItem(
                key=f"skill_{skill_val.lower().replace(' ', '_')}",
                value=skill_val,
                memory_type=MemoryType.SKILL.value,
                importance=ImportanceLevel.MEDIUM.value if not is_speculative else ImportanceLevel.LOW.value,
                confidence=conf,
                source="extracted",
            )

        return None

    @classmethod
    def _normalize_key_and_type(cls, raw_key: str, raw_val: str) -> Tuple[str, str]:
        """Normalize user key text into a standard key and MemoryType."""
        k = raw_key.lower().strip()
        v = raw_val.lower().strip()

        if "name" in k:
            return "user_name", MemoryType.IDENTITY.value
        if any(term in k for term in ["theme", "mode", "ui", "interface", "style", "preference"]) or any(term in v for term in ["dark", "light"]):
            return "ui_theme", MemoryType.PREFERENCE.value
        if "project" in k:
            return "current_project", MemoryType.PROJECT.value
        if "skill" in k or "tool" in k:
            return f"tool_{k.replace(' ', '_')}", MemoryType.SKILL.value
        if "goal" in k or "aim" in k:
            return "user_goal", MemoryType.GOAL.value
        if "instruction" in k or "rule" in k:
            return "instruction", MemoryType.INSTRUCTION.value

        clean_key = re.sub(r"[^a-z0-9_]+", "_", k).strip("_")
        return clean_key or "user_fact", MemoryType.FACT.value

    @classmethod
    def _infer_key_from_fact(cls, fact: str) -> Tuple[str, str, str]:
        """Infer an appropriate key, type, and clean value from a freeform fact statement."""
        lower = fact.lower()
        val = fact

        if "name is" in lower or "call me" in lower:
            m = re.search(r"(?:name\s+is|call\s+me)\s+(.+)", fact, re.I)
            clean_v = m.group(1).strip() if m else fact
            return "user_name", MemoryType.IDENTITY.value, clean_v
        if any(term in lower for term in ["dark ui", "light ui", "dark mode", "light mode", "prefer dark", "prefer light", "ui theme", "interface"]):
            clean_v = "dark" if "dark" in lower else ("light" if "light" in lower else fact)
            return "ui_theme", MemoryType.PREFERENCE.value, clean_v
        if "language is" in lower or "prefer python" in lower or "code in" in lower:
            m = re.search(r"(?:prefer\s+|language\s+is\s+|code\s+in\s+)(.+)", fact, re.I)
            clean_v = m.group(1).strip() if m else fact
            return "preferred_language", MemoryType.PREFERENCE.value, clean_v
        if any(term in lower for term in ["building", "working on", "project"]):
            m = re.search(r"(?:building|working\s+on|project\s+is)\s+(.+)", fact, re.I)
            clean_v = m.group(1).strip() if m else fact
            return "current_project", MemoryType.PROJECT.value, clean_v
        if any(term in lower for term in ["keep responses", "concise", "be brief", "always answer"]):
            return "response_style", MemoryType.INSTRUCTION.value, fact
        if any(term in lower for term in ["goal is", "want to learn", "aspire to"]):
            return "user_goal", MemoryType.GOAL.value, fact

        # Generic key: first 3-4 salient words
        words = re.findall(r"[a-z0-9]+", lower)
        stopwords = {"i", "my", "that", "the", "a", "an", "is", "am", "are", "to", "for", "in", "on"}
        salient = [w for w in words if w not in stopwords][:3]
        key = "_".join(salient) if salient else "fact"
        return key, MemoryType.FACT.value, val
