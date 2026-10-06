"""
axen/memory/memory_policy.py
============================
Safety and privacy policy for AXEN Phase 3 Memory System.

Enforces:
1. Blocked storage of secrets (API keys, passwords, bearer tokens, credit cards).
2. Content sanitization and validation.
3. Length and size bounds.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple


class MemoryPolicy:
    """Security and sanitization policy for persistent memories."""

    # Sensitive patterns that must NEVER be persisted
    _SECRET_PATTERNS = [
        # API Keys (e.g. OpenAI, AWS, GitHub, Generic)
        (re.compile(r"\b(?:sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{30,}|AKIA[0-9A-Z]{16})\b"), "API key / token"),
        (re.compile(r"\b(?:api[_-]?key|secret[_-]?key|auth[_-]?token|bearer\s+[a-zA-Z0-9\._\-]{20,})\b", re.I), "Authentication secret"),
        # Passwords / Pins
        (re.compile(r"\b(?:my\s+password\s+is|password:\s*|pwd:\s*|passcode:\s*)\s*(\S+)", re.I), "Password / passcode"),
        # Credit Card Numbers (13 to 19 digits)
        (re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"), "Credit card number"),
        # Social Security / National ID numbers
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "Social security number"),
        # Private Keys
        (re.compile(r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----"), "Private cryptographic key"),
    ]

    MAX_KEY_LENGTH = 100
    MAX_VALUE_LENGTH = 1000

    @classmethod
    def validate_for_storage(cls, key: str, value: str) -> Tuple[bool, Optional[str]]:
        """
        Validate whether a candidate key-value pair is safe for storage.
        Returns (is_allowed, rejection_reason).
        """
        if re.search(r"(?:^|[_\W])(password|passwd|passcode|secret_key|api_key|auth_token|token|credential)(?:[_\W]|$)", key, re.I):
            return False, "Blocked: Key indicates confidential credential."

        combined = f"{key} {value}"

        for pattern, desc in cls._SECRET_PATTERNS:
            if pattern.search(combined):
                return False, f"Blocked: Contains potentially sensitive content ({desc})."

        if len(key.strip()) == 0:
            return False, "Blocked: Memory key cannot be empty."

        if len(value.strip()) == 0:
            return False, "Blocked: Memory value cannot be empty."

        if len(key) > cls.MAX_KEY_LENGTH:
            return False, f"Blocked: Memory key exceeds maximum length ({cls.MAX_KEY_LENGTH} chars)."

        if len(value) > cls.MAX_VALUE_LENGTH:
            return False, f"Blocked: Memory value exceeds maximum length ({cls.MAX_VALUE_LENGTH} chars)."

        return True, None

    @classmethod
    def is_secret(cls, text: str) -> bool:
        """Check if raw text contains any secret or sensitive credentials."""
        for pattern, _ in cls._SECRET_PATTERNS:
            if pattern.search(text):
                return True
        return False

    @classmethod
    def validate_candidate(cls, value: str, memory_type: Any = None, key: str = "") -> Tuple[bool, Optional[str]]:
        """Validate whether a memory candidate satisfies security policy."""
        ok, reason = cls.validate_for_storage(key or "item", value)
        if not ok:
            return False, f"Security Policy: {reason}"
        return True, None
