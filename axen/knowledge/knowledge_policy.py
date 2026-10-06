"""
axen/knowledge/knowledge_policy.py
==================================
Security, privacy, and sensitive file filter for AXEN Knowledge & RAG System.

Enforces:
1. Blocked indexing of secret / sensitive files (.env, private keys, credential files).
2. Blocked system directories (C:\\Windows, System32, .git).
3. Local-only validation (zero external network transmissions).
4. Content sanity bounds (file size limits to prevent OOM).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Tuple


class KnowledgePolicy:
    """Security policy governing document ingestion."""

    MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB limit per file

    # Blocked sensitive filename patterns
    _BLOCKED_FILENAME_PATTERNS = [
        re.compile(r"^\.env(?:\..*)?$", re.IGNORECASE),
        re.compile(r"^id_(?:rsa|dsa|ecdsa|ed25519)(?:\.pub)?$", re.IGNORECASE),
        re.compile(r".*\.(?:key|pem|p12|pfx|pkcs12|kdbx|crt|cer)$", re.IGNORECASE),
        re.compile(r"^(?:credentials|secrets|passwords?|shadow|passwd)(?:\.json|\.yaml|\.yml|\.txt)?$", re.IGNORECASE),
        re.compile(r".*auth[_-]?token.*", re.IGNORECASE),
    ]

    # Blocked directories
    _BLOCKED_DIRECTORIES = {
        ".git", ".svn", ".hg", "__pycache__", ".idea", ".vscode",
        "node_modules", ".venv", "venv", "System32", "Windows",
    }

    # Secret content patterns inside text
    _SECRET_CONTENT_PATTERNS = [
        re.compile(r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----"),
        re.compile(r"\b(?:sk-[a-zA-Z0-9]{24,}|ghp_[a-zA-Z0-9]{30,}|AKIA[0-9A-Z]{16})\b"),
    ]

    @classmethod
    def validate_file_for_ingestion(cls, file_path: Path | str) -> Tuple[bool, Optional[str]]:
        """
        Check if a file is safe for ingestion.
        Returns (is_allowed, rejection_reason).
        """
        path = Path(file_path).resolve()

        if not path.exists():
            return False, f"File does not exist: {path}"

        if path.is_dir():
            return False, f"Path is a directory, not a file: {path}"

        # 1. Directory safety check
        for part in path.parts:
            if part in cls._BLOCKED_DIRECTORIES:
                return False, f"Blocked: Located in restricted directory ({part})."

        # 2. Filename safety check
        filename = path.name
        for pattern in cls._BLOCKED_FILENAME_PATTERNS:
            if pattern.match(filename):
                return False, f"Blocked: Filename matches sensitive file policy ({filename})."

        # 3. File size check
        try:
            size = path.stat().st_size
            if size > cls.MAX_FILE_SIZE_BYTES:
                return False, f"Blocked: File size ({size} bytes) exceeds limit ({cls.MAX_FILE_SIZE_BYTES} bytes)."
            if size == 0:
                return False, "Blocked: File is empty (0 bytes)."
        except Exception as exc:
            return False, f"Could not inspect file metadata: {exc}"

        return True, None

    @classmethod
    def sanitize_text(cls, text: str) -> Tuple[bool, str, Optional[str]]:
        """
        Inspect text for embedded private keys or high-risk credentials.
        Returns (is_allowed, sanitized_text, warning).
        """
        for pattern in cls._SECRET_CONTENT_PATTERNS:
            if pattern.search(text):
                return False, "", "Blocked: Document content contains high-risk credentials or private keys."
        return True, text, None
