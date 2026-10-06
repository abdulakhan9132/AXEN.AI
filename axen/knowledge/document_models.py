"""
axen/knowledge/document_models.py
=================================
Data models and schemas for AXEN Phase 4 Local Knowledge & RAG System.

Defines:
- DocumentType: supported file formats
- Document: imported document with metadata and content hash
- DocumentChunk: structure-aware text chunk with source location
- ScoredChunk: retrieval candidate with semantic, lexical, and hybrid scores
- CollectionInfo: collection metadata
- GroundingStatus & GroundingResult: verification of answer against sources
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, List, Optional


class DocumentType(str, Enum):
    TXT = "txt"
    MARKDOWN = "md"
    PDF = "pdf"
    DOCX = "docx"
    CSV = "csv"
    JSON = "json"
    CODE = "code"
    OTHER = "other"


class GroundingStatus(str, Enum):
    GROUNDED = "GROUNDED"
    PARTIALLY_GROUNDED = "PARTIALLY_GROUNDED"
    UNSUPPORTED = "UNSUPPORTED"
    NO_SOURCE = "NO_SOURCE"


def compute_content_hash(text_or_bytes: str | bytes) -> str:
    """Compute deterministic SHA-256 content hash."""
    if isinstance(text_or_bytes, str):
        data = text_or_bytes.encode("utf-8")
    else:
        data = text_or_bytes
    return hashlib.sha256(data).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Document:
    """An ingested knowledge document."""
    document_id: str
    filename: str
    filepath: str
    file_type: DocumentType
    file_size: int
    content_hash: str
    collection_id: str = "default"
    title: Optional[str] = None
    created_at: str = field(default_factory=_now_iso)
    modified_at: str = field(default_factory=_now_iso)
    indexed_at: str = field(default_factory=_now_iso)
    chunk_count: int = 0
    status: str = "indexed"  # indexed, modified, deleted, error
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["file_type"] = self.file_type.value if hasattr(self.file_type, "value") else str(self.file_type)
        return d


@dataclass
class DocumentChunk:
    """A bounded, structure-aware passage from a document."""
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    collection_id: str = "default"
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    source_location: Optional[str] = None  # e.g. "Section: Agent Loop", "Page 3", "Function: run"
    token_estimate: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.token_estimate:
            self.token_estimate = max(1, len(self.text) // 4)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoredChunk:
    """A retrieved chunk with hybrid scoring and source attribution."""
    chunk: DocumentChunk
    score: float
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    selected: bool = False
    citation: str = ""

    def __post_init__(self):
        if not self.citation:
            loc = f" — {self.chunk.source_location}" if self.chunk.source_location else ""
            self.citation = f"{self.chunk.metadata.get('filename', 'doc')}{loc}"

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def content(self) -> str:
        return self.chunk.text

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "document_id": self.chunk.document_id,
            "text": self.chunk.text,
            "score": round(self.score, 4),
            "semantic_score": round(self.semantic_score, 4),
            "lexical_score": round(self.lexical_score, 4),
            "selected": self.selected,
            "citation": self.citation,
            "source_location": self.chunk.source_location,
            "page_number": self.chunk.page_number,
            "section_title": self.chunk.section_title,
        }


@dataclass
class CollectionInfo:
    """Metadata for a logical knowledge collection."""
    collection_id: str
    name: str
    description: str = ""
    created_at: str = field(default_factory=_now_iso)
    document_count: int = 0
    chunk_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GroundingResult:
    """Evaluation result verifying answer claims against retrieved sources."""
    status: GroundingStatus
    grounded_ratio: float
    supported_claims: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    sources_used: list[str] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "grounded_ratio": round(self.grounded_ratio, 2),
            "supported_claims": self.supported_claims,
            "unsupported_claims": self.unsupported_claims,
            "sources_used": self.sources_used,
            "detail": self.detail,
        }
