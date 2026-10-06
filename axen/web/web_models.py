"""
axen/web/web_models.py
======================
Core data models for the AXEN Web Agent.

Defines types for search queries, search results, extracted page contents,
claims, evidence, citations, research depths, task types, and permissions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class WebPermissionLevel(str, Enum):
    """Permission levels for Web operations."""
    WEB_DISABLED = "web_disabled"          # No network access allowed
    WEB_READ = "web_read"                  # Fetch single specified page only
    WEB_RESEARCH = "web_research"          # Full search, multi-source read, citation
    WEB_INTERACTION = "web_interaction"    # Interactive browser (disabled in Phase 1)


class ResearchDepth(str, Enum):
    """Depth levels for web research tasks."""
    QUICK = "quick"          # 1-2 sources, fast summary
    STANDARD = "standard"    # 3-4 sources, comparative verification
    DEEP = "deep"            # 5+ sources, thorough cross-check and synthesis


class SourceType(str, Enum):
    """Authority classification of a source."""
    PRIMARY = "primary"        # Official documentation, original research, official release
    SECONDARY = "secondary"    # Reputable journalism, technical overview, curated guide
    TERTIARY = "tertiary"      # Forums, blogs, community posts
    UNKNOWN = "unknown"        # Unclassified source


class ClaimStatus(str, Enum):
    """Grounding status of a synthesized claim."""
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONFLICTING = "conflicting"
    UNSUPPORTED = "unsupported"


@dataclass
class WebSearchQuery:
    """A structured search request."""
    query: str
    max_results: int = 5
    preferred_domains: List[str] = field(default_factory=list)
    depth: ResearchDepth = ResearchDepth.STANDARD


@dataclass
class WebSearchResult:
    """A single structured search result returned by a search provider."""
    title: str
    url: str
    snippet: str
    domain: str = ""
    published_at: Optional[str] = None
    source_type: SourceType = SourceType.UNKNOWN
    relevance_score: float = 0.0

    def __post_init__(self) -> None:
        if not self.domain and "://" in self.url:
            try:
                from urllib.parse import urlparse
                self.domain = urlparse(self.url).netloc.lower()
            except Exception:
                self.domain = ""


@dataclass
class PageContent:
    """Extracted and cleaned content of a web page."""
    url: str
    title: str
    domain: str
    text: str
    headings: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    published_at: Optional[str] = None
    source_type: SourceType = SourceType.UNKNOWN
    retrieved_at: float = field(default_factory=time.time)
    content_hash: str = ""
    is_truncated: bool = False
    status_code: int = 200
    error: Optional[str] = None

    @property
    def is_success(self) -> bool:
        return self.error is None and bool(self.text)


@dataclass
class Citation:
    """A traceable, verifiable reference to a web source."""
    index: int
    title: str
    url: str
    domain: str
    published_at: Optional[str] = None
    source_type: SourceType = SourceType.UNKNOWN
    quoted_snippet: str = ""

    def format_citation(self) -> str:
        """Format citation for display in AXEN responses."""
        date_str = f" ({self.published_at})" if self.published_at else ""
        return f"[{self.index}] {self.domain} — {self.title}{date_str}\n    URL: {self.url}"


@dataclass
class ClaimEvidence:
    """An extracted claim linked to supporting or conflicting evidence."""
    claim: str
    evidence_text: str
    source_url: str
    source_domain: str
    source_type: SourceType = SourceType.UNKNOWN
    status: ClaimStatus = ClaimStatus.SUPPORTED
    confidence: float = 1.0


@dataclass
class WebAgentResult:
    """Complete structured output from a Web Agent execution."""
    success: bool
    summary: str
    task_type: str = "web_research"
    depth: ResearchDepth = ResearchDepth.STANDARD
    searches_performed: int = 0
    pages_read: int = 0
    sources_selected: List[PageContent] = field(default_factory=list)
    sources_rejected: int = 0
    citations: List[Citation] = field(default_factory=list)
    claims: List[ClaimEvidence] = field(default_factory=list)
    conflicts_detected: List[str] = field(default_factory=list)
    is_grounded: bool = True
    grounding_detail: str = "PASS"
    duration_s: float = 0.0
    debug_block: str = ""
    error: Optional[str] = None
