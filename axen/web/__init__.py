"""
axen/web/__init__.py
====================
Public exports for AXEN Web Agent module.
"""

from axen.web.citation_manager import CitationManager
from axen.web.page_reader import PageReader
from axen.web.search_provider import DuckDuckGoSearchProvider, MockSearchProvider, WebSearchProvider
from axen.web.source_manager import SourceManager
from axen.web.web_agent import WebAgent
from axen.web.web_cache import WebCache
from axen.web.web_models import (
    Citation,
    ClaimEvidence,
    ClaimStatus,
    PageContent,
    ResearchDepth,
    SourceType,
    WebAgentResult,
    WebPermissionLevel,
    WebSearchQuery,
    WebSearchResult,
)
from axen.web.web_policy import WebPolicy, WebSecurityError

__all__ = [
    "WebAgent",
    "WebSearchProvider",
    "MockSearchProvider",
    "DuckDuckGoSearchProvider",
    "PageReader",
    "SourceManager",
    "CitationManager",
    "WebCache",
    "WebPolicy",
    "WebSecurityError",
    "WebPermissionLevel",
    "ResearchDepth",
    "SourceType",
    "ClaimStatus",
    "WebSearchQuery",
    "WebSearchResult",
    "PageContent",
    "Citation",
    "ClaimEvidence",
    "WebAgentResult",
]
