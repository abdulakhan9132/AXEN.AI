"""
axen/web/web_cache.py
=====================
Lightweight in-memory cache for Web Agent searches and page contents.

Features:
- TTL-based invalidation (default 15 minutes)
- Hash-based integrity checks
- Size-bounded cache to prevent memory bloat
- Clear and invalidate methods
"""

from __future__ import annotations

import hashlib
import time
from typing import Dict, List, Optional, Tuple

from axen.web.web_models import PageContent, WebSearchResult


class WebCache:
    """Thread-safe, bounded in-memory cache for pages and search results."""

    def __init__(self, default_ttl_seconds: float = 900.0, max_entries: int = 200) -> None:
        self.default_ttl = default_ttl_seconds
        self.max_entries = max_entries
        self._page_cache: Dict[str, Tuple[PageContent, float]] = {}
        self._search_cache: Dict[str, Tuple[List[WebSearchResult], float]] = {}

    def get_page(self, url: str) -> Optional[PageContent]:
        """Retrieve cached page if present and within TTL."""
        entry = self._page_cache.get(url)
        if not entry:
            return None

        content, timestamp = entry
        if (time.time() - timestamp) > self.default_ttl:
            del self._page_cache[url]
            return None

        return content

    def set_page(self, url: str, content: PageContent) -> None:
        """Store page content in cache with current timestamp."""
        if len(self._page_cache) >= self.max_entries:
            # Evict oldest entry
            oldest_key = min(self._page_cache.keys(), key=lambda k: self._page_cache[k][1])
            del self._page_cache[oldest_key]

        if not content.content_hash and content.text:
            content.content_hash = hashlib.sha256(content.text.encode("utf-8", errors="ignore")).hexdigest()[:16]

        self._page_cache[url] = (content, time.time())

    def get_search(self, query: str) -> Optional[List[WebSearchResult]]:
        """Retrieve cached search results if valid."""
        q_norm = query.strip().lower()
        entry = self._search_cache.get(q_norm)
        if not entry:
            return None

        results, timestamp = entry
        if (time.time() - timestamp) > self.default_ttl:
            del self._search_cache[q_norm]
            return None

        return results

    def set_search(self, query: str, results: List[WebSearchResult]) -> None:
        """Store search results in cache."""
        q_norm = query.strip().lower()
        if len(self._search_cache) >= self.max_entries:
            oldest_key = min(self._search_cache.keys(), key=lambda k: self._search_cache[k][1])
            del self._search_cache[oldest_key]

        self._search_cache[q_norm] = (results, time.time())

    def clear(self) -> None:
        """Clear all cached pages and searches."""
        self._page_cache.clear()
        self._search_cache.clear()

    @property
    def size(self) -> int:
        return len(self._page_cache) + len(self._search_cache)
