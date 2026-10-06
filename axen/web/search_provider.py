"""
axen/web/search_provider.py
===========================
Provider abstraction for Web Search in AXEN.

Implements:
1. `WebSearchProvider`: Abstract base class decoupling search engine from the Brain.
2. `MockSearchProvider`: High-fidelity, deterministic search provider for offline testing and evaluation.
3. `DuckDuckGoSearchProvider`: Production-grade live search using DuckDuckGo HTML endpoint without external API keys.
"""

from __future__ import annotations

import abc
import json
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from axen.web.web_models import PageContent, SourceType, WebSearchResult
from axen.web.web_policy import WebPolicy, WebSecurityError


class WebSearchProvider(abc.ABC):
    """Abstract interface for all web search backends."""

    def __init__(self, policy: Optional[WebPolicy] = None) -> None:
        self.policy = policy or WebPolicy()

    @abc.abstractmethod
    def search(self, query: str, max_results: int = 5) -> List[WebSearchResult]:
        """Perform a search and return structured results."""
        pass

    @abc.abstractmethod
    def open_page(self, url: str, timeout: Optional[float] = None) -> PageContent:
        """Fetch and extract content from a specific web URL."""
        pass


class MockSearchProvider(WebSearchProvider):
    """
    Deterministic search provider for offline automated tests.
    Allows registering custom responses for queries and URLs.
    """

    def __init__(self, policy: Optional[WebPolicy] = None) -> None:
        super().__init__(policy)
        self.registered_queries: Dict[str, List[WebSearchResult]] = {}
        self.registered_pages: Dict[str, PageContent] = {}
        self.search_call_count: int = 0
        self.page_call_count: int = 0
        self.searched_queries: List[str] = []
        self.opened_urls: List[str] = []

    def register_query(self, query: str, results: List[WebSearchResult]) -> None:
        """Register canned search results for a query."""
        self.registered_queries[query.strip().lower()] = results

    def register_page(self, url: str, content: PageContent) -> None:
        """Register canned page content for a URL."""
        self.registered_pages[url.strip()] = content

    def search(self, query: str, max_results: int = 5) -> List[WebSearchResult]:
        self.policy.check_permission("search")
        self.search_call_count += 1
        self.searched_queries.append(query)

        q_norm = query.strip().lower()
        if q_norm in self.registered_queries:
            return self.registered_queries[q_norm][:max_results]

        # Fuzzy substring match in registered queries
        for k, results in self.registered_queries.items():
            if k in q_norm or q_norm in k:
                return results[:max_results]

        # Keyword overlap match
        q_words = set(re.findall(r"\w+", q_norm))
        best_match = None
        best_overlap = 0
        for k, results in self.registered_queries.items():
            k_words = set(re.findall(r"\w+", k))
            overlap = len(q_words.intersection(k_words))
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = results

        if best_match and best_overlap >= 2:
            return best_match[:max_results]

        # Generic fallback mock result
        return [
            WebSearchResult(
                title=f"Search Result for '{query}'",
                url="https://example.com/result1",
                snippet=f"Information related to {query} from a reliable primary source.",
                domain="example.com",
                source_type=SourceType.SECONDARY,
                relevance_score=0.8,
            )
        ]

    def open_page(self, url: str, timeout: Optional[float] = None) -> PageContent:
        self.policy.check_permission("read")
        self.page_call_count += 1
        self.opened_urls.append(url)

        # Validate URL via security policy
        valid, err = self.policy.validate_url(url)
        if not valid:
            raise WebSecurityError(err or f"Invalid URL '{url}'")

        if url in self.registered_pages:
            content = self.registered_pages[url]
            # Check prompt injection sanitization
            sanitized_text, has_inj, _ = self.policy.sanitize_untrusted_text(content.text)
            content.text = sanitized_text
            return content

        # Generic default mock page
        from urllib.parse import urlparse
        domain = urlparse(url).netloc or "example.com"
        return PageContent(
            url=url,
            title=f"Sample Page from {domain}",
            domain=domain,
            text=f"This is mock content retrieved from {url}. All operations are verified.",
            source_type=SourceType.SECONDARY,
        )


class DuckDuckGoSearchProvider(WebSearchProvider):
    """
    Standard HTTP-based search provider connecting to DuckDuckGo's HTML endpoint.
    Operates without requiring API keys, user tracking, or cloud credentials.
    """

    SEARCH_URL = "https://html.duckduckgo.com/html/"
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 AXEN/1.0"

    def search(self, query: str, max_results: int = 5) -> List[WebSearchResult]:
        self.policy.check_permission("search")
        data = urllib.parse.urlencode({"q": query}).encode("utf-8")
        req = urllib.request.Request(
            self.SEARCH_URL,
            data=data,
            headers={
                "User-Agent": self.USER_AGENT,
                "Accept": "text/html,application/xhtml+xml",
            },
        )

        results: List[WebSearchResult] = []
        timeout = self.policy.request_timeout

        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                html = response.read().decode("utf-8", errors="replace")
                results = self._parse_ddg_html(html, max_results)
        except Exception:
            # Fallback to direct lite search or return empty on network error
            try:
                lite_url = f"https://lite.duckduckgo.com/lite/?q={urllib.parse.quote(query)}"
                lite_req = urllib.request.Request(lite_url, headers={"User-Agent": self.USER_AGENT})
                with urllib.request.urlopen(lite_req, timeout=timeout) as resp:
                    html = resp.read().decode("utf-8", errors="replace")
                    results = self._parse_ddg_html(html, max_results)
            except Exception:
                return []

        return results[:max_results]

    def open_page(self, url: str, timeout: Optional[float] = None) -> PageContent:
        self.policy.check_permission("read")
        valid, err = self.policy.validate_url(url)
        if not valid:
            raise WebSecurityError(err or f"Prohibited URL '{url}'")

        from axen.web.page_reader import PageReader
        reader = PageReader(policy=self.policy)
        return reader.read_page(url, timeout=timeout)

    def _parse_ddg_html(self, html: str, max_results: int) -> List[WebSearchResult]:
        """Extract structured search results from DuckDuckGo HTML response."""
        results: List[WebSearchResult] = []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            # DuckDuckGo HTML structure has result elements with class 'result'
            for res_div in soup.find_all("div", class_=re.compile(r"result\b")):
                a_tag = res_div.find("a", class_=re.compile(r"result__url|result__snippet|result__a"))
                if not a_tag or not a_tag.get("href"):
                    continue

                raw_href = a_tag["href"]
                # DDG uses redirect links like /l/?kh=-1&uddg=https%3A%2F%2F...
                parsed_href = raw_href
                if "uddg=" in raw_href:
                    match = re.search(r"uddg=([^&]+)", raw_href)
                    if match:
                        parsed_href = urllib.parse.unquote(match.group(1))

                title = a_tag.get_text(strip=True)
                snippet_tag = res_div.find(class_=re.compile(r"result__snippet"))
                snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""

                domain = urllib.parse.urlparse(parsed_href).netloc.lower()
                results.append(
                    WebSearchResult(
                        title=title or domain,
                        url=parsed_href,
                        snippet=snippet,
                        domain=domain,
                        source_type=self._classify_domain(domain),
                    )
                )
                if len(results) >= max_results:
                    break
        except Exception:
            # Simple regex fallback if BS4 fails
            pattern = re.compile(r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.IGNORECASE)
            for m in pattern.finditer(html):
                raw_href, raw_title = m.groups()
                clean_title = re.sub(r"<[^>]+>", "", raw_title).strip()
                domain = urllib.parse.urlparse(raw_href).netloc.lower()
                results.append(
                    WebSearchResult(
                        title=clean_title or domain,
                        url=raw_href,
                        snippet="",
                        domain=domain,
                        source_type=self._classify_domain(domain),
                    )
                )
                if len(results) >= max_results:
                    break

        return results

    @staticmethod
    def _classify_domain(domain: str) -> SourceType:
        """Infer source authority from domain suffix or organization."""
        d = domain.lower()
        if any(d.endswith(sfx) for sfx in (".gov", ".edu", ".org", "docs.python.org", "github.com", "arxiv.org", "wikipedia.org")):
            return SourceType.PRIMARY
        if any(k in d for k in ("documentation", "official", "developer", "nvidia.com", "microsoft.com", "apple.com", "google.com")):
            return SourceType.PRIMARY
        if any(d.endswith(sfx) for sfx in ("reuters.com", "bbc.com", "apnews.com", "bloomberg.com", "nature.com")):
            return SourceType.SECONDARY
        return SourceType.UNKNOWN
