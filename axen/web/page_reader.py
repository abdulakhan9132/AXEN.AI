"""
axen/web/page_reader.py
=======================
Safe web page retrieval and content extraction.

Key Features:
- Strips script, style, nav, footer, header, and advertising elements.
- Extracts clean text, headings (H1-H3), and internal/external hyperlinks.
- Automatically bounds content to `max_content_size` (default 15,000 characters).
- Sanitizes untrusted text to detect prompt injection attempts.
- Enforces strict SSRF and URL validation before any network request.
"""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
from typing import List, Optional

from axen.web.web_models import PageContent, SourceType
from axen.web.web_policy import WebPolicy, WebSecurityError


class PageReader:
    """Safe page reader and text extraction engine."""

    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 AXEN/1.0"

    def __init__(self, policy: Optional[WebPolicy] = None) -> None:
        self.policy = policy or WebPolicy()

    def read_page(self, url: str, timeout: Optional[float] = None) -> PageContent:
        """
        Fetch HTML from URL, validate safety, strip boilerplate, and return PageContent.
        """
        valid, err = self.policy.validate_url(url)
        if not valid:
            raise WebSecurityError(err or f"Prohibited URL: {url}")

        t_out = timeout or self.policy.request_timeout
        domain = urllib.parse.urlparse(url).netloc.lower()

        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": self.USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml,text/plain",
                },
            )
            with urllib.request.urlopen(req, timeout=t_out) as response:
                content_type = response.headers.get("Content-Type", "")
                if any(x in content_type for x in ("application/zip", "application/octet-stream", "application/x-msdownload")):
                    raise WebSecurityError("Refusing to download binary file stream.")

                html_bytes = response.read(self.policy.max_content_size * 2)
                charset = response.headers.get_content_charset() or "utf-8"
                html_text = html_bytes.decode(charset, errors="replace")
                status_code = getattr(response, "status", 200)

                return self.extract_content(url, html_text, status_code=status_code)
        except Exception as exc:
            return PageContent(
                url=url,
                title="",
                domain=domain,
                text="",
                status_code=500,
                error=str(exc),
            )

    def extract_content(self, url: str, html: str, status_code: int = 200) -> PageContent:
        """
        Parse raw HTML string and extract clean structured text, headings, and links.
        """
        domain = urllib.parse.urlparse(url).netloc.lower()
        title = ""
        headings: List[str] = []
        links: List[str] = []
        text = ""

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")

            # Extract title
            title_tag = soup.find("title")
            if title_tag and title_tag.string:
                title = title_tag.string.strip()

            # Remove noise: scripts, styles, forms, headers, footers, navs
            for noisy in soup.find_all(["script", "style", "nav", "footer", "header", "noscript", "aside", "form", "svg"]):
                noisy.decompose()

            # Extract headings
            for h in soup.find_all(["h1", "h2", "h3"]):
                h_text = h.get_text(strip=True)
                if h_text and len(h_text) > 3:
                    headings.append(h_text)

            # Extract links
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if href and not href.startswith(("#", "javascript:", "mailto:")):
                    full_url = urllib.parse.urljoin(url, href)
                    links.append(full_url)

            # Extract clean main text
            text = soup.get_text(separator="\n", strip=True)
            # Collapse excess whitespace
            text = re.sub(r"\n{3,}", "\n\n", text)
        except Exception:
            # Fallback regex extraction if BS4 unavailable
            title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
            if title_match:
                title = re.sub(r"<[^>]+>", "", title_match.group(1)).strip()

            clean = re.sub(r"<(script|style|nav|footer|header)[^>]*>.*?</\1>", "", html, flags=re.I | re.S)
            clean = re.sub(r"<[^>]+>", " ", clean)
            text = re.sub(r"\s+", " ", clean).strip()

        # Enforce size bounds
        max_size = self.policy.max_content_size
        is_truncated = False
        if len(text) > max_size:
            text = text[:max_size] + "\n...[Content truncated for context safety]"
            is_truncated = True

        # Sanitize against prompt injection patterns
        sanitized_text, has_injection, _ = self.policy.sanitize_untrusted_text(text)

        # Date extraction heuristic (e.g. meta tags or ISO dates)
        published_at = self._extract_date(html)

        return PageContent(
            url=url,
            title=title or domain,
            domain=domain,
            text=sanitized_text,
            headings=headings[:15],
            links=links[:30],
            published_at=published_at,
            source_type=self._classify_domain(domain),
            is_truncated=is_truncated,
            status_code=status_code,
        )

    @staticmethod
    def _extract_date(html: str) -> Optional[str]:
        """Detect published or modified dates from meta tags."""
        patterns = [
            r'<meta[^>]+property=["\']article:published_time["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+name=["\'](?:pubdate|publishdate|date)["\'][^>]+content=["\']([^"\']+)["\']',
            r'<time[^>]+datetime=["\']([^"\']+)["\']',
            r'\b(202[0-6]-[01]\d-[0-3]\d)\b',
        ]
        for pat in patterns:
            m = re.search(pat, html, re.IGNORECASE)
            if m:
                return m.group(1)[:10]
        return None

    @staticmethod
    def _classify_domain(domain: str) -> SourceType:
        """Infer source authority."""
        d = domain.lower()
        if any(d.endswith(s) for s in (".gov", ".edu", ".org", "docs.python.org", "github.com", "arxiv.org")):
            return SourceType.PRIMARY
        if any(k in d for k in ("documentation", "official", "developer", "nvidia.com", "microsoft.com", "apple.com")):
            return SourceType.PRIMARY
        if any(d.endswith(s) for s in ("reuters.com", "bbc.com", "apnews.com", "bloomberg.com")):
            return SourceType.SECONDARY
        return SourceType.UNKNOWN
