"""
axen/web/citation_manager.py
============================
Citation generation, formatting, and traceability management.

Ensures every factual claim derived from web research has a verifiable,
traceable citation with URL, domain, title, and optional publication timestamp.
Distinguishes web citations cleanly from local RAG documents.
"""

from __future__ import annotations

from typing import List, Optional

from axen.web.web_models import Citation, PageContent


class CitationManager:
    """Manages and formats citations for the AXEN Web Agent."""

    def build_citations(self, pages: List[PageContent]) -> List[Citation]:
        """
        Convert retrieved PageContent objects into structured, indexed Citations.
        Deduplicates by URL and excludes failed/empty retrievals.
        """
        citations: List[Citation] = []
        seen_urls = set()

        idx = 1
        for p in pages:
            if not p.is_success or not p.url:
                continue
            if p.url in seen_urls:
                continue
            seen_urls.add(p.url)

            citations.append(
                Citation(
                    index=idx,
                    title=p.title or p.domain,
                    url=p.url,
                    domain=p.domain,
                    published_at=p.published_at,
                    source_type=p.source_type,
                    quoted_snippet=p.text[:150].strip() if p.text else "",
                )
            )
            idx += 1

        return citations

    def format_citations_block(
        self,
        citations: List[Citation],
        heading: str = "Web Sources",
    ) -> str:
        """
        Render formatted citations block for appending to user responses.
        """
        if not citations:
            return ""

        lines = [f"\n\n{heading}:"]
        for c in citations:
            date_info = f" ({c.published_at})" if c.published_at else ""
            lines.append(f"[{c.index}] {c.domain} — {c.title}{date_info}")
            lines.append(f"    URL: {c.url}")

        return "\n".join(lines)
