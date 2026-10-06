"""
axen/web/source_manager.py
==========================
Source evaluation, authority ranking, freshness tracking, and multi-source
conflict detection for AXEN Web Agent.

Capabilities:
1. Authority & Relevance Scoring: Weights primary/official sources highest.
2. Freshness Ranking: Prioritizes recent dates for current/latest queries.
3. Multi-source Conflict Detection: Compares extracted claims and detects discrepancies.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from axen.web.web_models import ClaimEvidence, ClaimStatus, PageContent, SourceType, WebSearchResult


class SourceManager:
    """Evaluates, ranks, and compares web sources for grounding and evidence."""

    # Authority weights
    AUTHORITY_WEIGHTS = {
        SourceType.PRIMARY: 1.0,
        SourceType.SECONDARY: 0.7,
        SourceType.TERTIARY: 0.4,
        SourceType.UNKNOWN: 0.3,
    }

    def rank_search_results(
        self,
        results: List[WebSearchResult],
        query: str,
        preferred_domains: Optional[List[str]] = None,
        prefer_fresh: bool = False,
    ) -> List[WebSearchResult]:
        """
        Rank search results considering query keyword overlap, domain authority,
        and freshness.
        """
        if not results:
            return []

        q_terms = set(re.findall(r"\w+", query.lower()))
        prefs = set(d.lower() for d in (preferred_domains or []))

        def _score(r: WebSearchResult) -> float:
            score = 0.0

            # 1. Authority score
            auth_w = self.AUTHORITY_WEIGHTS.get(r.source_type, 0.3)
            score += auth_w * 3.0

            # 2. Preferred domain boost
            if any(p in r.domain.lower() for p in prefs):
                score += 2.5

            # 3. Query relevance overlap
            text_terms = set(re.findall(r"\w+", (r.title + " " + r.snippet).lower()))
            overlap = len(q_terms.intersection(text_terms))
            score += min(overlap * 0.5, 4.0)

            # 4. Freshness boost
            if prefer_fresh and r.published_at:
                # If published recently (e.g. contains current year)
                if any(yr in str(r.published_at) for yr in ("2025", "2026", "2024")):
                    score += 2.0

            return score

        return sorted(results, key=_score, reverse=True)

    def select_best_sources(
        self,
        pages: List[PageContent],
        query: str,
        max_sources: int = 4,
        prefer_fresh: bool = False,
    ) -> List[PageContent]:
        """
        Select and rank the top page contents for synthesis.
        """
        if not pages:
            return []

        valid_pages = [p for p in pages if p.is_success and p.text]
        if not valid_pages:
            return []

        q_terms = set(re.findall(r"\w+", query.lower()))

        def _page_score(p: PageContent) -> float:
            score = 0.0
            score += self.AUTHORITY_WEIGHTS.get(p.source_type, 0.3) * 3.0

            # Content relevance
            p_terms = set(re.findall(r"\w+", p.text[:3000].lower()))
            overlap = len(q_terms.intersection(p_terms))
            score += min(overlap * 0.3, 5.0)

            # Freshness
            if prefer_fresh and p.published_at:
                if any(yr in str(p.published_at) for yr in ("2025", "2026", "2024")):
                    score += 2.5

            return score

        sorted_pages = sorted(valid_pages, key=_page_score, reverse=True)
        return sorted_pages[:max_sources]

    def detect_conflicts(
        self,
        pages: List[PageContent],
        topic: str,
    ) -> Tuple[List[ClaimEvidence], List[str]]:
        """
        Extract numerical, price, and version claims from multiple pages and detect
        any conflicting values.
        Returns: (claims_list, conflicts_descriptions)
        """
        claims: List[ClaimEvidence] = []
        conflicts: List[str] = []

        if len(pages) < 2:
            # Single source cannot have an inter-source conflict
            for p in pages:
                if p.is_success:
                    claims.append(
                        ClaimEvidence(
                            claim=f"Information from {p.domain}",
                            evidence_text=p.text[:300],
                            source_url=p.url,
                            source_domain=p.domain,
                            source_type=p.source_type,
                            status=ClaimStatus.SUPPORTED,
                        )
                    )
            return claims, conflicts

        # 1. Price pattern check (e.g. $999 vs $1,049)
        price_by_source: Dict[str, Tuple[str, str]] = {}
        price_pat = re.compile(r"\$\s*([0-9]{1,5}(?:,[0-9]{3})*(?:\.[0-9]{2})?)")

        # 2. Version pattern check (e.g. Python 3.13 vs Python 3.14)
        version_by_source: Dict[str, Tuple[str, str]] = {}
        ver_pat = re.compile(r"\b(?:version|v|release)\s*([0-9]+\.[0-9]+(?:\.[0-9]+)?)\b", re.IGNORECASE)

        for p in pages:
            if not p.is_success:
                continue

            # Price extraction
            price_match = price_pat.search(p.text)
            if price_match:
                val = price_match.group(1).replace(",", "")
                price_by_source[p.domain] = (val, p.url)

            # Version extraction
            ver_match = ver_pat.search(p.text)
            if ver_match:
                ver_val = ver_match.group(1)
                version_by_source[p.domain] = (ver_val, p.url)

            # Record claim
            claims.append(
                ClaimEvidence(
                    claim=f"Summary claim from {p.domain}",
                    evidence_text=p.text[:300],
                    source_url=p.url,
                    source_domain=p.domain,
                    source_type=p.source_type,
                    status=ClaimStatus.SUPPORTED,
                )
            )

        # Evaluate Price Discrepancies
        if len(price_by_source) >= 2:
            unique_prices = set(v[0] for v in price_by_source.values())
            if len(unique_prices) > 1:
                details = ", ".join(f"{dom} reports ${val}" for dom, (val, _) in price_by_source.items())
                conflicts.append(f"Price discrepancy detected across sources: {details}.")

        # Evaluate Version Discrepancies
        if len(version_by_source) >= 2:
            unique_versions = set(v[0] for v in version_by_source.values())
            if len(unique_versions) > 1:
                details = ", ".join(f"{dom} reports v{val}" for dom, (val, _) in version_by_source.items())
                conflicts.append(f"Version discrepancy detected across sources: {details}.")

        # Mark claims as CONFLICTING if conflicts found
        if conflicts:
            for c in claims:
                c.status = ClaimStatus.CONFLICTING

        return claims, conflicts
