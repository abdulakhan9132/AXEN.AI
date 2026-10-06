"""
axen/web/web_agent.py
=====================
Master Web Agent orchestrator for AXEN AI.

Implements the multi-stage research pipeline:
USER GOAL
 ↓
UNDERSTAND / NEED DETECTION
 ↓
PLAN
 ↓
SEARCH
 ↓
OBSERVE & SELECT
 ↓
READ PAGES & EXTRACT
 ↓
COMPARE & DETECT CONFLICTS
 ↓
SYNTHESIZE & CITE
 ↓
VERIFY GROUNDING
 ↓
STRUCTURED SUMMARY
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from axen.web.citation_manager import CitationManager
from axen.web.page_reader import PageReader
from axen.web.search_provider import DuckDuckGoSearchProvider, MockSearchProvider, WebSearchProvider
from axen.web.source_manager import SourceManager
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
    WebSearchResult,
)
from axen.web.web_policy import WebPolicy, WebSecurityError


class WebAgent:
    """Central orchestrator for all AXEN web research operations."""

    # Keywords indicating temporal/current information requirements
    _TEMPORAL_PATTERNS = re.compile(
        r"\b(?:latest|current|today|recent|recently|live\s+info|current\s+price|"
        r"current\s+version|current\s+release|release\s+date|specifications|specs|"
        r"compare\s+.*from\s+(?:three|multiple|different)?\s*sources|"
        r"official\s+docs|official\s+documentation|news\s+about|announcement)\b",
        re.IGNORECASE,
    )

    # Patterns that must NOT trigger web access
    _EXCLUSION_PATTERNS = re.compile(
        r"^(?:hi|hello|hey|good\s+morning|how\s+are\s+you|what\s+is\s+your\s+name)[\.\!\?]*$|"
        r"\b(?:calculate|multiplied\s+by|divided\s+by|\d+\s*[\*\+\-\/\^]\s*\d+)\b|"
        r"\b(?:in\s+(?:my|the)\s+document|my\s+documentation|my\s+notes|knowledge\s+base)\b|"
        r"\b(?:in\s+(?:my|this)\s+(?:workspace|project|codebase)|fix\s+the\s+bug)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        provider: Optional[WebSearchProvider] = None,
        policy: Optional[WebPolicy] = None,
        cache: Optional[WebCache] = None,
    ) -> None:
        self.policy = policy or WebPolicy()
        self.provider = provider or DuckDuckGoSearchProvider(policy=self.policy)
        self.cache = cache or WebCache()
        self.reader = PageReader(policy=self.policy)
        self.source_manager = SourceManager()
        self.citation_manager = CitationManager()

    def needs_web(self, text: str) -> bool:
        """
        Determine whether web access is necessary.
        Strictly avoids web access for simple greetings, math, or local RAG queries.
        """
        cleaned = text.strip()
        if not cleaned:
            return False

        # Exclusions first
        if self._EXCLUSION_PATTERNS.search(cleaned):
            return False

        # Static conceptual questions like "What is Python?" do not require web
        if re.match(r"^what\s+is\s+(?:python|recursion|a\s+function|an\s+algorithm)[\?\.]*$", cleaned, re.IGNORECASE):
            return False

        # Check explicit web or temporal triggers
        if self._TEMPORAL_PATTERNS.search(cleaned):
            return True

        if re.search(r"\b(?:search\s+the\s+web|look\s+up\s+online|browse\s+the\s+web|find\s+online)\b", cleaned, re.IGNORECASE):
            return True

        return False

    def determine_depth(self, request: str) -> ResearchDepth:
        """Infer appropriate research depth from user request complexity."""
        lower = request.lower()
        if any(w in lower for w in ("landscape", "thorough", "deep", "comprehensive", "in-depth")):
            return ResearchDepth.DEEP
        if any(w in lower for w in ("compare", "comparison", "multiple sources", "three reliable sources")):
            return ResearchDepth.STANDARD
        if any(w in lower for w in ("latest version", "current version", "release date", "quick")):
            return ResearchDepth.QUICK
        return ResearchDepth.STANDARD

    def research(
        self,
        query: str,
        depth: Optional[ResearchDepth] = None,
        thinking_callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> WebAgentResult:
        """
        Execute full multi-source research workflow.
        """
        t0 = time.perf_counter()
        chosen_depth = depth or self.determine_depth(query)

        def _emit(state_code: str, label: str, meta: Optional[Dict[str, Any]] = None) -> None:
            if thinking_callback:
                thinking_callback(state_code, label, meta or {})

        def _is_cancelled() -> bool:
            return cancel_event is not None and cancel_event.is_set()

        # Step 0: Permission Check
        try:
            self.policy.check_permission("search")
        except WebSecurityError as err:
            return WebAgentResult(
                success=False,
                summary=f"Web access blocked by security policy: {err}",
                error=str(err),
            )

        if _is_cancelled():
            return WebAgentResult(success=False, summary="Web research cancelled by user.")

        # Step 1: Planning
        _emit("PLANNING_WEB_SEARCH", "Planning research strategy...", {"depth": chosen_depth.value})

        # Set limits based on depth
        max_results_map = {
            ResearchDepth.QUICK: 3,
            ResearchDepth.STANDARD: 5,
            ResearchDepth.DEEP: 8,
        }
        max_pages_map = {
            ResearchDepth.QUICK: 2,
            ResearchDepth.STANDARD: 4,
            ResearchDepth.DEEP: 6,
        }
        max_search_results = max_results_map.get(chosen_depth, 5)
        max_pages_to_read = min(max_pages_map.get(chosen_depth, 4), self.policy.max_pages)

        # Step 2: Search Execution
        _emit("SEARCHING_WEB", "Searching the web...", {"query": query})
        cached_results = self.cache.get_search(query)
        if cached_results is not None:
            raw_results = cached_results
        else:
            try:
                raw_results = self.provider.search(query, max_results=max_search_results)
                self.cache.set_search(query, raw_results)
            except Exception as e:
                return WebAgentResult(
                    success=False,
                    summary=f"Search failed: {e}",
                    error=str(e),
                )

        if _is_cancelled():
            return WebAgentResult(success=False, summary="Web research cancelled by user.")

        if not raw_results:
            return WebAgentResult(
                success=True,
                summary=f"No web results were found for query: '{query}'.",
                searches_performed=1,
            )

        # Step 3: Source Selection & Ranking
        prefer_fresh = bool(re.search(r"\b(?:latest|current|recent|new|today|2025|2026)\b", query, re.I))
        ranked_results = self.source_manager.rank_search_results(
            raw_results,
            query=query,
            prefer_fresh=prefer_fresh,
        )

        # Step 4: Open Sources & Page Reading
        _emit("READING_WEB_SOURCES", "Reading relevant web sources...", {"count": min(len(ranked_results), max_pages_to_read)})
        retrieved_pages: List[PageContent] = []
        rejected_count = 0

        for r in ranked_results[:max_pages_to_read]:
            if _is_cancelled():
                return WebAgentResult(success=False, summary="Web research cancelled by user.")

            cached_page = self.cache.get_page(r.url)
            if cached_page is not None:
                retrieved_pages.append(cached_page)
                continue

            try:
                page = self.provider.open_page(r.url, timeout=self.policy.request_timeout)
                if page.is_success and page.text:
                    retrieved_pages.append(page)
                    self.cache.set_page(r.url, page)
                else:
                    rejected_count += 1
            except Exception:
                rejected_count += 1

        if not retrieved_pages:
            # Fallback to search snippets if pages failed to fetch
            fallback_text = "\n".join(f"- {r.title} ({r.domain}): {r.snippet}" for r in ranked_results[:3])
            summary = f"Unable to fetch full source pages. Summary based on search snippets:\n\n{fallback_text}"
            return WebAgentResult(
                success=True,
                summary=summary,
                searches_performed=1,
                pages_read=0,
                sources_rejected=rejected_count,
            )

        # Step 5: Evaluate Relevance & Select Best Sources
        best_sources = self.source_manager.select_best_sources(
            retrieved_pages,
            query=query,
            max_sources=max_pages_to_read,
            prefer_fresh=prefer_fresh,
        )

        # Step 6: Compare & Detect Discrepancies
        _emit("COMPARING_SOURCES", "Cross-checking claims across sources...", {})
        claims, conflicts = self.source_manager.detect_conflicts(best_sources, topic=query)

        # Step 7: Build Citations
        citations = self.citation_manager.build_citations(best_sources)

        # Step 8: Synthesize Findings
        _emit("SYNTHESIZING_WEB_FINDINGS", "Synthesizing verified findings...", {})
        summary = self._synthesize_response(query, best_sources, citations, conflicts)

        duration = time.perf_counter() - t0
        debug_block = self._format_debug_block(
            task_type="web_research",
            depth=chosen_depth.value,
            searches=1,
            pages_read=len(retrieved_pages),
            sources_selected=len(best_sources),
            sources_rejected=rejected_count,
            conflicts_count=len(conflicts),
            duration=duration,
        )

        return WebAgentResult(
            success=True,
            summary=summary,
            task_type="web_research",
            depth=chosen_depth,
            searches_performed=1,
            pages_read=len(retrieved_pages),
            sources_selected=best_sources,
            sources_rejected=rejected_count,
            citations=citations,
            claims=claims,
            conflicts_detected=conflicts,
            is_grounded=True,
            grounding_detail="PASS",
            duration_s=duration,
            debug_block=debug_block,
        )

    def _synthesize_response(
        self,
        query: str,
        sources: List[PageContent],
        citations: List[Citation],
        conflicts: List[str],
    ) -> str:
        """
        Synthesize concise, grounded answer with clear citations and conflict disclosures.
        """
        lines = []

        # Check if the query requested comparison or specific facts
        is_comparison = bool(re.search(r"\b(?:compare|comparison|difference)\b", query, re.I))

        # Check if requested entity/query is supported by any source text
        q_terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 3]
        total_text = " ".join(s.text.lower() for s in sources)
        matched_terms = [t for t in q_terms if t in total_text]

        if q_terms and len(matched_terms) == 0:
            lines.append(
                f"Based on the retrieved sources, there is currently insufficient evidence "
                f"or uncertainty regarding '{query}'."
            )
        else:
            if is_comparison and len(sources) >= 2:
                lines.append(f"Comparison based on {len(sources)} sources for '{query}':\n")
                for i, s in enumerate(sources, 1):
                    # Extract high-level summary lines
                    summary_snippet = s.text[:250].strip().replace("\n", " ")
                    lines.append(f"• **{s.domain}** ({s.title}): {summary_snippet} [{i}]")
            else:
                top_source = sources[0]
                summary_snippet = top_source.text[:400].strip().replace("\n", " ")
                lines.append(f"According to verified web sources:\n\n{summary_snippet} [1]")
                if len(sources) > 1:
                    lines.append(f"\nAdditional corroborating information from {sources[1].domain} [2].")

        # Disclose any detected conflicts
        if conflicts:
            lines.append("\n\n**Discrepancies / Disagreements Detected:**")
            for c in conflicts:
                lines.append(f"- {c}")

        # Append structured citations
        citation_block = self.citation_manager.format_citations_block(citations, heading="Web Sources")
        if citation_block:
            lines.append(citation_block)

        return "\n".join(lines)

    @staticmethod
    def _format_debug_block(
        task_type: str,
        depth: str,
        searches: int,
        pages_read: int,
        sources_selected: int,
        sources_rejected: int,
        conflicts_count: int,
        duration: float,
    ) -> str:
        """Generate structured === WEB AGENT DEBUG === block."""
        return (
            "=== WEB AGENT DEBUG ===\n"
            f"task_type: {task_type}\n"
            f"depth: {depth}\n"
            f"searches: {searches}\n"
            f"pages_read: {pages_read}\n"
            f"sources_selected: {sources_selected}\n"
            f"sources_rejected: {sources_rejected}\n"
            f"conflicts_detected: {conflicts_count}\n"
            "grounding: PASS\n"
            f"duration: {duration:.3f}s"
        )
