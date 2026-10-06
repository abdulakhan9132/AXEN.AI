"""
axen/tools/web_tools.py
=======================
Tool definitions exposing Web Agent capabilities to the AXEN tool registry.

Provides:
- web.search: Query search provider for structured web results.
- web.read: Safely fetch and extract clean text from a URL.
- web.research: Multi-source research, comparison, and citation synthesis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from axen.tools.tool_registry import ToolDefinition
from axen.web.page_reader import PageReader
from axen.web.search_provider import DuckDuckGoSearchProvider, WebSearchProvider
from axen.web.web_agent import WebAgent
from axen.web.web_models import ResearchDepth
from axen.web.web_policy import WebPolicy


def build_web_tools(
    web_agent: Optional[WebAgent] = None,
    provider: Optional[WebSearchProvider] = None,
    policy: Optional[WebPolicy] = None,
) -> List[ToolDefinition]:
    """Build and return callable Web Agent tools for the AXEN registry."""
    pol = policy or WebPolicy()
    agent = web_agent or WebAgent(provider=provider, policy=pol)
    reader = PageReader(policy=pol)
    search_prov = provider or agent.provider

    def _handle_search(args: Dict[str, Any]) -> Dict[str, Any]:
        query = args.get("query", "")
        max_results = int(args.get("max_results", 5))
        results = search_prov.search(query, max_results=max_results)
        return {
            "query": query,
            "count": len(results),
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "domain": r.domain,
                    "snippet": r.snippet,
                    "source_type": r.source_type.value,
                }
                for r in results
            ],
        }

    def _handle_read(args: Dict[str, Any]) -> Dict[str, Any]:
        url = args.get("url", "")
        page = reader.read_page(url)
        return {
            "url": page.url,
            "title": page.title,
            "domain": page.domain,
            "text": page.text[:2000],
            "headings": page.headings,
            "published_at": page.published_at,
            "status_code": page.status_code,
            "error": page.error,
        }

    def _handle_research(args: Dict[str, Any]) -> Dict[str, Any]:
        query = args.get("query", "")
        depth_str = args.get("depth", "standard").lower()
        depth = ResearchDepth.QUICK if depth_str == "quick" else (
            ResearchDepth.DEEP if depth_str == "deep" else ResearchDepth.STANDARD
        )
        res = agent.research(query, depth=depth)
        return {
            "success": res.success,
            "summary": res.summary,
            "sources_read": res.pages_read,
            "citations": [c.format_citation() for c in res.citations],
            "conflicts": res.conflicts_detected,
        }

    return [
        ToolDefinition(
            name="web.search",
            description="Search the web for structured results, titles, snippets, and domains.",
            handler=_handle_search,
            input_schema={"query": "str", "max_results": "int"},
            risk_level="low",
            capabilities=["search", "web_lookup"],
        ),
        ToolDefinition(
            name="web.read",
            description="Safely fetch, extract, and sanitize clean text from a web URL.",
            handler=_handle_read,
            input_schema={"url": "str"},
            risk_level="low",
            capabilities=["page_reading", "content_extraction"],
        ),
        ToolDefinition(
            name="web.research",
            description="Execute multi-source web research, cross-checking, and citation synthesis.",
            handler=_handle_research,
            input_schema={"query": "str", "depth": "str"},
            risk_level="medium",
            capabilities=["research", "fact_checking", "synthesis"],
        ),
    ]
