"""
axen/tools/knowledge_tool.py
============================
Knowledge retrieval tool for AXEN Agent Brain.
Provides structured document search and passage retrieval with citations.
"""

from __future__ import annotations

from typing import Any, Optional

from axen.knowledge.knowledge_manager import KnowledgeManager
from .tool_registry import ToolDefinition


_DEFAULT_KNOWLEDGE_MANAGER: Optional[KnowledgeManager] = None


def get_default_knowledge_manager() -> KnowledgeManager:
    global _DEFAULT_KNOWLEDGE_MANAGER
    if _DEFAULT_KNOWLEDGE_MANAGER is None:
        _DEFAULT_KNOWLEDGE_MANAGER = KnowledgeManager()
    return _DEFAULT_KNOWLEDGE_MANAGER


def search_knowledge(args: dict[str, Any]) -> dict[str, Any]:
    """
    Search and retrieve grounded passages from indexed knowledge documents.
    Args:
        args["query"]: Question or search term.
        args["collection_id"]: Optional collection to limit search to.
        args["top_k"]: Number of passages to return (default 4).
    """
    query = str(args.get("query") or args.get("request") or "").strip()
    if not query:
        return {"success": False, "error": "Missing 'query' argument."}

    collection_id = args.get("collection_id")
    top_k = int(args.get("top_k", 4))

    manager = get_default_knowledge_manager()
    scored_chunks = manager.retrieve(query=query, collection_id=collection_id, top_k=top_k)

    if not scored_chunks:
        return {
            "success": True,
            "type": "knowledge_result",
            "query": query,
            "sources": [],
            "chunks": [],
            "count": 0,
            "message": "No relevant document passages found.",
        }

    sources = list(dict.fromkeys(sc.citation for sc in scored_chunks))
    formatted_context, citations = manager.format_prompt_context(scored_chunks)

    return {
        "success": True,
        "type": "knowledge_result",
        "query": query,
        "sources": sources,
        "citations": citations,
        "chunks": [sc.to_dict() for sc in scored_chunks],
        "context_text": formatted_context,
        "count": len(scored_chunks),
        "confidence": round(scored_chunks[0].score, 3) if scored_chunks else 0.0,
    }


def build_knowledge_tool() -> ToolDefinition:
    """Return the AXEN knowledge retrieval ToolDefinition."""
    return ToolDefinition(
        name="knowledge.retrieve",
        description=(
            "Search and retrieve relevant passages from the local knowledge base with source citations. "
            "Args: {'query': '<search_query>', 'collection_id': '<optional_collection>', 'top_k': 4}"
        ),
        handler=search_knowledge,
        risk="low",
        capabilities=["knowledge_retrieval", "document_search", "rag"],
        deterministic=True,
        requires_confirmation=False,
        supports_verification=True,
    )
