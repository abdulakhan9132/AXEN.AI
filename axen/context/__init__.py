"""
axen/context/__init__.py
========================
Context Engine for AXEN AI.

Coordinates:
- Scoped memory (Global, Project, Conversation)
- Fast-path filtering for simple intents (greetings, calculations, conversions)
- Trust boundary enforcement (Files, Knowledge, Web are UNTRUSTED DATA)
- Hierarchical context assembly
"""

from .context_models import (
    ContextItem,
    ContextPriority,
    ContextRequest,
    AssembledContext,
    ContextDebugInfo,
)
from .relevance import RelevanceFilter, ScoredContextItem
from .context_policy import ContextPolicy
from .context_assembler import ContextAssembler
from .context_engine import ContextEngine

__all__ = [
    "ContextItem",
    "ContextPriority",
    "ContextRequest",
    "AssembledContext",
    "ContextDebugInfo",
    "RelevanceFilter",
    "ScoredContextItem",
    "ContextPolicy",
    "ContextAssembler",
    "ContextEngine",
]
