"""
axen/knowledge/__init__.py
==========================
AXEN Phase 4: Local Knowledge & RAG System.
"""

from .document_models import (
    CollectionInfo,
    Document,
    DocumentChunk,
    DocumentType,
    GroundingResult,
    GroundingStatus,
    ScoredChunk,
)
from .document_loader import DocumentLoader
from .document_chunker import DocumentChunker
from .embedding_provider import EmbeddingProvider, LocalEmbeddingProvider
from .vector_store import SQLiteVectorStore, DEFAULT_KNOWLEDGE_DB
from .grounding_verifier import GroundingVerifier
from .knowledge_retriever import KnowledgeRetriever
from .knowledge_manager import KnowledgeManager, IngestResult
from .knowledge_policy import KnowledgePolicy

__all__ = [
    "CollectionInfo",
    "Document",
    "DocumentChunk",
    "DocumentType",
    "GroundingResult",
    "GroundingStatus",
    "ScoredChunk",
    "DocumentLoader",
    "DocumentChunker",
    "EmbeddingProvider",
    "LocalEmbeddingProvider",
    "SQLiteVectorStore",
    "DEFAULT_KNOWLEDGE_DB",
    "GroundingVerifier",
    "KnowledgeRetriever",
    "KnowledgeManager",
    "IngestResult",
    "KnowledgePolicy",
]
