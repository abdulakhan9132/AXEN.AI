"""
axen/knowledge/knowledge_manager.py
===================================
Central Knowledge Service and Orchestrator for AXEN Local Knowledge & RAG.

Provides:
- Ingestion, content-hash deduplication, chunking, embedding, indexing
- Hybrid retrieval and reranking
- Document and collection management
- Directory synchronization (lightweight sync)
- Grounding verification and citation formatting
- Structured debug reporting (Step 21)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .document_chunker import DocumentChunker
from .document_loader import DocumentLoader
from .document_models import (
    CollectionInfo,
    Document,
    DocumentChunk,
    GroundingResult,
    GroundingStatus,
    ScoredChunk,
)
from .embedding_provider import EmbeddingProvider, LocalEmbeddingProvider
from .grounding_verifier import GroundingVerifier
from .knowledge_retriever import KnowledgeRetriever
from .vector_store import DEFAULT_KNOWLEDGE_DB, SQLiteVectorStore

logger = logging.getLogger("axen.knowledge.manager")


@dataclass
class IngestResult:
    """Outcome of a document ingestion request."""
    success: bool
    document_id: Optional[str] = None
    filename: Optional[str] = None
    chunks_created: int = 0
    status: str = "indexed"  # indexed, skipped_duplicate, updated, error
    message: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "document_id": self.document_id,
            "filename": self.filename,
            "chunks_created": self.chunks_created,
            "status": self.status,
            "message": self.message,
            "error": self.error,
        }


class KnowledgeManager:
    """Unified Facade for all AXEN Local Knowledge and RAG operations."""

    def __init__(
        self,
        db_path: Optional[Path | str] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> None:
        self.store = SQLiteVectorStore(db_path or DEFAULT_KNOWLEDGE_DB)
        self.embedding_provider = embedding_provider or LocalEmbeddingProvider()
        self.chunker = DocumentChunker()
        self.retriever = KnowledgeRetriever(self.store, self.embedding_provider)
        self.verifier = GroundingVerifier()

    # ------------------------------------------------------------------
    # Document Ingestion & Management
    # ------------------------------------------------------------------

    def add_document(
        self,
        file_path: Path | str,
        collection_id: str = "default",
        custom_metadata: Optional[dict[str, Any]] = None,
        force_reindex: bool = False,
    ) -> IngestResult:
        """
        Ingest a file into the knowledge base.
        Validates content hash to prevent redundant re-indexing (Step 3 & 10).
        """
        doc, text, err = DocumentLoader.load(file_path, collection_id=collection_id, custom_metadata=custom_metadata)
        if err or not doc:
            return IngestResult(
                success=False,
                filename=Path(file_path).name,
                status="error",
                error=err or "Unknown document load failure",
            )

        # Content hash deduplication check (Step 3 & Step 10)
        existing_doc = self.store.find_by_content_hash(doc.content_hash, collection_id=collection_id)
        if existing_doc and not force_reindex:
            return IngestResult(
                success=True,
                document_id=existing_doc.document_id,
                filename=existing_doc.filename,
                chunks_created=existing_doc.chunk_count,
                status="skipped_duplicate",
                message=f"Document '{doc.filename}' with identical content hash already indexed.",
            )

        # Structure-aware chunking (Step 4)
        chunks = self.chunker.chunk_document(doc, text)
        doc.chunk_count = len(chunks)

        # Generate local vector embeddings (Step 5)
        texts_to_embed = [c.text for c in chunks]
        embeddings = self.embedding_provider.embed_documents(texts_to_embed)

        # Save to SQLite store
        self.store.save_document(doc)
        self.store.save_chunks(chunks, embeddings)

        logger.info("Indexed '%s' into collection '%s' (%d chunks)", doc.filename, collection_id, len(chunks))
        return IngestResult(
            success=True,
            document_id=doc.document_id,
            filename=doc.filename,
            chunks_created=len(chunks),
            status="indexed",
            message=f"Successfully indexed '{doc.filename}' with {len(chunks)} chunks.",
        )

    def ingest_text(
        self,
        text: str,
        title: str = "document.txt",
        collection_id: str = "default",
        collection: Optional[str] = None,
    ) -> IngestResult:
        """Ingest raw text into a knowledge collection."""
        import tempfile
        cid = collection or collection_id
        with tempfile.NamedTemporaryFile("w", suffix=Path(title).suffix or ".txt", delete=False, encoding="utf-8") as tf:
            tf.write(text)
            temp_path = tf.name
        try:
            return self.add_document(temp_path, collection_id=cid)
        finally:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass


    def remove_document(self, document_id: str) -> bool:
        """Remove a document and all its chunks from the knowledge base."""
        return self.store.delete_document(document_id)

    def update_document(
        self,
        file_path: Path | str,
        collection_id: str = "default",
    ) -> IngestResult:
        """Re-index a modified document."""
        # Find existing by filepath
        existing = self.store.find_by_filepath(str(file_path), collection_id=collection_id)
        if existing:
            self.remove_document(existing.document_id)
        return self.add_document(file_path, collection_id=collection_id, force_reindex=True)

    def list_documents(self, collection_id: Optional[str] = None) -> List[Document]:
        """List indexed documents."""
        return self.store.list_documents(collection_id=collection_id)

    def get_document(self, document_id: str) -> Optional[Document]:
        """Fetch document metadata by ID."""
        return self.store.get_document(document_id)

    # ------------------------------------------------------------------
    # Directory Synchronization (Step 16)
    # ------------------------------------------------------------------

    def sync_folder(
        self,
        folder_path: Path | str,
        collection_id: str = "default",
        recursive: bool = False,
    ) -> Dict[str, int]:
        """
        Synchronize a directory:
        - Ingest new files
        - Reindex modified files (via content hash)
        - Skip unchanged files
        """
        path = Path(folder_path).resolve()
        if not path.is_dir():
            return {"indexed": 0, "skipped": 0, "errors": 0}

        pattern = "**/*" if recursive else "*"
        stats = {"indexed": 0, "skipped": 0, "errors": 0}

        for p in path.glob(pattern):
            if p.is_file():
                res = self.add_document(p, collection_id=collection_id)
                if res.success:
                    if res.status == "skipped_duplicate":
                        stats["skipped"] += 1
                    else:
                        stats["indexed"] += 1
                else:
                    stats["errors"] += 1

        return stats

    # ------------------------------------------------------------------
    # Retrieval & Search (Step 8 & 9)
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        collection_id: Optional[str] = None,
        top_k: int = 4,
        filter_file: Optional[str] = None,
    ) -> List[ScoredChunk]:
        """Retrieve relevant scored passages with hybrid search and reranking."""
        return self.retriever.retrieve(
            query=query,
            collection_id=collection_id,
            top_k=top_k,
            filter_file=filter_file,
        )

    def search(
        self,
        query: str,
        collection_id: Optional[str] = None,
        top_k: int = 4,
    ) -> List[dict[str, Any]]:
        """Convenience method returning serialized search results."""
        scored = self.retrieve(query, collection_id=collection_id, top_k=top_k)
        return [sc.to_dict() for sc in scored]

    # ------------------------------------------------------------------
    # Collections (Step 7)
    # ------------------------------------------------------------------

    def list_collections(self) -> List[CollectionInfo]:
        """List all knowledge collections."""
        return self.store.list_collections()

    def create_collection(self, collection_id: str, name: str, description: str = "") -> CollectionInfo:
        """Create a new collection."""
        return self.store.create_collection(collection_id, name, description)

    def clear_collection(self, collection_id: str) -> int:
        """Clear all contents of a collection."""
        return self.store.clear_collection(collection_id)

    # ------------------------------------------------------------------
    # Grounding Verification (Step 22)
    # ------------------------------------------------------------------

    def verify_grounding(
        self,
        answer: str,
        sources: List[ScoredChunk],
    ) -> GroundingResult:
        """Verify whether generated response is grounded in retrieved chunks."""
        return self.verifier.verify_grounding(answer, sources)

    # ------------------------------------------------------------------
    # Prompt Context Assembly (Step 19 & Step 10)
    # ------------------------------------------------------------------

    def format_prompt_context(self, chunks: List[ScoredChunk]) -> Tuple[str, List[str]]:
        """
        Assemble retrieved passages into a structured prompt block with citations.
        Returns: (formatted_context_string, list_of_citations)
        """
        if not chunks:
            return "", []

        lines = [
            "KNOWLEDGE SOURCES (Ground your answer strictly on the following information. Do not invent facts):"
        ]
        citations = []

        for i, sc in enumerate(chunks, 1):
            citation_label = f"[{i}] {sc.citation}"
            citations.append(citation_label)
            lines.append(f"\n--- Source {citation_label} ---")
            lines.append(sc.chunk.text.strip())

        return "\n".join(lines), citations

    # ------------------------------------------------------------------
    # Structured Debug Block (Step 21)
    # ------------------------------------------------------------------

    def format_debug_block(
        self,
        query: str,
        chunks: List[ScoredChunk],
        grounding_result: Optional[GroundingResult] = None,
        collection_id: str = "default",
    ) -> str:
        """
        Format structured RAG debug block (Step 21):
        === KNOWLEDGE DEBUG ===
        query: ...
        collection: ...
        retrieval: hybrid
        candidate_chunks: ...
        selected_chunks: ...
        sources: ...
        context_tokens: ...
        grounding: PASS / FAIL
        """
        sources = list(dict.fromkeys(sc.chunk.metadata.get("filename", "unknown") for sc in chunks))
        total_tokens = sum(sc.chunk.token_estimate for sc in chunks)

        grounding_status = "UNKNOWN"
        if grounding_result:
            grounding_status = "PASS" if grounding_result.status in (GroundingStatus.GROUNDED, GroundingStatus.PARTIALLY_GROUNDED) else "FAIL"

        lines = [
            "=== KNOWLEDGE DEBUG ===",
            f"query: {query[:80]}",
            f"collection: {collection_id}",
            "retrieval: hybrid (vector + lexical)",
            f"selected_chunks: {len(chunks)}",
            f"sources: {', '.join(sources) if sources else 'none'}",
            f"context_tokens: {total_tokens}",
            f"grounding: {grounding_status}",
        ]

        if grounding_result and grounding_result.detail:
            lines.append(f"grounding_detail: {grounding_result.detail}")

        return "\n".join(lines)
