"""
axen/knowledge/knowledge_retriever.py
=====================================
Hybrid retrieval and reranking engine for AXEN Knowledge Base:
- Vector search (semantic relevance)
- Lexical search (BM25 keyword term matches)
- Hybrid fusion scoring
- Reranking and thresholding
- Context window protection (bounds total characters and tokens)
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import List, Optional

from .document_models import DocumentChunk, ScoredChunk
from .embedding_provider import EmbeddingProvider
from .vector_store import SQLiteVectorStore

logger = logging.getLogger("axen.knowledge.retriever")


class KnowledgeRetriever:
    """Retrieves and reranks relevant document passages for user queries."""

    # Default parameters
    SEMANTIC_WEIGHT = 0.55
    LEXICAL_WEIGHT = 0.45
    DEFAULT_THRESHOLD = 0.20
    DEFAULT_TOP_K = 4
    MAX_CONTEXT_CHARS = 3000
    MAX_CONTEXT_TOKENS = 800

    _STOPWORDS = frozenset({
        "a", "an", "the", "is", "are", "was", "were", "what", "which", "who", "whom",
        "where", "how", "when", "why", "find",
        "this", "that", "these", "those", "have", "has", "had", "for", "to", "in",
        "on", "at", "by", "from", "and", "or", "but", "about", "with", "can", "could",
        "would", "should", "will", "tell", "show", "give", "please", "my", "your",
        "according", "document", "documentation", "file", "say", "does", "do",
    })

    def __init__(
        self,
        store: SQLiteVectorStore,
        embedding_provider: EmbeddingProvider,
        relevance_threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.store = store
        self.embedding_provider = embedding_provider
        self.relevance_threshold = relevance_threshold

    def retrieve(
        self,
        query: str,
        collection_id: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
        filter_file: Optional[str] = None,
    ) -> List[ScoredChunk]:
        """
        Execute hybrid retrieval, score normalization, reranking, and context bounds.
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        # 1. Semantic Vector Search
        q_vec = self.embedding_provider.embed_text(clean_query)
        vector_results = self.store.search_vectors(q_vec, collection_id=collection_id, top_k=top_k * 3)

        # 2. Lexical / Keyword Search
        keywords = self._extract_keywords(clean_query)
        lexical_results = self.store.search_lexical(keywords, collection_id=collection_id, top_k=top_k * 3)

        # 3. Hybrid Fusion & Normalization
        combined_chunks: dict[str, DocumentChunk] = {}
        sem_scores: dict[str, float] = {}
        lex_scores: dict[str, float] = {}

        for chunk, score in vector_results:
            combined_chunks[chunk.chunk_id] = chunk
            sem_scores[chunk.chunk_id] = max(0.0, score)

        # Lexical scores are already normalized to [0, 1] by coverage and term quality
        for chunk, score in lexical_results:
            combined_chunks[chunk.chunk_id] = chunk
            lex_scores[chunk.chunk_id] = min(1.0, max(0.0, score))

        # 4. Score Candidate Chunks
        scored_candidates: list[ScoredChunk] = []
        for chunk_id, chunk in combined_chunks.items():
            # If query specified a target file, filter out non-matching files
            if filter_file and filter_file.lower() not in chunk.metadata.get("filename", "").lower():
                continue

            s_score = sem_scores.get(chunk_id, 0.0)
            l_score = lex_scores.get(chunk_id, 0.0)

            # Metadata relevance bonus: only when there is baseline content relevance
            meta_bonus = 0.0
            if s_score >= 0.20 or l_score >= 0.20:
                filename = chunk.metadata.get("filename", "").lower()
                clean_q_lower = clean_query.lower()
                clean_q_stripped = clean_q_lower.replace(" ", "").replace("_", "").replace("-", "")

                if filename:
                    fn_stem = Path(filename).stem.lower().replace("_", "").replace("-", "")
                    if (filename in clean_q_lower) or (fn_stem and len(fn_stem) > 3 and fn_stem in clean_q_stripped):
                        meta_bonus += 0.20

                if chunk.section_title:
                    sec_lower = chunk.section_title.lower()
                    sec_parts = [p.strip() for p in re.split(r"[\._\s\-]+", sec_lower) if len(p.strip()) > 2]
                    for part in sec_parts:
                        if part in clean_q_lower:
                            meta_bonus += 0.15
                            break

            final_score = (self.SEMANTIC_WEIGHT * s_score) + (self.LEXICAL_WEIGHT * l_score) + meta_bonus

            # Build readable citation
            sec = chunk.section_title or chunk.source_location or f"Part {chunk.chunk_index + 1}"
            citation = f"{chunk.metadata.get('filename', 'doc')} — {sec}"

            scored_candidates.append(
                ScoredChunk(
                    chunk=chunk,
                    score=final_score,
                    semantic_score=s_score,
                    lexical_score=l_score,
                    citation=citation,
                )
            )

        # 5. Rerank descending by score
        scored_candidates.sort(key=lambda sc: sc.score, reverse=True)

        # 6. Apply Relevance Threshold and Context Bounds (Step 19)
        selected: list[ScoredChunk] = []
        total_chars = 0
        total_tokens = 0

        for sc in scored_candidates:
            if sc.score >= self.relevance_threshold:
                chunk_len = len(sc.chunk.text)
                chunk_tokens = sc.chunk.token_estimate

                # Enforce safe context limits for TinyLlama
                if total_chars + chunk_len > self.MAX_CONTEXT_CHARS:
                    # Truncate chunk text if partial room exists
                    remaining = self.MAX_CONTEXT_CHARS - total_chars
                    trunc_suffix = "... [truncated]"
                    if remaining > len(trunc_suffix) + 30:
                        safe_slice = remaining - len(trunc_suffix)
                        sc.chunk.text = sc.chunk.text[:safe_slice] + trunc_suffix
                        sc.selected = True
                        selected.append(sc)
                        total_chars += len(sc.chunk.text)
                    break

                sc.selected = True
                selected.append(sc)
                total_chars += chunk_len
                total_tokens += chunk_tokens

                if len(selected) >= top_k:
                    break

        return selected

    def _extract_keywords(self, query: str) -> list[str]:
        """Extract meaningful alphanumeric keywords, filtering stopwords."""
        tokens = re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", query.lower())
        return [t for t in tokens if t not in self._STOPWORDS and len(t) > 2]
