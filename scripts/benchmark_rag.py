"""
scripts/benchmark_rag.py
========================
Performance benchmark for AXEN Phase 4 Local Knowledge & RAG System:
Measures:
- Ingestion time
- Chunking time
- Embedding time
- Indexing time
- Retrieval time
- Reranking time
- Context assembly time
- Total RAG latency
"""

import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from axen.knowledge.document_chunker import DocumentChunker
from axen.knowledge.document_loader import DocumentLoader
from axen.knowledge.document_models import Document, DocumentType
from axen.knowledge.embedding_provider import LocalEmbeddingProvider
from axen.knowledge.knowledge_manager import KnowledgeManager
from axen.knowledge.knowledge_retriever import KnowledgeRetriever
from axen.knowledge.vector_store import SQLiteVectorStore


def run_benchmark():
    print("=" * 60)
    print("AXEN PHASE 4 — LOCAL KNOWLEDGE & RAG PERFORMANCE BENCHMARK")
    print("=" * 60)

    # Prepare temporary test documents of various types and sizes
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        temp_path = Path(temp_dir)
        db_path = temp_path / "bench_knowledge.db"

        # 1. Create realistic benchmark documents
        doc_paths = []
        for i in range(10):
            p = temp_path / f"doc_{i}.md"
            content = f"# System Architecture Component {i}\n\n"
            for j in range(8):
                content += f"## Subsystem {j}\n"
                content += (
                    f"Component {i} subsystem {j} coordinates modular execution, "
                    f"managing local tensor operations and deterministic cache storage. "
                    f"Parameters are bounded to ensure safe execution on 4GB VRAM and 8GB RAM devices.\n\n"
                )
            p.write_text(content, encoding="utf-8")
            doc_paths.append(p)

        km = KnowledgeManager(db_path=str(db_path))

        # Bench 1: Document Ingestion (full pipeline)
        t0 = time.perf_counter()
        total_chunks = 0
        for p in doc_paths:
            res = km.add_document(p, collection_id="benchmarks")
            assert res.success
            total_chunks += res.chunks_created
        ingest_total_s = time.perf_counter() - t0
        ingest_per_doc_ms = (ingest_total_s / len(doc_paths)) * 1000

        print(f"1. Document Ingestion (10 docs, {total_chunks} chunks):")
        print(f"   Total: {ingest_total_s*1000:.2f} ms ({ingest_per_doc_ms:.2f} ms/doc)")

        # Bench 2: Chunking time isolated
        chunker = DocumentChunker()
        sample_doc = km.list_documents()[0]
        sample_text = DocumentLoader.load(doc_paths[0])[1]

        t0 = time.perf_counter()
        for _ in range(50):
            _ = chunker.chunk_document(sample_doc, sample_text)
        chunking_ms = ((time.perf_counter() - t0) / 50) * 1000
        print(f"2. Chunking Time (isolated, ~1.5KB doc): {chunking_ms:.3f} ms")

        # Bench 3: Embedding time isolated
        ep = LocalEmbeddingProvider()
        test_chunk_text = "Component 0 subsystem 0 coordinates modular execution and local cache storage."
        t0 = time.perf_counter()
        for _ in range(200):
            _ = ep.embed_text(test_chunk_text)
        embed_ms = ((time.perf_counter() - t0) / 200) * 1000
        print(f"3. Embedding Time (isolated, 1 chunk): {embed_ms:.3f} ms")

        # Bench 4: SQLite Indexing / Upsert time
        t0 = time.perf_counter()
        extra_doc = temp_path / "extra.md"
        extra_doc.write_text("# Extra\n\nSome indexed content.", encoding="utf-8")
        km.add_document(extra_doc, collection_id="benchmarks")
        index_ms = (time.perf_counter() - t0) * 1000
        print(f"4. Indexing & Storage Write Time (1 new doc): {index_ms:.2f} ms")

        # Bench 5: Retrieval & Reranking breakdown
        query = "modular execution and tensor operations"
        clean_q = query.strip()
        q_vec = ep.embed_text(clean_q)
        keywords = km.retriever._extract_keywords(clean_q)

        # 5a. Vector search alone
        t0 = time.perf_counter()
        for _ in range(50):
            _ = km.store.search_vectors(q_vec, collection_id="benchmarks", top_k=10)
        vec_search_ms = ((time.perf_counter() - t0) / 50) * 1000
        print(f"5a. Vector Similarity Search (SQLite binary blobs): {vec_search_ms:.3f} ms")

        # 5b. Lexical search alone
        t0 = time.perf_counter()
        for _ in range(50):
            _ = km.store.search_lexical(keywords, collection_id="benchmarks", top_k=10)
        lex_search_ms = ((time.perf_counter() - t0) / 50) * 1000
        print(f"5b. Lexical Keyword Search (SQLite): {lex_search_ms:.3f} ms")

        # 5c. Complete Hybrid Retrieval + Reranking + Context Assembly
        t0 = time.perf_counter()
        iterations = 50
        for _ in range(iterations):
            results = km.retrieve(query, collection_id="benchmarks", top_k=4)
        total_retrieval_ms = ((time.perf_counter() - t0) / iterations) * 1000
        print(f"6. End-to-End Retrieval Pipeline (Search + Rerank + Context Assembly): {total_retrieval_ms:.3f} ms")
        print(f"   Chunks retrieved: {len(results)}")
        if results:
            print(f"   Top score: {results[0].score:.4f}, citation: {results[0].citation}")

        # Bench 7: Context Assembly
        t0 = time.perf_counter()
        for _ in range(100):
            context_str = "\n\n".join(f"[{i+1}] {r.citation}\n{r.chunk.text}" for i, r in enumerate(results))
        assembly_ms = ((time.perf_counter() - t0) / 100) * 1000
        print(f"7. Context Assembly Time: {assembly_ms:.4f} ms")

        print("=" * 60)
        print("BENCHMARK COMPLETED SUCCESSFULLY")
        print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
