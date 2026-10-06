"""
tests/test_rag_system.py
========================
Comprehensive test suite for AXEN AI Phase 4: Local Knowledge & RAG System.

Validates all 15 Phase 4 requirements:
  TEST 1  - Ingestion (Document indexed successfully with metadata)
  TEST 2  - Retrieval (Relevant chunk retrieved for query)
  TEST 3  - Source Citations (Source metadata and location preserved)
  TEST 4  - Grounded Answer (Answer strictly supported by source documents)
  TEST 5  - Unknown / Missing Info (Missing information gracefully declined, no hallucinations)
  TEST 6  - Multi-Document Retrieval (Passages retrieved from multiple indexed docs)
  TEST 7  - Irrelevant Document Filtering (Unrelated content does not contaminate results)
  TEST 8  - Document Update (Modified document reindexed, old content deactivated)
  TEST 9  - Document Deletion (Deleted document completely removed from retrieval)
  TEST 10 - Content Hash Deduplication (Identical document skipped, no duplicate chunks)
  TEST 11 - Collection Isolation (Collections remain completely isolated)
  TEST 12 - Memory vs Knowledge Separation (Personal memory never cited as document knowledge)
  TEST 13 - Large Document & Context Protection (Bounded chunks, no context overflow)
  TEST 14 - PDF Processing (PDF text extraction and page metadata)
  TEST 15 - Code Ingestion & Retrieval (Python code files indexed and retrieved)
  TEST 16 - Sensitive File Security Policy (Secret files and private keys blocked)
  TEST 17 - Grounding Verification (GroundingVerifier checks claims against sources)
"""

from __future__ import annotations

import os
import tempfile
import zlib
from pathlib import Path
import pytest

from axen.knowledge.document_models import (
    Document,
    DocumentChunk,
    DocumentType,
    GroundingResult,
    GroundingStatus,
    ScoredChunk,
)
from axen.knowledge.document_loader import DocumentLoader
from axen.knowledge.document_chunker import DocumentChunker
from axen.knowledge.embedding_provider import LocalEmbeddingProvider
from axen.knowledge.vector_store import SQLiteVectorStore
from axen.knowledge.knowledge_retriever import KnowledgeRetriever
from axen.knowledge.knowledge_manager import KnowledgeManager, IngestResult
from axen.knowledge.knowledge_policy import KnowledgePolicy
from axen.knowledge.grounding_verifier import GroundingVerifier
from axen.memory.memory_manager import MemoryManager
from axen.memory.memory_models import MemoryType
from axen.assistant.controller import AssistantController
from axen.tools import build_default_registry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_knowledge_dir(tmp_path):
    """Temporary directory with sample test documents."""
    docs_dir = tmp_path / "test_docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    # 1. axen_overview.md
    overview_path = docs_dir / "axen_overview.md"
    overview_path.write_text(
        "# AXEN AI Overview\n\n"
        "AXEN is a local-first, private, modular, agentic AI assistant designed to run on consumer hardware.\n\n"
        "## Core Principles\n\n"
        "All data stays strictly on the user machine with zero external cloud dependencies. "
        "AXEN provides tool use, mathematical solving, and local knowledge retrieval.\n",
        encoding="utf-8",
    )

    # 2. memory.md
    memory_path = docs_dir / "memory.md"
    memory_path.write_text(
        "# AXEN Memory Architecture\n\n"
        "## Memory Separation\n\n"
        "AXEN clearly separates short-term conversation memory from long-term user memory.\n\n"
        "## Persistence\n\n"
        "Long-term memories are stored in a dedicated local SQLite database called axen_memory.db. "
        "Memory stores facts about the user such as identity, preferences, and projects.\n",
        encoding="utf-8",
    )

    # 3. agent.md
    agent_path = docs_dir / "agent.md"
    agent_path.write_text(
        "# Agent Loop\n\n"
        "## The Reasoning Cycle\n\n"
        "The Agent Loop coordinates understanding, planning, tool execution, observation, and verification. "
        "The Agent Loop uses long-term memory to personalize responses while keeping instructions authoritative.\n",
        encoding="utf-8",
    )

    # 4. sample.txt
    txt_path = docs_dir / "sample.txt"
    txt_path.write_text(
        "AXEN System Settings:\n"
        "Primary accent color: orange.\n"
        "Default theme setting: system adaptive.\n",
        encoding="utf-8",
    )

    # 5. sample.csv
    csv_path = docs_dir / "sample.csv"
    csv_path.write_text(
        "Module,Status,Version\n"
        "AgentLoop,Active,2.1\n"
        "MemoryStore,Active,3.0\n"
        "KnowledgeRAG,Active,4.0\n",
        encoding="utf-8",
    )

    return docs_dir


@pytest.fixture
def knowledge_manager(tmp_path):
    """Isolated KnowledgeManager with fresh SQLite DB."""
    db_file = tmp_path / "test_knowledge.db"
    mgr = KnowledgeManager(db_path=str(db_file))
    return mgr


# ===========================================================================
# Phase 4 Test Suite
# ===========================================================================

class TestPhase4RAGSystem:

    # -----------------------------------------------------------------------
    # TEST 1 — INGESTION
    # -----------------------------------------------------------------------
    def test_document_ingestion(self, knowledge_manager, temp_knowledge_dir):
        """Add a test document and verify it is indexed successfully with metadata and chunks."""
        file_path = temp_knowledge_dir / "axen_overview.md"
        result = knowledge_manager.add_document(file_path)

        assert result.success is True
        assert result.status == "indexed"
        assert result.chunks_created >= 1
        assert result.document_id is not None

        # Verify in store
        doc = knowledge_manager.get_document(result.document_id)
        assert doc is not None
        assert doc.filename == "axen_overview.md"
        assert doc.file_type == DocumentType.MARKDOWN
        assert len(doc.content_hash) == 64

    # -----------------------------------------------------------------------
    # TEST 2 — RETRIEVAL
    # -----------------------------------------------------------------------
    def test_relevant_retrieval(self, knowledge_manager, temp_knowledge_dir):
        """Query 'What is AXEN?' retrieves the relevant overview passage."""
        knowledge_manager.add_document(temp_knowledge_dir / "axen_overview.md")

        results = knowledge_manager.retrieve("What is AXEN?", top_k=2)
        assert len(results) >= 1
        top = results[0]
        assert "local-first" in top.chunk.text.lower()
        assert top.score > 0.20
        assert top.selected is True

    # -----------------------------------------------------------------------
    # TEST 3 — SOURCE CITATIONS
    # -----------------------------------------------------------------------
    def test_source_citations_preserved(self, knowledge_manager, temp_knowledge_dir):
        """Source metadata, section titles, and formatted citations are preserved."""
        knowledge_manager.add_document(temp_knowledge_dir / "axen_overview.md")

        results = knowledge_manager.retrieve("What is AXEN?", top_k=1)
        assert len(results) == 1
        chunk = results[0]
        assert "axen_overview.md" in chunk.citation
        assert chunk.chunk.metadata.get("filename") == "axen_overview.md"

        context_text, citations = knowledge_manager.format_prompt_context(results)
        assert "[1] axen_overview.md" in citations[0]
        assert "KNOWLEDGE SOURCES" in context_text

    # -----------------------------------------------------------------------
    # TEST 4 — GROUNDED ANSWER
    # -----------------------------------------------------------------------
    def test_grounded_answer(self, knowledge_manager, temp_knowledge_dir):
        """Grounded answer is generated and supported by document sources."""
        knowledge_manager.add_document(temp_knowledge_dir / "axen_overview.md")
        results = knowledge_manager.retrieve("What is AXEN?")
        assert len(results) >= 1

        passage = results[0].chunk.text
        grounding = knowledge_manager.verify_grounding(passage, results)
        assert grounding.status == GroundingStatus.GROUNDED
        assert grounding.grounded_ratio >= 0.75

    # -----------------------------------------------------------------------
    # TEST 5 — UNKNOWN / MISSING INFO
    # -----------------------------------------------------------------------
    def test_missing_info_no_hallucination(self, knowledge_manager, temp_knowledge_dir):
        """Query for missing information yields no matching chunks or a low score."""
        knowledge_manager.add_document(temp_knowledge_dir / "axen_overview.md")

        # AXEN documents do not mention any football team
        results = knowledge_manager.retrieve("What is AXEN's favorite football team?")
        # Either rejected by threshold or no semantic overlap
        assert len(results) == 0

    # -----------------------------------------------------------------------
    # TEST 6 — MULTI-DOCUMENT RETRIEVAL
    # -----------------------------------------------------------------------
    def test_multi_document_retrieval(self, knowledge_manager, temp_knowledge_dir):
        """Query about memory and agent loop retrieves chunks from both documents."""
        knowledge_manager.add_document(temp_knowledge_dir / "memory.md")
        knowledge_manager.add_document(temp_knowledge_dir / "agent.md")

        results = knowledge_manager.retrieve("How are memory and the Agent Loop related?", top_k=4)
        assert len(results) >= 2

        filenames = [sc.chunk.metadata.get("filename") for sc in results]
        assert "memory.md" in filenames
        assert "agent.md" in filenames

    # -----------------------------------------------------------------------
    # TEST 7 — IRRELEVANT DOCUMENT FILTERING
    # -----------------------------------------------------------------------
    def test_irrelevant_document_does_not_dominate(self, knowledge_manager, temp_knowledge_dir, tmp_path):
        """Unrelated document (e.g. recipe) does not dominate AXEN query."""
        knowledge_manager.add_document(temp_knowledge_dir / "memory.md")

        # Create unrelated file
        recipe = tmp_path / "pasta_recipe.txt"
        recipe.write_text("Boil pasta in salted water for 10 minutes. Add tomato sauce and parmesan.", encoding="utf-8")
        knowledge_manager.add_document(recipe)

        results = knowledge_manager.retrieve("What is AXEN memory?", top_k=2)
        assert len(results) >= 1
        assert results[0].chunk.metadata.get("filename") == "memory.md"
        assert "pasta_recipe.txt" not in [r.chunk.metadata.get("filename") for r in results]

    # -----------------------------------------------------------------------
    # TEST 8 — DOCUMENT UPDATE
    # -----------------------------------------------------------------------
    def test_document_update_reindexes(self, knowledge_manager, tmp_path):
        """Modifying a document and updating it replaces the old content."""
        doc_file = tmp_path / "config.txt"
        doc_file.write_text("AXEN Version: 1.0 (Alpha release)", encoding="utf-8")
        knowledge_manager.add_document(doc_file)

        initial = knowledge_manager.retrieve("What version is AXEN?")
        assert "1.0" in initial[0].chunk.text

        # Modify document
        doc_file.write_text("AXEN Version: 2.0 (Stable release)", encoding="utf-8")
        res_update = knowledge_manager.update_document(doc_file)
        assert res_update.success is True

        updated = knowledge_manager.retrieve("What version is AXEN?")
        assert "2.0" in updated[0].chunk.text
        assert "1.0" not in updated[0].chunk.text

    # -----------------------------------------------------------------------
    # TEST 9 — DOCUMENT DELETION
    # -----------------------------------------------------------------------
    def test_document_deletion(self, knowledge_manager, temp_knowledge_dir):
        """Deleting a document removes it completely from retrieval."""
        res = knowledge_manager.add_document(temp_knowledge_dir / "axen_overview.md")
        doc_id = res.document_id

        assert len(knowledge_manager.retrieve("What is AXEN?")) >= 1

        deleted = knowledge_manager.remove_document(doc_id)
        assert deleted is True

        assert len(knowledge_manager.retrieve("What is AXEN?")) == 0
        assert knowledge_manager.get_document(doc_id) is None

    # -----------------------------------------------------------------------
    # TEST 10 — CONTENT HASH DEDUPLICATION
    # -----------------------------------------------------------------------
    def test_content_hash_deduplication(self, knowledge_manager, temp_knowledge_dir):
        """Adding the exact same document twice skips re-indexing."""
        file_path = temp_knowledge_dir / "axen_overview.md"
        res1 = knowledge_manager.add_document(file_path)
        assert res1.status == "indexed"

        res2 = knowledge_manager.add_document(file_path)
        assert res2.status == "skipped_duplicate"

        docs = knowledge_manager.list_documents()
        assert len(docs) == 1

    # -----------------------------------------------------------------------
    # TEST 11 — COLLECTION ISOLATION
    # -----------------------------------------------------------------------
    def test_collection_isolation(self, knowledge_manager, temp_knowledge_dir, tmp_path):
        """Documents in Collection A do not contaminate searches in Collection B."""
        # Ingest into Collection A
        knowledge_manager.add_document(temp_knowledge_dir / "axen_overview.md", collection_id="axen_docs")

        # Ingest into Collection B
        ui_file = tmp_path / "ui_design.md"
        ui_file.write_text("# UI Design Guide\n\nFigma design system uses 8px grid and Inter font.", encoding="utf-8")
        knowledge_manager.add_document(ui_file, collection_id="ui_docs")

        # Search Collection A
        results_a = knowledge_manager.retrieve("design system", collection_id="axen_docs")
        assert len(results_a) == 0  # ui_design is in ui_docs, not axen_docs

        # Search Collection B
        results_b = knowledge_manager.retrieve("design system", collection_id="ui_docs")
        assert len(results_b) >= 1
        assert "ui_design.md" in results_b[0].citation

    # -----------------------------------------------------------------------
    # TEST 12 — MEMORY VS KNOWLEDGE SEPARATION
    # -----------------------------------------------------------------------
    def test_memory_and_knowledge_separation(self, knowledge_manager, tmp_path):
        """Personal memory is isolated from knowledge and never cited as a document source."""
        # 1. Store memory
        mem_db = str(tmp_path / "test_memory.db")
        mem_mgr = MemoryManager(db_path=mem_db)
        mem_mgr.remember(key="ui_theme", value="dark UI", memory_type=MemoryType.PREFERENCE)

        # 2. Ingest document with knowledge
        doc_file = tmp_path / "colors.txt"
        doc_file.write_text("AXEN default accent is orange.", encoding="utf-8")
        knowledge_manager.add_document(doc_file)

        # Query memory
        mem_ans = mem_mgr.retrieve("What UI style do I prefer?")
        assert len(mem_ans) >= 1
        assert "dark" in mem_ans[0].value

        # Query knowledge
        k_ans = knowledge_manager.retrieve("What is the accent color?")
        assert len(k_ans) >= 1
        assert "colors.txt" in k_ans[0].citation
        assert "dark UI" not in k_ans[0].chunk.text

    # -----------------------------------------------------------------------
    # TEST 13 — LARGE DOCUMENT CONTEXT PROTECTION
    # -----------------------------------------------------------------------
    def test_large_document_context_bounds(self, knowledge_manager, tmp_path):
        """A large document produces multiple bounded chunks without context overflow."""
        large_file = tmp_path / "large_guide.txt"
        # Create ~50 KB text file
        paragraphs = [f"Section {i}: AXEN modular architecture guidelines part {i}. Detailed specifications and design patterns." for i in range(200)]
        large_file.write_text("\n\n".join(paragraphs), encoding="utf-8")

        res = knowledge_manager.add_document(large_file)
        assert res.success is True
        assert res.chunks_created > 5

        # Retrieval must be bounded within MAX_CONTEXT_CHARS
        retrieved = knowledge_manager.retrieve("architecture guidelines", top_k=4)
        assert len(retrieved) <= 4
        total_chars = sum(len(sc.chunk.text) for sc in retrieved)
        assert total_chars <= knowledge_manager.retriever.MAX_CONTEXT_CHARS

    # -----------------------------------------------------------------------
    # TEST 14 — PDF TEXT EXTRACTION
    # -----------------------------------------------------------------------
    def test_pdf_extraction(self, knowledge_manager, tmp_path):
        """PDF document loading extracts text and preserves page metadata."""
        # Generate a valid minimal PDF binary with a FlateDecode text stream
        pdf_path = tmp_path / "sample.pdf"
        stream_content = b"BT /F1 12 Tf 72 712 Td (AXEN PDF Documentation - Page 1 Content) Tj ET"
        compressed = zlib.compress(stream_content)

        pdf_bytes = (
            b"%PDF-1.4\n"
            b"1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj\n"
            b"2 0 obj <</Type /Pages /Kids [3 0 R] /Count 1>> endobj\n"
            b"3 0 obj <</Type /Page /Parent 2 0 R /Contents 4 0 R>> endobj\n"
            b"4 0 obj <</Length " + str(len(compressed)).encode("ascii") + b" /Filter /FlateDecode>>\n"
            b"stream\n" + compressed + b"\nendstream\nendobj\n"
            b"xref\n0 5\n0000000000 65535 f \n"
            b"trailer <</Size 5 /Root 1 0 R>>\nstartxref\n999\n%%EOF"
        )
        pdf_path.write_bytes(pdf_bytes)

        res = knowledge_manager.add_document(pdf_path)
        assert res.success is True

        retrieved = knowledge_manager.retrieve("PDF Documentation")
        assert len(retrieved) >= 1
        assert "PDF Documentation" in retrieved[0].chunk.text
        assert "Page 1" in retrieved[0].citation or "sample.pdf" in retrieved[0].citation

    # -----------------------------------------------------------------------
    # TEST 15 — CODE INGESTION & RETRIEVAL
    # -----------------------------------------------------------------------
    def test_code_ingestion_and_retrieval(self, knowledge_manager, tmp_path):
        """Python source code is indexed and searchable by class/function name."""
        code_file = tmp_path / "agent_loop.py"
        code_file.write_text(
            "class AgentLoop:\n"
            "    def __init__(self, tools, memory):\n"
            "        self.tools = tools\n"
            "        self.memory = memory\n\n"
            "    def execute_plan(self, plan):\n"
            "        for step in plan:\n"
            "            self.run_step(step)\n",
            encoding="utf-8",
        )

        res = knowledge_manager.add_document(code_file)
        assert res.success is True

        retrieved = knowledge_manager.retrieve("Where is AgentLoop implemented?")
        assert len(retrieved) >= 1
        assert "class AgentLoop" in retrieved[0].chunk.text

    # -----------------------------------------------------------------------
    # TEST 16 — SENSITIVE FILE PROTECTION
    # -----------------------------------------------------------------------
    def test_sensitive_file_blocking(self, knowledge_manager, tmp_path):
        """.env and private key files are strictly blocked by KnowledgePolicy."""
        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=sk-1234567890abcdef12345678", encoding="utf-8")
        res_env = knowledge_manager.add_document(env_file)
        assert res_env.success is False
        assert "sensitive file policy" in res_env.error

        key_file = tmp_path / "id_rsa"
        key_file.write_text("-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA...", encoding="utf-8")
        res_key = knowledge_manager.add_document(key_file)
        assert res_key.success is False

    # -----------------------------------------------------------------------
    # TEST 17 — GROUNDING VERIFIER
    # -----------------------------------------------------------------------
    def test_grounding_verifier(self, knowledge_manager, temp_knowledge_dir):
        """GroundingVerifier detects grounded vs unsupported claims."""
        knowledge_manager.add_document(temp_knowledge_dir / "axen_overview.md")
        chunks = knowledge_manager.retrieve("What is AXEN?")
        assert len(chunks) >= 1

        verifier = GroundingVerifier()

        # 1. Supported answer
        res_grounded = verifier.verify_grounding(
            "AXEN is a local-first AI assistant designed for consumer hardware with zero cloud dependencies.",
            chunks,
        )
        assert res_grounded.status == GroundingStatus.GROUNDED
        assert res_grounded.grounded_ratio > 0.70

        # 2. Unsupported / fabricated answer
        res_unsupported = verifier.verify_grounding(
            "AXEN is an extraterrestrial satellite operating on lunar solar panels in orbit around Mars.",
            chunks,
        )
        assert res_unsupported.status in (GroundingStatus.UNSUPPORTED, GroundingStatus.PARTIALLY_GROUNDED)
        assert len(res_unsupported.unsupported_claims) >= 1
