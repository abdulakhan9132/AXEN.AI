"""
axen/knowledge/vector_store.py
==============================
Thread-safe SQLite persistent storage for AXEN Knowledge Base:
- Collections table
- Documents table
- Chunks table
- Vector embeddings storage (binary blob / json)
- Full isolation by collection_id
- Transaction safety and WAL mode
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import struct
import threading
from pathlib import Path
from typing import Any, List, Optional, Tuple
import numpy as np

from .document_models import CollectionInfo, Document, DocumentChunk, DocumentType

logger = logging.getLogger("axen.knowledge.store")

DEFAULT_KNOWLEDGE_DB = Path("data/axen_knowledge.db")


class SQLiteVectorStore:
    """Persistent SQLite database storing documents, chunks, metadata, and embeddings."""

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_KNOWLEDGE_DB
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=15.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create tables and indexes with WAL mode."""
        with self._lock, self._get_connection() as conn:
            if str(self.db_path) != ":memory:":
                conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")

            # 1. Collections
            conn.execute("""
            CREATE TABLE IF NOT EXISTS collections (
                collection_id TEXT PRIMARY KEY,
                name          TEXT NOT NULL,
                description   TEXT DEFAULT '',
                created_at    TEXT NOT NULL
            );
            """)

            # 2. Documents
            conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id   TEXT PRIMARY KEY,
                filename      TEXT NOT NULL,
                filepath      TEXT NOT NULL,
                file_type     TEXT NOT NULL,
                file_size     INTEGER NOT NULL,
                content_hash  TEXT NOT NULL,
                collection_id TEXT NOT NULL,
                title         TEXT,
                created_at    TEXT NOT NULL,
                modified_at   TEXT NOT NULL,
                indexed_at    TEXT NOT NULL,
                chunk_count   INTEGER DEFAULT 0,
                status        TEXT DEFAULT 'indexed',
                metadata_json TEXT DEFAULT '{}'
            );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_collection ON documents(collection_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_hash ON documents(content_hash, collection_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_path ON documents(filepath, collection_id);")

            # 3. Document Chunks & Embeddings
            conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id        TEXT PRIMARY KEY,
                document_id     TEXT NOT NULL,
                chunk_index     INTEGER NOT NULL,
                collection_id   TEXT NOT NULL,
                text            TEXT NOT NULL,
                page_number     INTEGER,
                section_title   TEXT,
                source_location TEXT,
                token_estimate  INTEGER NOT NULL,
                embedding_blob  BLOB,
                metadata_json   TEXT DEFAULT '{}',
                FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
            );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_doc ON chunks(document_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_collection ON chunks(collection_id);")

            # Default collection
            conn.execute("""
            INSERT OR IGNORE INTO collections (collection_id, name, description, created_at)
            VALUES ('default', 'Default Knowledge', 'General knowledge sources', datetime('now'));
            """)
            conn.commit()

    # ------------------------------------------------------------------
    # Document Operations
    # ------------------------------------------------------------------

    def save_document(self, doc: Document) -> None:
        """Insert or update a document record."""
        sql = """
        INSERT INTO documents (
            document_id, filename, filepath, file_type, file_size, content_hash,
            collection_id, title, created_at, modified_at, indexed_at,
            chunk_count, status, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(document_id) DO UPDATE SET
            filename=excluded.filename,
            filepath=excluded.filepath,
            file_type=excluded.file_type,
            file_size=excluded.file_size,
            content_hash=excluded.content_hash,
            modified_at=excluded.modified_at,
            indexed_at=excluded.indexed_at,
            chunk_count=excluded.chunk_count,
            status=excluded.status,
            metadata_json=excluded.metadata_json;
        """
        with self._lock, self._get_connection() as conn:
            conn.execute(
                sql,
                (
                    doc.document_id,
                    doc.filename,
                    doc.filepath,
                    doc.file_type.value if hasattr(doc.file_type, "value") else str(doc.file_type),
                    doc.file_size,
                    doc.content_hash,
                    doc.collection_id,
                    doc.title,
                    doc.created_at,
                    doc.modified_at,
                    doc.indexed_at,
                    doc.chunk_count,
                    doc.status,
                    json.dumps(doc.metadata),
                ),
            )
            conn.commit()

    def get_document(self, document_id: str) -> Optional[Document]:
        """Fetch a document by ID."""
        sql = "SELECT * FROM documents WHERE document_id = ?;"
        with self._lock, self._get_connection() as conn:
            row = conn.execute(sql, (document_id,)).fetchone()
            return self._row_to_doc(dict(row)) if row else None

    def find_by_content_hash(self, content_hash: str, collection_id: Optional[str] = None) -> Optional[Document]:
        """Find if a document with this exact content hash already exists."""
        sql = "SELECT * FROM documents WHERE content_hash = ?"
        params: list[Any] = [content_hash]
        if collection_id:
            sql += " AND collection_id = ?"
            params.append(collection_id)
        sql += " LIMIT 1;"

        with self._lock, self._get_connection() as conn:
            row = conn.execute(sql, params).fetchone()
            return self._row_to_doc(dict(row)) if row else None

    def find_by_filepath(self, filepath: str, collection_id: Optional[str] = None) -> Optional[Document]:
        """Find document by source file path."""
        sql = "SELECT * FROM documents WHERE filepath = ?"
        params: list[Any] = [str(filepath)]
        if collection_id:
            sql += " AND collection_id = ?"
            params.append(collection_id)
        sql += " LIMIT 1;"

        with self._lock, self._get_connection() as conn:
            row = conn.execute(sql, params).fetchone()
            return self._row_to_doc(dict(row)) if row else None

    def list_documents(self, collection_id: Optional[str] = None) -> List[Document]:
        """List all documents, optionally filtered by collection."""
        sql = "SELECT * FROM documents"
        params: list[Any] = []
        if collection_id:
            sql += " WHERE collection_id = ?"
            params.append(collection_id)
        sql += " ORDER BY indexed_at DESC;"

        with self._lock, self._get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_doc(dict(r)) for r in rows]

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its associated chunks."""
        with self._lock, self._get_connection() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?;", (document_id,))
            cursor = conn.execute("DELETE FROM documents WHERE document_id = ?;", (document_id,))
            conn.commit()
            return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Chunk & Vector Operations
    # ------------------------------------------------------------------

    def save_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: Optional[List[List[float]]] = None,
    ) -> int:
        """Save chunks and their vector embeddings."""
        if not chunks:
            return 0

        # Remove existing chunks for this document first
        doc_id = chunks[0].document_id
        with self._lock, self._get_connection() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?;", (doc_id,))

            sql = """
            INSERT INTO chunks (
                chunk_id, document_id, chunk_index, collection_id, text,
                page_number, section_title, source_location, token_estimate,
                embedding_blob, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """

            for i, chunk in enumerate(chunks):
                blob = None
                if embeddings and i < len(embeddings):
                    # Pack floats into bytes
                    blob = struct.pack(f"{len(embeddings[i])}f", *embeddings[i])

                conn.execute(
                    sql,
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.chunk_index,
                        chunk.collection_id,
                        chunk.text,
                        chunk.page_number,
                        chunk.section_title,
                        chunk.source_location,
                        chunk.token_estimate,
                        blob,
                        json.dumps(chunk.metadata),
                    ),
                )
            conn.commit()
            return len(chunks)

    def search_vectors(
        self,
        query_vector: List[float],
        collection_id: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Cosine similarity search over stored chunk embeddings.
        Returns top_k tuples of (DocumentChunk, cosine_similarity).
        """
        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm < 1e-6:
            return []

        sql = "SELECT * FROM chunks WHERE embedding_blob IS NOT NULL"
        params: list[Any] = []
        if collection_id:
            sql += " AND collection_id = ?"
            params.append(collection_id)

        with self._lock, self._get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()

        if not rows:
            return []

        candidates: list[Tuple[DocumentChunk, float]] = []
        dim = len(query_vector)

        for row in rows:
            blob = row["embedding_blob"]
            if not blob or len(blob) != dim * 4:
                continue

            chunk_vec = np.array(struct.unpack(f"{dim}f", blob), dtype=np.float32)
            c_norm = np.linalg.norm(chunk_vec)
            if c_norm < 1e-6:
                continue

            # Cosine similarity
            sim = float(np.dot(q_vec, chunk_vec) / (q_norm * c_norm))
            chunk = self._row_to_chunk(dict(row))
            candidates.append((chunk, sim))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def search_lexical(
        self,
        keywords: List[str],
        collection_id: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Keyword / BM25-style lexical matching over chunk text with whole-word boundaries."""
        valid_kws = [k.lower().strip() for k in keywords if k.lower().strip()]
        if not valid_kws:
            return []

        sql = "SELECT * FROM chunks WHERE 1=1"
        params: list[Any] = []
        if collection_id:
            sql += " AND collection_id = ?"
            params.append(collection_id)

        with self._lock, self._get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()

        scored: list[Tuple[DocumentChunk, float]] = []
        for row in rows:
            text_lower = row["text"].lower()
            sec_title = (row["section_title"] or "").lower()
            matched_kws = 0
            term_score = 0.0

            for kw in valid_kws:
                # Use regex word boundaries for precise matching
                matches = len(re.findall(r"\b" + re.escape(kw) + r"\b", text_lower))
                if matches > 0:
                    matched_kws += 1
                    term_score += 1.0 + min(2.0, math.log(1.0 + matches))
                    if sec_title and re.search(r"\b" + re.escape(kw) + r"\b", sec_title):
                        term_score += 1.0

            if matched_kws > 0:
                coverage = matched_kws / len(valid_kws)
                avg_term_qual = min(1.0, term_score / (matched_kws * 2.0))
                cov_weight = (coverage ** 2.0) if coverage < 0.5 else coverage
                score = cov_weight * avg_term_qual
                if score >= 0.05:
                    scored.append((self._row_to_chunk(dict(row)), score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Collections
    # ------------------------------------------------------------------

    def list_collections(self) -> List[CollectionInfo]:
        """List all collections with document and chunk counts."""
        sql = """
        SELECT c.collection_id, c.name, c.description, c.created_at,
               COUNT(DISTINCT d.document_id) AS doc_count,
               COUNT(k.chunk_id) AS chunk_count
        FROM collections c
        LEFT JOIN documents d ON c.collection_id = d.collection_id
        LEFT JOIN chunks k ON c.collection_id = k.collection_id
        GROUP BY c.collection_id;
        """
        with self._lock, self._get_connection() as conn:
            rows = conn.execute(sql).fetchall()
            return [
                CollectionInfo(
                    collection_id=r["collection_id"],
                    name=r["name"],
                    description=r["description"] or "",
                    created_at=r["created_at"],
                    document_count=r["doc_count"],
                    chunk_count=r["chunk_count"],
                )
                for r in rows
            ]

    def create_collection(self, collection_id: str, name: str, description: str = "") -> CollectionInfo:
        """Create a new collection."""
        sql = """
        INSERT INTO collections (collection_id, name, description, created_at)
        VALUES (?, ?, ?, datetime('now'))
        ON CONFLICT(collection_id) DO UPDATE SET name=excluded.name, description=excluded.description;
        """
        with self._lock, self._get_connection() as conn:
            conn.execute(sql, (collection_id, name, description))
            conn.commit()
        return CollectionInfo(collection_id=collection_id, name=name, description=description)

    def clear_collection(self, collection_id: str) -> int:
        """Remove all documents and chunks from a collection."""
        with self._lock, self._get_connection() as conn:
            conn.execute("DELETE FROM chunks WHERE collection_id = ?;", (collection_id,))
            cur = conn.execute("DELETE FROM documents WHERE collection_id = ?;", (collection_id,))
            conn.commit()
            return cur.rowcount

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_doc(row: dict[str, Any]) -> Document:
        meta = json.loads(row.get("metadata_json") or "{}")
        return Document(
            document_id=row["document_id"],
            filename=row["filename"],
            filepath=row["filepath"],
            file_type=DocumentType(row["file_type"]),
            file_size=row["file_size"],
            content_hash=row["content_hash"],
            collection_id=row["collection_id"],
            title=row.get("title"),
            created_at=row["created_at"],
            modified_at=row["modified_at"],
            indexed_at=row["indexed_at"],
            chunk_count=row.get("chunk_count", 0),
            status=row.get("status", "indexed"),
            metadata=meta,
        )

    @staticmethod
    def _row_to_chunk(row: dict[str, Any]) -> DocumentChunk:
        meta = json.loads(row.get("metadata_json") or "{}")
        return DocumentChunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            text=row["text"],
            collection_id=row["collection_id"],
            page_number=row.get("page_number"),
            section_title=row.get("section_title"),
            source_location=row.get("source_location"),
            token_estimate=row.get("token_estimate", 0),
            metadata=meta,
        )
