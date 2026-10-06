"""
axen/knowledge/document_chunker.py
==================================
Structure-aware document chunking for AXEN RAG.

Splits documents intelligently based on document type:
- Markdown: headings (#, ##, ###), sections, paragraphs
- Code: class and function boundaries
- PDF: page markers and paragraphs
- Text / CSV / JSON: semantic paragraphs with bounded sliding window

Preserves metadata:
- chunk_id, document_id, chunk_index
- section_title, page_number, source_location
- token_estimate
"""

from __future__ import annotations

import re
from typing import List, Optional

from .document_models import Document, DocumentChunk, DocumentType


class DocumentChunker:
    """Structure-aware chunker producing bounded semantic passages."""

    def __init__(
        self,
        target_chunk_chars: int = 800,
        max_chunk_chars: int = 1200,
        min_chunk_chars: int = 80,
        overlap_chars: int = 100,
    ) -> None:
        self.target_chunk_chars = target_chunk_chars
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars
        self.overlap_chars = overlap_chars

    def chunk_document(self, document: Document, text: str) -> List[DocumentChunk]:
        """Split document text into bounded, structure-aware DocumentChunk items."""
        dtype = document.file_type

        if dtype == DocumentType.MARKDOWN:
            raw_chunks = self._chunk_markdown(text)
        elif dtype == DocumentType.CODE:
            raw_chunks = self._chunk_code(text, document.filename)
        elif dtype == DocumentType.PDF:
            raw_chunks = self._chunk_pdf(text)
        else:
            raw_chunks = self._chunk_paragraphs(text)

        # Assemble DocumentChunk objects
        chunks: List[DocumentChunk] = []
        for i, (chunk_text, section, page, loc) in enumerate(raw_chunks):
            clean_text = chunk_text.strip()
            if len(clean_text) < self.min_chunk_chars and len(raw_chunks) > 1:
                # Merge tiny trailing chunks into previous chunk if within budget
                if chunks and len(chunks[-1].text) + len(clean_text) < self.max_chunk_chars:
                    chunks[-1].text += "\n" + clean_text
                    chunks[-1].token_estimate = max(1, len(chunks[-1].text) // 4)
                    continue

            chunk_id = f"{document.document_id}_c{i}"
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                document_id=document.document_id,
                chunk_index=i,
                text=clean_text,
                collection_id=document.collection_id,
                page_number=page,
                section_title=section,
                source_location=loc or section or (f"Page {page}" if page else f"Part {i+1}"),
                token_estimate=max(1, len(clean_text) // 4),
                metadata={
                    "filename": document.filename,
                    "filepath": document.filepath,
                    "collection_id": document.collection_id,
                },
            )
            chunks.append(chunk)

        # Ensure at least 1 chunk even for short documents
        if not chunks and text.strip():
            chunk_id = f"{document.document_id}_c0"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    chunk_index=0,
                    text=text.strip(),
                    collection_id=document.collection_id,
                    source_location=document.filename,
                    token_estimate=max(1, len(text.strip()) // 4),
                    metadata={"filename": document.filename, "filepath": document.filepath},
                )
            )

        return chunks

    # ------------------------------------------------------------------
    # Markdown Structure Chunking
    # ------------------------------------------------------------------

    def _chunk_markdown(self, text: str) -> list[tuple[str, Optional[str], Optional[int], Optional[str]]]:
        """Split Markdown along headers (#, ##, ###) and paragraphs."""
        lines = text.splitlines()
        sections: list[tuple[str, str]] = []  # (heading, content)
        current_heading = "Overview"
        current_lines: list[str] = []

        heading_pattern = re.compile(r"^(#{1,4})\s+(.+)$")

        for line in lines:
            h_match = heading_pattern.match(line.strip())
            if h_match:
                if current_lines:
                    sections.append((current_heading, "\n".join(current_lines)))
                    current_lines = []
                current_heading = h_match.group(2).strip()
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((current_heading, "\n".join(current_lines)))

        # Subdivide sections into bounded chunks
        results: list[tuple[str, Optional[str], Optional[int], Optional[str]]] = []
        for heading, sec_text in sections:
            paras = [p.strip() for p in sec_text.split("\n\n") if p.strip()]
            buf = f"### {heading}\n"
            for p in paras:
                if len(buf) + len(p) > self.max_chunk_chars:
                    results.append((buf.strip(), heading, None, f"Section: {heading}"))
                    # Overlap
                    buf = f"### {heading} (cont.)\n" + p + "\n\n"
                else:
                    buf += p + "\n\n"
            if buf.strip():
                results.append((buf.strip(), heading, None, f"Section: {heading}"))

        return results

    # ------------------------------------------------------------------
    # Code Chunking
    # ------------------------------------------------------------------

    def _chunk_code(self, text: str, filename: str) -> list[tuple[str, Optional[str], Optional[int], Optional[str]]]:
        """Split code files by classes, functions, or block limits."""
        lines = text.splitlines()
        chunks: list[tuple[str, Optional[str], Optional[int], Optional[str]]] = []

        class_pattern = re.compile(r"^class\s+([a-zA-Z0-9_]+)")
        func_pattern = re.compile(r"^\s*(?:async\s+def|def)\s+([a-zA-Z0-9_]+)")
        current_class: Optional[str] = None
        current_symbol = "Module"
        current_lines: list[str] = []

        for line in lines:
            cm = class_pattern.match(line)
            fm = func_pattern.match(line)
            if cm:
                current_class = cm.group(1)
                new_symbol = current_class
                if current_lines and len("\n".join(current_lines)) >= self.min_chunk_chars:
                    loc = f"{filename} — {current_symbol}"
                    chunks.append(("\n".join(current_lines), current_symbol, None, loc))
                    current_lines = []
                current_symbol = new_symbol
            elif fm:
                fname = fm.group(1)
                new_symbol = f"{current_class}.{fname}" if current_class else fname
                if current_lines and len("\n".join(current_lines)) >= self.min_chunk_chars:
                    loc = f"{filename} — {current_symbol}"
                    chunks.append(("\n".join(current_lines), current_symbol, None, loc))
                    current_lines = []
                current_symbol = new_symbol

            current_lines.append(line)

            if len("\n".join(current_lines)) > self.max_chunk_chars:
                loc = f"{filename} — {current_symbol}"
                chunks.append(("\n".join(current_lines), current_symbol, None, loc))
                current_lines = []

        if current_lines:
            loc = f"{filename} — {current_symbol}"
            chunks.append(("\n".join(current_lines), current_symbol, None, loc))

        return chunks

    # ------------------------------------------------------------------
    # PDF Page & Paragraph Chunking
    # ------------------------------------------------------------------

    def _chunk_pdf(self, text: str) -> list[tuple[str, Optional[str], Optional[int], Optional[str]]]:
        """Split PDF text by page markers [Page N] and paragraphs."""
        pages = re.split(r"\[Page\s+(\d+)\]", text)
        results: list[tuple[str, Optional[str], Optional[int], Optional[str]]] = []

        if len(pages) > 1:
            # pages alternate: [preamble, page_num_1, page_text_1, page_num_2, page_text_2...]
            for i in range(1, len(pages), 2):
                page_num = int(pages[i])
                page_content = pages[i + 1].strip()
                paras = [p.strip() for p in page_content.split("\n\n") if p.strip()]
                buf = ""
                for p in paras:
                    if len(buf) + len(p) > self.max_chunk_chars:
                        results.append((buf.strip(), f"Page {page_num}", page_num, f"Page {page_num}"))
                        buf = p + "\n\n"
                    else:
                        buf += p + "\n\n"
                if buf.strip():
                    results.append((buf.strip(), f"Page {page_num}", page_num, f"Page {page_num}"))
        else:
            return self._chunk_paragraphs(text)

        return results

    # ------------------------------------------------------------------
    # Generic Paragraphs / Sliding Window
    # ------------------------------------------------------------------

    def _chunk_paragraphs(self, text: str) -> list[tuple[str, Optional[str], Optional[int], Optional[str]]]:
        """Split generic text by paragraphs into bounded chunks."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        results: list[tuple[str, Optional[str], Optional[int], Optional[str]]] = []

        buf = ""
        part_idx = 1
        for p in paragraphs:
            if len(buf) + len(p) > self.max_chunk_chars:
                results.append((buf.strip(), None, None, f"Part {part_idx}"))
                part_idx += 1
                buf = p + "\n\n"
            else:
                buf += p + "\n\n"

        if buf.strip():
            results.append((buf.strip(), None, None, f"Part {part_idx}"))

        return results
