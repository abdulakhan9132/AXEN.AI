"""
axen/knowledge/document_loader.py
=================================
Extensible, local document loader supporting:
- Plain Text (TXT)
- Markdown (MD)
- PDF Documents (PDF)
- Microsoft Word (DOCX)
- Tabular Data (CSV)
- Structured Data (JSON)
- Source Code (Python, JS, TS, HTML, CSS, C++, Rust, Go, etc.)

Extracts raw text, detects structure, and tags metadata safely and offline.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import xml.etree.ElementTree as ET
import zipfile
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .document_models import Document, DocumentType, compute_content_hash
from .knowledge_policy import KnowledgePolicy

logger = logging.getLogger("axen.knowledge.loader")


class DocumentLoader:
    """Loads, extracts, and normalizes text from local files."""

    _EXTENSION_MAP = {
        ".txt": DocumentType.TXT,
        ".text": DocumentType.TXT,
        ".log": DocumentType.TXT,
        ".md": DocumentType.MARKDOWN,
        ".markdown": DocumentType.MARKDOWN,
        ".pdf": DocumentType.PDF,
        ".docx": DocumentType.DOCX,
        ".csv": DocumentType.CSV,
        ".json": DocumentType.JSON,
        ".py": DocumentType.CODE,
        ".js": DocumentType.CODE,
        ".ts": DocumentType.CODE,
        ".html": DocumentType.CODE,
        ".css": DocumentType.CODE,
        ".cpp": DocumentType.CODE,
        ".c": DocumentType.CODE,
        ".h": DocumentType.CODE,
        ".rs": DocumentType.CODE,
        ".go": DocumentType.CODE,
        ".java": DocumentType.CODE,
        ".sh": DocumentType.CODE,
        ".sql": DocumentType.CODE,
    }

    @classmethod
    def detect_type(cls, file_path: Path | str) -> DocumentType:
        """Infer DocumentType from file extension."""
        ext = Path(file_path).suffix.lower()
        return cls._EXTENSION_MAP.get(ext, DocumentType.OTHER)

    @classmethod
    def load(
        cls,
        file_path: Path | str,
        collection_id: str = "default",
        custom_metadata: Optional[dict[str, Any]] = None,
    ) -> Tuple[Optional[Document], str, Optional[str]]:
        """
        Load a file and extract its text content and metadata.
        Returns: (document_obj, extracted_text, error_message)
        """
        path = Path(file_path).resolve()

        # 1. Security policy check
        is_safe, reason = KnowledgePolicy.validate_file_for_ingestion(path)
        if not is_safe:
            return None, "", reason

        dtype = cls.detect_type(path)

        try:
            # 2. Format-specific extraction
            if dtype in (DocumentType.TXT, DocumentType.MARKDOWN):
                text = cls._load_text(path)
            elif dtype == DocumentType.PDF:
                text = cls._load_pdf(path)
            elif dtype == DocumentType.DOCX:
                text = cls._load_docx(path)
            elif dtype == DocumentType.CSV:
                text = cls._load_csv(path)
            elif dtype == DocumentType.JSON:
                text = cls._load_json(path)
            elif dtype == DocumentType.CODE:
                text = cls._load_text(path)
            else:
                # Fallback to UTF-8 text attempt
                text = cls._load_text(path)

            if not text.strip():
                return None, "", "File produced empty text output."

            # 3. Content security check
            allowed, text, warn = KnowledgePolicy.sanitize_text(text)
            if not allowed:
                return None, "", warn

            content_hash = compute_content_hash(text)
            file_stat = path.stat()

            doc_id = f"doc_{content_hash[:16]}"
            meta = custom_metadata or {}
            meta.update({
                "original_path": str(path),
                "extension": path.suffix.lower(),
                "char_count": len(text),
                "line_count": len(text.splitlines()),
            })

            doc = Document(
                document_id=doc_id,
                filename=path.name,
                filepath=str(path),
                file_type=dtype,
                file_size=file_stat.st_size,
                content_hash=content_hash,
                collection_id=collection_id,
                title=path.stem.replace("_", " ").title(),
                modified_at=cls._format_timestamp(file_stat.st_mtime),
                metadata=meta,
            )

            return doc, text, None

        except Exception as exc:
            logger.error("Failed to load document %s: %s", path, exc)
            return None, "", f"Document load error: {exc}"

    # ------------------------------------------------------------------
    # Format Extractors
    # ------------------------------------------------------------------

    @classmethod
    def _load_text(cls, path: Path) -> str:
        """Load standard UTF-8 text."""
        return path.read_text(encoding="utf-8", errors="replace")

    @classmethod
    def _load_pdf(cls, path: Path) -> str:
        """
        Extract text from PDF.
        Attempts pypdf if available, otherwise uses built-in FlateDecode stream parser.
        """
        # 1. Try pypdf if installed
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            pages = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append(f"[Page {i+1}]\n{page_text}")
            if pages:
                return "\n\n".join(pages)
        except ImportError:
            pass

        # 2. Pure-Python fallback using raw stream inflation and text extraction
        with open(path, "rb") as f:
            raw = f.read()

        text_parts = []
        # Find compressed streams
        stream_matches = re.finditer(b"stream\r?\n(.*?)\r?\nendstream", raw, re.DOTALL)
        page_idx = 1
        for match in stream_matches:
            data = match.group(1)
            decompressed = None
            try:
                decompressed = zlib.decompress(data)
            except Exception:
                try:
                    decompressed = zlib.decompress(data, -zlib.MAX_WBITS)
                except Exception:
                    decompressed = None

            if decompressed:
                try:
                    stream_str = decompressed.decode("latin1", errors="ignore")
                    # Extract strings inside parentheses after operators or TJ arrays
                    # e.g. (Hello World) Tj or [(Hello) -10 (World)] TJ
                    tj_strings = re.findall(r"\((.*?)\)\s*Tj", stream_str)
                    array_strings = re.findall(r"\[(.*?)\]\s*TJ", stream_str)
                    for arr in array_strings:
                        sub_strs = re.findall(r"\((.*?)\)", arr)
                        tj_strings.extend(sub_strs)

                    if tj_strings:
                        clean_extracted = " ".join(re.sub(r"\\[0-9]{3}|\\.", " ", s) for s in tj_strings)
                        clean_extracted = re.sub(r"\s+", " ", clean_extracted).strip()
                        if len(clean_extracted) > 10:
                            text_parts.append(f"[Page {page_idx}]\n{clean_extracted}")
                            page_idx += 1
                except Exception:
                    continue

        if text_parts:
            return "\n\n".join(text_parts)

        # 3. Direct ASCII string scanner fallback
        ascii_matches = re.findall(r"\(([A-Za-z0-9\s,\.\?\!\-\:;]{4,})\)", raw.decode("latin1", errors="ignore"))
        if ascii_matches:
            return " ".join(ascii_matches)

        return f"[PDF: {path.name} - Text could not be extracted from scanned/rasterized binary stream]"

    @classmethod
    def _load_docx(cls, path: Path) -> str:
        """
        Extract text from DOCX (ZIP archive containing word/document.xml).
        Zero external library required.
        """
        with zipfile.ZipFile(path) as z:
            xml_content = z.read("word/document.xml")

        tree = ET.fromstring(xml_content)
        # Word XML namespaces
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

        paragraphs = []
        for p in tree.iterfind(".//w:p", ns):
            texts = [node.text for node in p.iterfind(".//w:t", ns) if node.text]
            if texts:
                paragraphs.append("".join(texts))

        return "\n\n".join(paragraphs)

    @classmethod
    def _load_csv(cls, path: Path) -> str:
        """Load tabular CSV with headers and row context."""
        rows = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = None
            for i, row in enumerate(reader):
                if i == 0:
                    header = row
                    rows.append("Header: " + ", ".join(header))
                else:
                    if header and len(row) == len(header):
                        pairs = [f"{header[j]}: {row[j]}" for j in range(len(header))]
                        rows.append(f"Row {i}: " + "; ".join(pairs))
                    else:
                        rows.append(f"Row {i}: " + ", ".join(row))
        return "\n".join(rows)

    @classmethod
    def _load_json(cls, path: Path) -> str:
        """Load and prettify structured JSON."""
        content = path.read_text(encoding="utf-8", errors="replace")
        data = json.loads(content)
        return json.dumps(data, indent=2)

    @staticmethod
    def _format_timestamp(ts: float) -> str:
        from datetime import datetime, timezone
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
