"""
axen/storage/file_processor.py
==============================
Modular, safe text and metadata extraction pipeline for uploaded files.

Supports:
- Documents: PDF (PDFProcessor), TXT (TXTProcessor), Markdown (MarkdownProcessor), DOCX (DOCXProcessor)
- Data: CSV (CSVProcessor), JSON (JSONProcessor)
- Code: Python, JS, TS, HTML, CSS, C/C++, Java, SQL, Shell, etc. (CodeProcessor)
- Images: Metadata and dimension extraction (ImageProcessor)
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from axen.storage.storage_models import (
    FileCategory,
    FileStatus,
    UploadedFile,
    ProcessedContent,
    log_file_event,
)

logger = logging.getLogger("axen.storage.processor")


class TXTProcessor:
    """Extracts text and line counts from plain text files."""

    def extract(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = len(text.splitlines())
        return text, {"line_count": lines, "format": "text"}


class MarkdownProcessor:
    """Extracts text and top-level headings from Markdown files."""

    def extract(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        text = path.read_text(encoding="utf-8", errors="replace")
        headers = re.findall(r"^#{1,3}\s+(.+)$", text, re.MULTILINE)
        return text, {"headings": headers[:15], "format": "markdown"}


class PDFProcessor:
    """Extracts text from PDF documents using pypdf, pymupdf, or regex stream fallback."""

    def extract(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        text_parts: List[str] = []
        page_count = 0

        # Try pypdf
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            page_count = len(reader.pages)
            for idx, page in enumerate(reader.pages):
                ptxt = page.extract_text() or ""
                if ptxt.strip():
                    text_parts.append(f"--- Page {idx + 1} ---\n{ptxt.strip()}")
            if text_parts:
                return "\n\n".join(text_parts), {"page_count": page_count, "format": "pdf"}
        except ImportError:
            pass

        # Try pymupdf (fitz)
        try:
            import fitz
            doc = fitz.open(str(path))
            page_count = len(doc)
            for idx in range(page_count):
                ptxt = doc[idx].get_text()
                if ptxt.strip():
                    text_parts.append(f"--- Page {idx + 1} ---\n{ptxt.strip()}")
            if text_parts:
                return "\n\n".join(text_parts), {"page_count": page_count, "format": "pdf"}
        except ImportError:
            pass

        # Robust regex stream fallback
        raw_bytes = path.read_bytes()
        import zlib
        stream_pat = re.compile(b"stream[\r\n]+(.*?)[\r\n]+endstream", re.DOTALL)
        for m in stream_pat.finditer(raw_bytes):
            try:
                decomp = zlib.decompress(m.group(1))
                text_matches = re.findall(rb"\((.*?)\)\s*Tj|\[(.*?)\]\s*TJ", decomp)
                for t, tarr in text_matches:
                    chunk = (t or tarr).decode("latin-1", errors="ignore")
                    clean = re.sub(r"[^a-zA-Z0-9\s.,;:!?'\"()\-_]", "", chunk).strip()
                    if len(clean) > 3:
                        text_parts.append(clean)
            except Exception:
                continue

        extracted = " ".join(text_parts) if text_parts else "[PDF document loaded. Text extraction limited.]"
        return extracted, {"page_count": max(1, page_count), "format": "pdf"}


class DOCXProcessor:
    """Extracts text from Word (.docx) documents via internal XML."""

    def extract(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        try:
            with zipfile.ZipFile(path, "r") as z:
                xml_content = z.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            paragraphs = []
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            for p in tree.iter(f"{{{ns['w']}}}p"):
                texts = [node.text for node in p.iter(f"{{{ns['w']}}}t") if node.text]
                if texts:
                    paragraphs.append("".join(texts))
            full_text = "\n\n".join(paragraphs)
            return full_text, {"paragraph_count": len(paragraphs), "format": "docx"}
        except Exception as exc:
            return f"[DOCX parsing error: {exc}]", {"format": "docx", "error": str(exc)}


class CSVProcessor:
    """Extracts tabular CSV data with headers, row count, and formatted markdown table preview."""

    def extract(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        content = path.read_text(encoding="utf-8", errors="replace")
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
        if not rows:
            return "", {"row_count": 0, "format": "csv"}

        headers = rows[0]
        data_rows = rows[1:]
        row_count = len(data_rows)

        lines = [
            f"CSV Table: {path.name}",
            f"Columns ({len(headers)}): {', '.join(headers)}",
            f"Total Rows: {row_count}\n",
        ]

        # Preview first 30 rows in readable markdown format
        preview_rows = [f"| {' | '.join(headers)} |", f"| {' | '.join(['---'] * len(headers))} |"]
        for r in data_rows[:30]:
            cells = (r + [""] * len(headers))[:len(headers)]
            preview_rows.append(f"| {' | '.join(cells)} |")

        if row_count > 30:
            preview_rows.append(f"\n... and {row_count - 30} more rows.")

        lines.extend(preview_rows)
        return "\n".join(lines), {"headers": headers, "row_count": row_count, "format": "csv"}


class JSONProcessor:
    """Extracts and formats structured JSON documents."""

    def extract(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        content = path.read_text(encoding="utf-8", errors="replace")
        try:
            data = json.loads(content)
            formatted = json.dumps(data, indent=2)
            keys = list(data.keys()) if isinstance(data, dict) else [f"array_length={len(data)}"]
            return formatted, {"top_level_keys": keys[:10], "format": "json"}
        except json.JSONDecodeError as err:
            return content, {"format": "json", "error": f"JSON syntax error: {err}"}


class CodeProcessor:
    """Extracts source code, line counts, function/class metrics."""

    def extract(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        code = path.read_text(encoding="utf-8", errors="replace")
        lines = code.splitlines()
        classes = len(re.findall(r"^\s*class\s+\w+", code, re.MULTILINE))
        funcs = len(re.findall(r"^\s*(?:def|async def|function)\s+\w+", code, re.MULTILINE))

        return code, {
            "line_count": len(lines),
            "class_count": classes,
            "function_count": funcs,
            "format": "code",
        }


class ImageProcessor:
    """Extracts image metadata and dimensions (preparing for multimodal support)."""

    def extract(self, path: Path, original_filename: str = "") -> Tuple[str, Dict[str, Any]]:
        width, height = 0, 0
        img_format = path.suffix.upper().lstrip(".")

        try:
            from PIL import Image
            with Image.open(path) as im:
                width, height = im.size
                img_format = im.format or img_format
        except Exception:
            pass

        display_name = original_filename or path.name
        info_text = (
            f"[Image Attachment: {display_name}]\n"
            f"Format: {img_format}\n"
            f"Dimensions: {width}x{height}\n"
            f"Size: {path.stat().st_size} bytes\n"
            "Note: AXEN's current local text model (TinyLlama) processes metadata and text descriptions. "
            "Vision analysis will be supported in multimodal phases."
        )
        return info_text, {
            "width": width,
            "height": height,
            "format": img_format.lower(),
        }


class FileProcessor:
    """Extracts normalized text, structure, and metadata from uploaded files."""

    MAX_TEXT_CHARS: int = 40000  # Context protection limit

    PROMPT_INJECTION_PATTERNS: List[re.Pattern] = [
        re.compile(r"\b(?:ignore|forget|disregard)\s+(?:all\s+)?(?:previous|prior)\s+instructions\b", re.IGNORECASE),
        re.compile(r"\b(?:you\s+are\s+now|act\s+as)\s+(?:a|an)?\s*(?:unrestricted|evil|dan|jailbreak)\b", re.IGNORECASE),
        re.compile(r"\b(?:reveal|show|dump|print)\s+(?:your\s+)?(?:system\s+prompt|hidden\s+instructions|secret\s+key)\b", re.IGNORECASE),
        re.compile(r"\b(?:send|exfiltrate|post|upload)\s+(?:secrets|passwords|credentials)\b", re.IGNORECASE),
        re.compile(r"\b(?:run|execute)\s+(?:the\s+following\s+)?(?:command|shell|script|cmd|powershell)\b", re.IGNORECASE),
    ]

    def __init__(self) -> None:
        self.txt_processor = TXTProcessor()
        self.markdown_processor = MarkdownProcessor()
        self.pdf_processor = PDFProcessor()
        self.docx_processor = DOCXProcessor()
        self.csv_processor = CSVProcessor()
        self.json_processor = JSONProcessor()
        self.code_processor = CodeProcessor()
        self.image_processor = ImageProcessor()

    def process(self, file_obj: UploadedFile) -> UploadedFile:
        """
        Execute specialized extractor based on file category and extension.
        Updates file_obj in-place, records structured event, and returns it.
        """
        t0 = time.perf_counter()
        file_path = Path(file_obj.file_path)
        if not file_path.exists():
            file_obj.status = FileStatus.FAILED
            file_obj.error = "File does not exist on disk."
            log_file_event(
                "file_processing_failed",
                file_id=file_obj.file_id,
                filename=file_obj.original_filename,
                processing_time=time.perf_counter() - t0,
                status="failed",
                extra={"error": file_obj.error},
            )
            return file_obj

        file_obj.status = FileStatus.PROCESSING

        try:
            raw_text, meta = self._extract_content(
                file_path,
                file_obj.extension,
                file_obj.category,
                original_filename=file_obj.original_filename,
            )

            # Sanitize against prompt injection
            sanitized_text, has_inj = self._sanitize_text(raw_text)

            # Enforce size bounds
            is_truncated = False
            if len(sanitized_text) > self.MAX_TEXT_CHARS:
                sanitized_text = sanitized_text[:self.MAX_TEXT_CHARS] + "\n...[Document truncated for context safety]"
                is_truncated = True

            file_obj.extracted_text = sanitized_text
            file_obj.is_truncated = is_truncated
            file_obj.metadata.update(meta)
            if has_inj:
                file_obj.metadata["prompt_injection_detected"] = True

            file_obj.status = FileStatus.READY
            dur = time.perf_counter() - t0
            log_file_event(
                "file_processed",
                file_id=file_obj.file_id,
                filename=file_obj.original_filename,
                processing_time=dur,
                status="ready",
                extra={"text_length": len(sanitized_text), "category": file_obj.category.value},
            )

        except Exception as exc:
            dur = time.perf_counter() - t0
            logger.error("Processing failed for %s: %s", file_obj.file_id, exc)
            file_obj.status = FileStatus.FAILED
            file_obj.error = str(exc)
            log_file_event(
                "file_processing_failed",
                file_id=file_obj.file_id,
                filename=file_obj.original_filename,
                processing_time=dur,
                status="failed",
                extra={"error": str(exc)},
            )

        return file_obj

    def _extract_content(
        self,
        path: Path,
        ext: str,
        category: FileCategory,
        original_filename: str = "",
    ) -> Tuple[str, Dict[str, Any]]:
        """Dispatch extraction by extension and category to specialized processor."""
        ext_lower = ext.lower()

        if ext_lower == ".pdf":
            return self.pdf_processor.extract(path)
        elif ext_lower in (".txt", ".text", ".log"):
            return self.txt_processor.extract(path)
        elif ext_lower in (".md", ".markdown"):
            return self.markdown_processor.extract(path)
        elif ext_lower == ".docx":
            return self.docx_processor.extract(path)
        elif ext_lower == ".csv":
            return self.csv_processor.extract(path)
        elif ext_lower == ".json":
            return self.json_processor.extract(path)
        elif category == FileCategory.CODE:
            return self.code_processor.extract(path)
        elif category == FileCategory.IMAGE:
            return self.image_processor.extract(path, original_filename=original_filename)
        else:
            return self.txt_processor.extract(path)

    def _sanitize_text(self, text: str) -> Tuple[str, bool]:
        """Detect and quarantine adversarial prompt injection inside documents."""
        if not text:
            return "", False

        detected = False
        for pat in self.PROMPT_INJECTION_PATTERNS:
            if pat.search(text):
                detected = True
                break

        if detected:
            sanitized = f"[DOCUMENT CONTENT CONTAINS UNTRUSTED TEXT - INSTRUCTIONS NEUTRALIZED]\n{text}"
            return sanitized, True

        return text, False
