"""
axen/storage/storage_models.py
==============================
Data structures and configuration for the AXEN local file storage and upload system.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class FileStatus(str, Enum):
    QUEUED = "queued"
    UPLOADING = "uploading"
    VALIDATING = "validating"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class FileCategory(str, Enum):
    DOCUMENT = "document"
    DATA = "data"
    CODE = "code"
    IMAGE = "image"
    UNSUPPORTED = "unsupported"


@dataclass
class UploadConfig:
    """Centralized configuration limits for local file uploads."""
    max_file_size_bytes: int = 25 * 1024 * 1024       # 25 MB per file
    max_total_upload_bytes: int = 100 * 1024 * 1024   # 100 MB total per batch
    max_files_per_message: int = 10                   # Max 10 files attached at once
    allowed_extensions: List[str] = field(default_factory=lambda: [
        # Documents
        ".pdf", ".txt", ".md", ".markdown", ".docx",
        # Data
        ".csv", ".json",
        # Code
        ".py", ".js", ".ts", ".html", ".css", ".yaml", ".yml", ".xml",
        ".sql", ".java", ".c", ".cpp", ".h", ".hpp", ".sh", ".ps1",
        # Images
        ".png", ".jpg", ".jpeg", ".webp",
    ])


@dataclass
class UploadedFile:
    """Represents an uploaded file stored locally in AXEN."""
    file_id: str
    original_filename: str
    stored_filename: str
    file_path: str
    size_bytes: int
    sha256: str
    extension: str
    mime_type: str = "application/octet-stream"
    category: FileCategory = FileCategory.DOCUMENT
    status: FileStatus = FileStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    extracted_text: str = ""
    is_truncated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def filename(self) -> str:
        return self.original_filename

    @property
    def size(self) -> int:
        return self.size_bytes

    @property
    def text(self) -> str:
        return self.extracted_text

    @property
    def line_count(self) -> int:
        return int(self.metadata.get("line_count", 0))

    @property
    def file_type(self) -> str:
        return self.extension.lstrip(".").lower()


    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "file_id": self.file_id,
            "filename": self.original_filename,
            "stored_filename": self.stored_filename,
            "size": self.size_bytes,
            "size_bytes": self.size_bytes,
            "size_formatted": self.format_size(),
            "sha256": self.sha256,
            "extension": self.extension,
            "mime_type": self.mime_type,
            "category": self.category.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "is_truncated": self.is_truncated,
            "text": self.extracted_text,
            "extracted_text": self.extracted_text,
            "text_length": len(self.extracted_text),
            "metadata": self.metadata,
            "error": self.error,
        }

    def format_size(self) -> str:
        """Return human-readable file size."""
        b = self.size_bytes
        if b < 1024:
            return f"{b} B"
        elif b < 1024 * 1024:
            return f"{b / 1024:.1f} KB"
        else:
            return f"{b / (1024 * 1024):.1f} MB"


# Convenient alias
ProcessedContent = UploadedFile


file_logger = logging.getLogger("axen.storage.events")


def log_file_event(
    event: str,
    request_id: Optional[str] = None,
    file_id: Optional[str] = None,
    filename: Optional[str] = None,
    processing_time: Optional[float] = None,
    status: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log structured event for file observability without logging sensitive document contents.
    Events: upload_started, upload_validated, upload_failed, file_processed,
            file_processing_failed, file_deleted, file_indexed, file_search, file_retrieved
    """
    payload = {
        "event": event,
        "request_id": request_id or "",
        "file_id": file_id or "",
        "filename": filename or "",
        "processing_time": round(processing_time, 4) if processing_time is not None else None,
        "status": status or "",
    }
    if extra:
        payload["details"] = extra
    file_logger.info("AXEN_FILE_OBSERVABILITY: %s", json.dumps(payload))

