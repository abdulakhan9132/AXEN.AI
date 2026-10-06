"""
axen/storage/file_storage.py
============================
Safe local filesystem storage for uploaded files and metadata in AXEN.

Security Guarantees:
1. Files are stored with sanitized IDs to prevent path traversal or overwrites.
2. Raw files reside strictly within `storage/uploads/`.
3. File contents and metadata are segregated.
4. Duplicate detection via SHA-256 prevents redundant disk writes.
5. Deletion removes raw data and metadata completely.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from axen.storage.storage_models import FileCategory, FileStatus, UploadConfig, UploadedFile, log_file_event

logger = logging.getLogger("axen.storage")


class StorageSecurityError(Exception):
    """Raised when an upload or storage operation violates security policy."""
    pass


class LocalFileStorage:
    """Manages physical file persistence and metadata storage."""

    def __init__(
        self,
        base_dir: Optional[Path | str] = None,
        config: Optional[UploadConfig] = None,
    ) -> None:
        self.base_dir = Path(base_dir).resolve() if base_dir else Path.cwd() / "data" / "storage"
        self.config = config or UploadConfig()

        self.uploads_dir = self.base_dir / "uploads"
        self.metadata_dir = self.base_dir / "metadata"
        self.processed_dir = self.base_dir / "processed"

        # Ensure directories exist
        for d in (self.uploads_dir, self.metadata_dir, self.processed_dir):
            d.mkdir(parents=True, exist_ok=True)

    def validate_file_info(
        self,
        filename: str,
        size_bytes: int,
        content_stream: Optional[bytes] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate file name, extension, size, and emptiness.
        Returns: (is_valid, error_message)
        """
        if not filename or not filename.strip():
            return False, "File name cannot be empty."

        clean_name = Path(filename).name
        ext = Path(clean_name).suffix.lower()

        if not ext or ext not in self.config.allowed_extensions:
            return False, "AXEN doesn't currently support this file type."


        if size_bytes <= 0:
            return False, "File is empty (0 bytes)."

        if size_bytes > self.config.max_file_size_bytes:
            max_mb = self.config.max_file_size_bytes // (1024 * 1024)
            return False, f"File size ({size_bytes / (1024*1024):.1f} MB) exceeds maximum allowed limit ({max_mb} MB)."

        return True, None

    def store_file(
        self,
        filename: str | bytes,
        file_bytes: bytes | str,
        mime_type: Optional[str] = None,
    ) -> UploadedFile:
        """
        Save file bytes to disk safely, calculate SHA-256, and persist metadata.
        """
        if isinstance(filename, (bytes, bytearray)) and isinstance(file_bytes, (bytes, bytearray)):
            try:
                filename = filename.decode("utf-8")
            except Exception:
                filename = "uploaded_file.bin"
        elif isinstance(filename, (bytes, bytearray)) and isinstance(file_bytes, str):
            # Arguments swapped: (file_bytes, filename)
            file_bytes, filename = filename, file_bytes.encode("utf-8")
        elif isinstance(file_bytes, str):
            file_bytes = file_bytes.encode("utf-8")

        import time
        t0 = time.perf_counter()
        log_file_event("upload_started", filename=str(filename), status="started")

        filename = str(filename)
        size = len(file_bytes)
        valid, err = self.validate_file_info(filename, size)
        if not valid:
            log_file_event("upload_failed", filename=str(filename), status="failed", extra={"error": err})
            raise StorageSecurityError(err or "Invalid file")

        log_file_event("upload_validated", filename=str(filename), status="validated")


        # Sanitize filename
        safe_original_name = self._sanitize_filename(filename)
        ext = Path(safe_original_name).suffix.lower()

        # Compute SHA-256
        sha256 = hashlib.sha256(file_bytes).hexdigest()

        # Check for existing duplicate by hash
        existing = self.find_by_sha256(sha256)
        if existing:
            return existing

        # Generate unique internal file ID
        file_id = f"upl_{uuid.uuid4().hex[:12]}"
        stored_filename = f"{file_id}{ext}"

        stored_path = self.uploads_dir / stored_filename

        # Write safely
        stored_path.write_bytes(file_bytes)

        category = self._categorize_extension(ext)
        inferred_mime = mime_type or self._infer_mime(ext)

        uploaded = UploadedFile(
            file_id=file_id,
            original_filename=safe_original_name,
            stored_filename=stored_filename,
            file_path=str(stored_path),
            size_bytes=size,
            sha256=sha256,
            extension=ext,
            mime_type=inferred_mime,
            category=category,
            status=FileStatus.READY,
        )

        self.save_metadata(uploaded)
        return uploaded

    def save_metadata(self, file_obj: UploadedFile) -> None:
        """Persist metadata JSON to metadata directory."""
        meta_file = self.metadata_dir / f"{file_obj.file_id}.json"
        meta_file.write_text(json.dumps(file_obj.to_dict(), indent=2), encoding="utf-8")

    def get_file(self, file_id: str) -> Optional[UploadedFile]:
        """Load file metadata by file ID."""
        meta_file = self.metadata_dir / f"{file_id}.json"
        if not meta_file.exists():
            return None
        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            return self._dict_to_file(data)
        except Exception as exc:
            logger.error("Error reading file metadata for %s: %s", file_id, exc)
            return None

    def find_by_sha256(self, sha256: str) -> Optional[UploadedFile]:
        """Search metadata index for a file with the identical hash."""
        for meta_file in self.metadata_dir.glob("*.json"):
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
                if data.get("sha256") == sha256:
                    return self._dict_to_file(data)
            except Exception:
                continue
        return None

    def list_files(self) -> List[UploadedFile]:
        """Return list of all uploaded files sorted by newest first."""
        results: List[UploadedFile] = []
        for meta_file in self.metadata_dir.glob("*.json"):
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
                results.append(self._dict_to_file(data))
            except Exception:
                continue
        return sorted(results, key=lambda f: f.created_at, reverse=True)

    def delete_file(self, file_id: str) -> bool:
        """Remove file from uploads, processed directory, and metadata."""
        file_obj = self.get_file(file_id)
        if not file_obj:
            return False

        try:
            # Remove physical file
            path = Path(file_obj.file_path)
            if path.exists():
                path.unlink()

            # Remove processed text if exists
            proc_file = self.processed_dir / f"{file_id}.txt"
            if proc_file.exists():
                proc_file.unlink()

            # Remove metadata
            meta_file = self.metadata_dir / f"{file_id}.json"
            if meta_file.exists():
                meta_file.unlink()

            log_file_event("file_deleted", file_id=file_id, filename=file_obj.original_filename, status="deleted")
            return True
        except Exception as exc:
            logger.error("Failed to delete file %s: %s", file_id, exc)
            log_file_event("file_deleted", file_id=file_id, filename=file_obj.original_filename, status="failed", extra={"error": str(exc)})
            return False

    def clear(self) -> None:
        """Wipe all stored files and metadata."""
        for d in (self.uploads_dir, self.metadata_dir, self.processed_dir):
            if d.exists():
                shutil.rmtree(d)
                d.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Prevent path traversal and strip illegal characters."""
        base = Path(name).name
        # Remove any path traversal remnants or control chars
        clean = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", base)
        return clean or "unnamed_file"

    @staticmethod
    def _categorize_extension(ext: str) -> FileCategory:
        """Classify extension into general file category."""
        e = ext.lower()
        if e in (".pdf", ".txt", ".md", ".markdown", ".docx"):
            return FileCategory.DOCUMENT
        if e in (".csv", ".json"):
            return FileCategory.DATA
        if e in (".py", ".js", ".ts", ".html", ".css", ".yaml", ".yml", ".xml", ".sql", ".java", ".c", ".cpp", ".h", ".hpp", ".sh", ".ps1"):
            return FileCategory.CODE
        if e in (".png", ".jpg", ".jpeg", ".webp"):
            return FileCategory.IMAGE
        return FileCategory.UNSUPPORTED

    def get_file_path(self, file_id: str) -> Optional[Path]:
        """Return Path to stored file if it exists."""
        file_obj = self.get_file(file_id)
        if not file_obj:
            return None
        p = Path(file_obj.file_path)
        return p if p.exists() else None

    @staticmethod
    def _infer_mime(ext: str) -> str:
        """Map common extensions to standard MIME types."""
        mimes = {
            ".pdf": "application/pdf",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".csv": "text/csv",
            ".json": "application/json",
            ".py": "text/x-python",
            ".js": "application/javascript",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }
        return mimes.get(ext.lower(), "application/octet-stream")

    def _dict_to_file(self, data: Dict[str, Any]) -> UploadedFile:
        extracted = data.get("extracted_text")
        if extracted is None:
            extracted = data.get("text", "")
        return UploadedFile(
            file_id=data["file_id"],
            original_filename=data["filename"],
            stored_filename=data["stored_filename"],
            file_path=str(self.uploads_dir / data["stored_filename"]),
            size_bytes=data["size_bytes"],
            sha256=data["sha256"],
            extension=data["extension"],
            mime_type=data.get("mime_type", "application/octet-stream"),
            category=FileCategory(data.get("category", "document")),
            status=FileStatus(data.get("status", "ready")),
            created_at=data.get("created_at", 0.0),
            extracted_text=extracted,
            is_truncated=data.get("is_truncated", False),
            metadata=data.get("metadata", {}),
            error=data.get("error"),
        )
