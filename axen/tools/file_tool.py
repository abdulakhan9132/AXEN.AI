"""
axen/tools/file_tool.py
=======================
Safe local file reader tool for AXEN.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .tool_registry import ToolDefinition


def read_file(args: dict[str, Any]) -> dict[str, Any]:
    """
    Read text from a local file safely.

    Args:
        args["path"]: File path to read.
    """
    path_str = str(args.get("path", "")).strip().strip("'\"")
    if not path_str:
        return {"success": False, "error": "Missing 'path' argument."}

    file_path = Path(path_str).resolve()
    if not file_path.exists():
        return {"success": False, "error": f"File not found: {path_str}"}

    if file_path.is_dir():
        return {"success": False, "error": f"Path is a directory, not a file: {path_str}"}

    # Limit file size read for safety (max 100 KB)
    max_bytes = 100 * 1024
    if file_path.stat().st_size > max_bytes:
        return {
            "success": False,
            "error": f"File exceeds maximum allowed size (100 KB): {path_str}",
        }

    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
        return {
            "success": True,
            "path": str(file_path),
            "content": content,
            "lines": len(content.splitlines()),
        }
    except Exception as exc:
        return {"success": False, "error": f"Could not read file: {exc}"}


def build_file_tool() -> ToolDefinition:
    """Return the AXEN file reading ToolDefinition."""
    return ToolDefinition(
        name="file.read",
        description="Safely read the text content of a local file. Args: {'path': '<file_path>'}",
        handler=read_file,
        risk="low",
        capabilities=["file_read", "inspection"],
        deterministic=True,
        requires_confirmation=False,
        supports_verification=True,
    )
