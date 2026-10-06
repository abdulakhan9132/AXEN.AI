"""
app/tools/filesystem.py
=======================
Sandboxed filesystem tool definitions for app.agent (Phase 5).
Prevents directory traversal, system root access, and unvalidated writes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

from app.agent.tools import ToolDefinition
from axen.tools.base import ToolPermission

_RESTRICTED_PATH_RE = re.compile(
    r"(?:^|[/\\])\.\.(?:[/\\]|$)|"
    r"^[a-zA-Z]:[/\\](?!(?:[a-zA-Z0-9_\-\.\s]+[/\\])*(?:storage|projects|workspace)[/\\])|"
    r"^/(?:etc|sys|proc|var|usr|bin|sbin|dev)|"
    r"\\Windows(?:[/\\]|$)|\\System32(?:[/\\]|$)",
    re.IGNORECASE,
)


def _safe_resolve_path(path_str: str) -> Path:
    p = str(path_str).strip().strip("'\"")
    if not p:
        raise ValueError("Missing path argument.")
    if "\0" in p:
        raise ValueError("Null byte detected in path.")
    if _RESTRICTED_PATH_RE.search(p):
        raise PermissionError(f"Access denied: restricted system path or directory traversal in '{path_str}'")
    return Path(p)


def list_directory(args: dict) -> dict:
    path = args.get("path", ".")
    try:
        root = _safe_resolve_path(path)
        if not root.exists():
            return {"success": False, "error": f"Path not found: {path}"}
        entries = [str(p) for p in sorted(root.iterdir())]
        return {"success": True, "files": entries}
    except (ValueError, PermissionError) as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def read_file(args: dict) -> dict:
    path = args.get("path")
    if not path:
        return {"success": False, "error": "Missing path"}
    try:
        file_path = _safe_resolve_path(path)
        if not file_path.exists():
            return {"success": False, "error": f"File not found: {path}"}
        if file_path.is_dir():
            entries = [str(item.name) for item in sorted(file_path.iterdir())]
            return {"success": True, "content": "\n".join(entries), "files": entries, "is_directory": True}
        return {"success": True, "content": file_path.read_text(encoding="utf-8", errors="replace")}
    except (ValueError, PermissionError) as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def write_file(args: dict) -> dict:
    path = args.get("path")
    content = args.get("content", "")
    if not path:
        return {"success": False, "error": "Missing path"}
    try:
        file_path = _safe_resolve_path(path)
        # Block writing to absolute roots
        if str(file_path).startswith("/") or (len(str(file_path)) >= 2 and str(file_path)[1] == ":"):
            # Check if it is within current directory or storage
            cwd = Path.cwd().resolve()
            resolved = file_path.resolve()
            if not resolved.is_relative_to(cwd) and "storage" not in str(resolved).lower():
                return {"success": False, "error": "Refusing to write outside the workspace root."}
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return {"success": True, "path": str(file_path), "bytes": len(content.encode("utf-8"))}
    except (ValueError, PermissionError) as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def search_files(args: dict) -> dict:
    root = args.get("root", ".")
    pattern = args.get("pattern", "**/*")
    try:
        base = _safe_resolve_path(root)
        if not base.exists():
            return {"success": False, "error": f"Root does not exist: {root}"}
        matches: List[str] = []
        for item in base.rglob('*'):
            if pattern in {"**/*.pdf", "*.pdf"}:
                if item.is_file() and item.suffix.lower() == ".pdf":
                    matches.append(str(item.resolve()))
            elif item.is_file() and pattern in {"**/*", "*"}:
                matches.append(str(item.resolve()))
        return {"success": True, "files": matches}
    except (ValueError, PermissionError) as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def build_filesystem_tools() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="filesystem.list_dir",
            description="List the contents of a directory within the workspace.",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            output_schema={"type": "object", "properties": {"files": {"type": "array", "items": {"type": "string"}}}},
            permission=ToolPermission.LOCAL_FILE,
            risk="low",
            risk_level="low",
            handler=list_directory,
        ),
        ToolDefinition(
            name="filesystem.read",
            description="Read a file from disk within the authorized workspace.",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            output_schema={"type": "object", "properties": {"content": {"type": "string"}}},
            permission=ToolPermission.LOCAL_FILE,
            risk="low",
            risk_level="low",
            handler=read_file,
        ),
        ToolDefinition(
            name="filesystem.write",
            description="Create or overwrite a file within the authorized workspace.",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
            output_schema={"type": "object", "properties": {"path": {"type": "string"}, "bytes": {"type": "integer"}}},
            permission=ToolPermission.LOCAL_FILE,
            risk="medium",
            risk_level="medium",
            handler=write_file,
        ),
        ToolDefinition(
            name="filesystem.search",
            description="Search the workspace for matching files.",
            input_schema={"type": "object", "properties": {"root": {"type": "string"}, "pattern": {"type": "string"}}, "required": ["root"]},
            output_schema={"type": "object", "properties": {"files": {"type": "array", "items": {"type": "string"}}}},
            permission=ToolPermission.LOCAL_FILE,
            risk="low",
            risk_level="low",
            handler=search_files,
        ),
    ]
