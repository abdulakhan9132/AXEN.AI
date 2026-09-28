from __future__ import annotations

from pathlib import Path
from typing import List

from app.agent.tools import ToolDefinition


def list_directory(args: dict) -> dict:
    path = args.get("path", ".")
    root = Path(path)
    if not root.exists():
        return {"success": False, "error": f"Path not found: {path}"}
    entries = [str(p) for p in sorted(root.iterdir())]
    return {"success": True, "files": entries}


def read_file(args: dict) -> dict:
    path = args.get("path")
    if not path:
        return {"success": False, "error": "Missing path"}
    file_path = Path(path)
    if not file_path.exists():
        return {"success": False, "error": f"File not found: {path}"}
    if file_path.is_dir():
        entries = [str(item.name) for item in sorted(file_path.iterdir())]
        return {"success": True, "content": "\n".join(entries), "files": entries, "is_directory": True}
    return {"success": True, "content": file_path.read_text(encoding="utf-8")}


def write_file(args: dict) -> dict:
    path = args.get("path")
    content = args.get("content", "")
    if not path:
        return {"success": False, "error": "Missing path"}
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return {"success": True, "path": str(file_path), "bytes": len(content.encode("utf-8"))}


def search_files(args: dict) -> dict:
    root = args.get("root", ".")
    pattern = args.get("pattern", "**/*")
    matches: List[str] = []
    base = Path(root)
    if not base.exists():
        return {"success": False, "error": f"Root does not exist: {root}"}
    for item in base.rglob('*'):
        if pattern in {"**/*.pdf", "*.pdf"}:
            if item.is_file() and item.suffix.lower() == ".pdf":
                matches.append(str(item.resolve()))
        elif item.is_file() and pattern in {"**/*", "*"}:
            matches.append(str(item.resolve()))
    return {"success": True, "files": matches}


def build_filesystem_tools() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="filesystem.list_dir",
            description="List the contents of a directory.",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            output_schema={"type": "object", "properties": {"files": {"type": "array", "items": {"type": "string"}}}},
            risk="low",
            handler=list_directory,
        ),
        ToolDefinition(
            name="filesystem.read",
            description="Read a file from disk.",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            output_schema={"type": "object", "properties": {"content": {"type": "string"}}},
            risk="low",
            handler=read_file,
        ),
        ToolDefinition(
            name="filesystem.write",
            description="Create or overwrite a file.",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
            output_schema={"type": "object", "properties": {"path": {"type": "string"}, "bytes": {"type": "integer"}}},
            risk="medium",
            handler=write_file,
        ),
        ToolDefinition(
            name="filesystem.search",
            description="Search the workspace for matching files.",
            input_schema={"type": "object", "properties": {"root": {"type": "string"}, "pattern": {"type": "string"}}, "required": ["root"]},
            output_schema={"type": "object", "properties": {"files": {"type": "array", "items": {"type": "string"}}}},
            risk="low",
            handler=search_files,
        ),
    ]
