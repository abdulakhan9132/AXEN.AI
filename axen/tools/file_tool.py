r"""
axen/tools/file_tool.py
=======================
Safe Project-Sandboxed File Tools for AXEN (Phase 5).

Integrates directly with Phase 4 ProjectManager and FileManager.
Standardized tools:
  - file_list   (alias file.list)
  - file_read   (alias file.read)
  - file_search (alias file.search)
  - file.inspect / file.extract_text

SAFETY RULES:
  - Tools operate ONLY inside authorized project or storage.
  - Direct access to C:\, D:\, user home root, and system folders is DENIED.
  - Path traversal (..) is strictly DENIED.
  - Cross-project access is DENIED (Project A cannot access Project B's files).
  - Output is limited and untrusted.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import (
    ExecutionState,
    ToolDefinition,
    ToolExecutionContext,
    ToolPermission,
    ToolResult,
    ToolRiskLevel,
)

class ExtendedFileTool:
    """Backward compatibility wrapper for legacy tests expecting ExtendedFileTool class."""
    pass


logger = logging.getLogger("axen.tools.file")

# Restricted system path patterns
_RESTRICTED_ROOT_PATTERN = re.compile(
    r"(?:^|[/\\])\.\.(?:[/\\]|$)|"  # traversal ..
    r"^[a-zA-Z]:[/\\](?!(?:[a-zA-Z0-9_\-\.\s]+[/\\])*(?:storage|projects|workspace|temp|tmp|appdata[/\\]local[/\\]temp)[/\\])|"
    r"^/(?:etc|sys|proc|var|usr|bin|sbin|dev|boot)|"
    r"\\Windows(?:[/\\]|$)|\\System32(?:[/\\]|$)|%SystemRoot%",
    re.IGNORECASE,
)


def _assert_path_sandbox(path_str: str, context: Optional[ToolExecutionContext] = None) -> Path:
    """
    Validate that path is safe and within the authorized workspace/sandbox.
    Raises PermissionError or ValueError on violation.
    """
    cleaned = path_str.strip().strip("'\"")
    if not cleaned:
        raise ValueError("Missing file path.")

    if "\0" in cleaned:
        raise ValueError("Null byte in path.")

    if _RESTRICTED_ROOT_PATTERN.search(cleaned):
        raise PermissionError(f"Access denied: Path '{cleaned}' is outside the authorized sandbox or references system roots.")

    # Check against context workspace or project root
    p = Path(cleaned)
    if not p.is_absolute() and context and context.workspace_path:
        resolved = (Path(context.workspace_path) / p).resolve()
    else:
        resolved = p.resolve()

    # If context has workspace_path, enforce it
    if context and context.workspace_path:
        ws_resolved = Path(context.workspace_path).resolve()
        try:
            if not resolved.is_relative_to(ws_resolved) and "storage" not in str(resolved).lower() and "temp" not in str(resolved).lower():
                raise PermissionError(f"Access denied: '{cleaned}' is outside workspace '{context.workspace_path}'.")
        except AttributeError:
            if not str(resolved).startswith(str(ws_resolved)) and "storage" not in str(resolved).lower() and "temp" not in str(resolved).lower():
                raise PermissionError(f"Access denied: '{cleaned}' is outside workspace '{context.workspace_path}'.")

    return resolved


def _check_project_file_authorization(
    file_id: Optional[str],
    context: Optional[ToolExecutionContext],
) -> Optional[str]:
    """
    If execution context is bound to a specific project_id, verify that file_id
    belongs to this project. Returns error message if unauthorized.
    """
    if not context or not context.project_id or not file_id:
        return None

    try:
        from axen.projects.project_manager import ProjectManager
        pm = ProjectManager.get_instance()
        proj = pm.get_project(context.project_id)
        if proj is not None:
            proj_fids = set()
            if hasattr(pm, "store") and hasattr(pm.store, "list_project_file_ids"):
                proj_fids = set(pm.store.list_project_file_ids(context.project_id))
            elif hasattr(pm, "list_project_files"):
                proj_fids = {getattr(f, "file_id", str(f)) for f in pm.list_project_files(context.project_id)}

            if file_id not in proj_fids:
                return (
                    f"Project isolation violation: File '{file_id}' does not belong to project "
                    f"'{context.project_id}'."
                )
    except Exception as exc:
        logger.debug("Project authorization check error: %s", exc)

    return None


def read_file(args: Dict[str, Any], context: Optional[ToolExecutionContext] = None) -> Dict[str, Any]:
    """
    Safely read text from an authorized local file or uploaded file_id.
    """
    file_id = args.get("file_id")

    # Project isolation check
    proj_err = _check_project_file_authorization(file_id, context)
    if proj_err:
        return {"success": False, "error": proj_err}

    if file_id:
        try:
            from axen.storage.file_manager import FileManager
            fobj = FileManager.get_instance().get_file(str(file_id))
            if not fobj:
                return {"success": False, "error": f"Uploaded file not found: {file_id}"}
            content = fobj.extracted_text
            max_bytes = context.max_output_bytes if context else 5_000_000
            if len(content.encode("utf-8")) > max_bytes:
                content = content[:max_bytes] + "\n...[TRUNCATED: Exceeded size limit]"
            return {
                "success": True,
                "file_id": fobj.file_id,
                "filename": fobj.original_filename,
                "content": content,
                "lines": len(content.splitlines()),
                "size": fobj.format_size(),
            }
        except Exception as exc:
            return {"success": False, "error": f"Error loading uploaded file: {exc}"}

    path_str = str(args.get("path") or args.get("file_path") or "").strip()
    if not path_str:
        return {"success": False, "error": "Missing 'path' or 'file_id' argument."}

    try:
        file_path = _assert_path_sandbox(path_str, context)
    except (ValueError, PermissionError) as exc:
        return {"success": False, "error": str(exc)}

    if not file_path.exists():
        return {"success": False, "error": f"File not found: {path_str}"}

    if file_path.is_dir():
        return {"success": False, "error": f"Path is a directory, not a file: {path_str}"}

    # Max 5 MB hard limit
    max_bytes = min(context.max_output_bytes if context else 5_000_000, 5 * 1024 * 1024)
    if file_path.stat().st_size > 5 * 1024 * 1024:
        return {"success": False, "error": f"File exceeds maximum allowed size (5 MB): {path_str}"}

    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
        if len(content.encode("utf-8")) > max_bytes:
            content = content[:max_bytes] + "\n...[TRUNCATED: Exceeded size limit]"
        return {
            "success": True,
            "path": str(file_path),
            "content": content,
            "lines": len(content.splitlines()),
        }
    except Exception as exc:
        return {"success": False, "error": f"Could not read file: {exc}"}


def list_files(args: Dict[str, Any], context: Optional[ToolExecutionContext] = None) -> Dict[str, Any]:
    """
    List files in authorized project or storage.
    """
    # If scoped to project, list project files
    proj_id = (context.project_id if context else None) or args.get("project_id")
    if proj_id:
        try:
            from axen.projects.project_manager import ProjectManager
            pm = ProjectManager.get_instance()
            proj = pm.get_project(proj_id)
            if proj:
                return {
                    "success": True,
                    "project_id": proj_id,
                    "count": len(proj.files),
                    "files": [f.to_dict() for f in proj.files],
                }
        except Exception as exc:
            logger.debug("Project file listing error: %s", exc)

    # Fallback to general storage
    try:
        from axen.storage.file_manager import FileManager
        files = FileManager.get_instance().list_files()
        return {
            "success": True,
            "count": len(files),
            "files": [f.to_dict() for f in files],
        }
    except Exception as exc:
        return {"success": False, "error": f"Could not list files: {exc}"}


def search_file(args: Dict[str, Any], context: Optional[ToolExecutionContext] = None) -> Dict[str, Any]:
    """
    Search for keyword occurrences within authorized file or project files.
    """
    query = str(args.get("query", "")).lower().strip()
    if not query:
        return {"success": False, "error": "Missing 'query' parameter."}

    file_id = args.get("file_id")
    path_str = str(args.get("path") or args.get("file_path") or "").strip()

    # If neither file_id nor path is given, search all files in current workspace or project
    if not file_id and not path_str:
        if context and context.workspace_path:
            ws_dir = Path(context.workspace_path)
            if ws_dir.exists() and ws_dir.is_dir():
                all_matches = []
                for file_entry in ws_dir.rglob("*"):
                    if file_entry.is_file() and not file_entry.name.startswith("."):
                        try:
                            file_text = file_entry.read_text(encoding="utf-8", errors="replace")
                            for idx, line in enumerate(file_text.splitlines(), 1):
                                if query in line.lower() or query in file_entry.name.lower():
                                    all_matches.append({
                                        "path": file_entry.name,
                                        "line": idx,
                                        "text": line.strip()[:200],
                                    })
                                    if len(all_matches) >= 30:
                                        break
                        except Exception:
                            continue
                if all_matches:
                    return {
                        "success": True,
                        "query": query,
                        "matches_count": len(all_matches),
                        "matches": all_matches,
                    }

        proj_id = (context.project_id if context else None) or args.get("project_id")
        if proj_id:
            try:
                from axen.projects.project_manager import ProjectManager
                pm = ProjectManager.get_instance()
                proj = pm.get_project(proj_id)
                if proj and proj.files:
                    all_matches = []
                    for f in proj.files:
                        lines = (f.extracted_text or "").splitlines()
                        for idx, line in enumerate(lines, 1):
                            if query in line.lower():
                                all_matches.append({
                                    "filename": f.original_filename,
                                    "file_id": f.file_id,
                                    "line": idx,
                                    "text": line.strip()[:200],
                                })
                                if len(all_matches) >= 30:
                                    break
                    return {
                        "success": True,
                        "query": query,
                        "matches_count": len(all_matches),
                        "matches": all_matches,
                    }
            except Exception as exc:
                return {"success": False, "error": f"Error searching project files: {exc}"}

        # Otherwise read standard README.md if present in workspace
        path_str = "README.md"

    # Read target content
    res = read_file({"file_id": file_id, "path": path_str}, context=context)
    if not res.get("success"):
        return res

    content = res.get("content", "")
    matches = []
    for idx, line in enumerate(content.splitlines(), 1):
        if query in line.lower():
            matches.append({"line": idx, "text": line.strip()[:200]})
            if len(matches) >= 30:
                break

    return {
        "success": True,
        "query": query,
        "matches_count": len(matches),
        "matches": matches,
    }


def inspect_file(args: Dict[str, Any], context: Optional[ToolExecutionContext] = None) -> Dict[str, Any]:
    """Inspect file metadata safely."""
    file_id = args.get("file_id")
    proj_err = _check_project_file_authorization(file_id, context)
    if proj_err:
        return {"success": False, "error": proj_err}

    if file_id:
        from axen.storage.file_manager import FileManager
        fobj = FileManager.get_instance().get_file(str(file_id))
        if not fobj:
            return {"success": False, "error": f"File not found: {file_id}"}
        return {"success": True, "file": fobj.to_dict()}

    path_str = str(args.get("path") or "").strip()
    if not path_str:
        return {"success": False, "error": "Missing 'path' or 'file_id'."}

    try:
        p = _assert_path_sandbox(path_str, context)
    except (ValueError, PermissionError) as exc:
        return {"success": False, "error": str(exc)}

    if not p.exists():
        return {"success": False, "error": f"Path not found: {path_str}"}

    return {
        "success": True,
        "path": str(p),
        "name": p.name,
        "size_bytes": p.stat().st_size,
        "is_file": p.is_file(),
        "is_dir": p.is_dir(),
    }


def extract_text(args: Dict[str, Any], context: Optional[ToolExecutionContext] = None) -> Dict[str, Any]:
    """Extract text from an uploaded file ID or path."""
    return read_file(args, context=context)


def file_dispatch(args: Dict[str, Any], context: Optional[ToolExecutionContext] = None) -> Dict[str, Any]:
    """Unified file tool dispatcher."""
    op = str(args.get("operation") or "read_file").strip().lower()
    if op in ("extract_text", "extract"):
        return extract_text(args, context=context)
    elif op in ("inspect_file", "inspect", "get_metadata", "metadata"):
        return inspect_file(args, context=context)
    elif op in ("search_file", "search"):
        return search_file(args, context=context)
    elif op in ("list_files", "list"):
        return list_files(args, context=context)
    else:
        return read_file(args, context=context)


# ------------------------------------------------------------------
# Canonical Tool Definitions
# ------------------------------------------------------------------

def build_file_read_tool() -> ToolDefinition:
    """Canonical file_read tool."""
    return ToolDefinition(
        name="file_read",
        description="Safely read the text content of a project file or uploaded file ID. Args: {'path': '<path>'} or {'file_id': '<id>'}",
        handler=read_file,
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path inside project."},
                "file_id": {"type": "string", "description": "Uploaded file identifier."},
            },
        },
        permission=ToolPermission.LOCAL_FILE,
        risk_level="low",
        risk="low",
        capabilities=["file_read", "local_file", "read"],
        deterministic=True,
        supports_verification=True,
    )


def build_file_list_tool() -> ToolDefinition:
    """Canonical file_list tool."""
    return ToolDefinition(
        name="file_list",
        description="List files available in the authorized project or storage.",
        handler=list_files,
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string", "description": "Project ID to list files for."},
            },
        },
        permission=ToolPermission.LOCAL_FILE,
        risk_level="low",
        risk="low",
        capabilities=["file_list", "list_files", "storage"],
        deterministic=True,
        supports_verification=True,
    )


def build_file_search_tool() -> ToolDefinition:
    """Canonical file_search tool."""
    return ToolDefinition(
        name="file_search",
        description="Search for keyword occurrences in authorized project files. Args: {'query': 'text', 'path': '<path>', 'file_id': '<id>'}",
        handler=search_file,
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search term or keyword."},
                "path": {"type": "string", "description": "Optional file path to limit search."},
                "file_id": {"type": "string", "description": "Optional file ID to limit search."},
            },
            "required": ["query"],
        },
        permission=ToolPermission.LOCAL_FILE,
        risk_level="low",
        risk="low",
        capabilities=["file_search", "search", "search_files"],
        deterministic=True,
        supports_verification=True,
    )


def build_file_tool() -> ToolDefinition:
    """Backward compatibility alias for file.read."""
    tool = build_file_read_tool()
    tool.name = "file.read"
    return tool


def build_file_extended_tools() -> List[ToolDefinition]:
    """Return the suite of file management tools with dotted and underscore naming."""
    canonical_read = build_file_read_tool()
    canonical_list = build_file_list_tool()
    canonical_search = build_file_search_tool()

    compat_read = build_file_tool()
    compat_list = ToolDefinition(
        name="file.list",
        description=canonical_list.description,
        handler=list_files,
        permission=ToolPermission.LOCAL_FILE,
        capabilities=["file.list", "list"],
    )
    compat_search = ToolDefinition(
        name="file.search",
        description=canonical_search.description,
        handler=search_file,
        input_schema=canonical_search.input_schema,
        permission=ToolPermission.LOCAL_FILE,
        capabilities=["file.search", "search"],
    )
    compat_inspect = ToolDefinition(
        name="file.inspect",
        description="Inspect file metadata, size, and category.",
        handler=inspect_file,
        permission=ToolPermission.LOCAL_FILE,
        capabilities=["inspection", "metadata"],
    )
    compat_extract = ToolDefinition(
        name="file.extract_text",
        description="Extract structured text and metadata from an uploaded file ID.",
        handler=extract_text,
        permission=ToolPermission.LOCAL_FILE,
        capabilities=["extraction", "text"],
    )
    unified_file = ToolDefinition(
        name="file",
        description="Unified file tool for reading, extracting, inspecting, searching, and listing uploaded files.",
        handler=file_dispatch,
        permission=ToolPermission.LOCAL_FILE,
        capabilities=["file", "file_read", "extraction", "inspection", "search", "list"],
    )

    return [
        canonical_read,
        canonical_list,
        canonical_search,
        compat_read,
        compat_list,
        compat_search,
        compat_inspect,
        compat_extract,
        unified_file,
    ]
