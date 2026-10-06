"""
axen/tools/coding_tools.py
===========================
Tool definitions exposing the Coding Agent capabilities to the AXEN ToolRegistry:
- coding.inspect: inspect workspace layout, manifests, and map
- coding.search: search code, symbols, or filenames in workspace
- coding.patch: safely propose or apply unified diff patch
- coding.test: run tests and parse failure diagnostics
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from axen.coding import (
    CodeSearch,
    CommandRunner,
    PatchManager,
    PermissionLevel,
    PermissionPolicy,
    ProjectInspector,
    TestRunner,
    Workspace,
)
from .tool_registry import ToolDefinition


def _get_workspace(args: Dict[str, Any]) -> Workspace:
    ws_path = args.get("workspace_path", ".")
    perm_str = args.get("permission_level", "EXECUTE")
    try:
        level = PermissionLevel(perm_str)
    except Exception:
        level = PermissionLevel.EXECUTE
    return Workspace(ws_path, policy=PermissionPolicy(level))


def inspect_workspace_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Inspect workspace layout, manifests, and map."""
    try:
        ws = _get_workspace(args)
        inspector = ProjectInspector(ws)
        summary = inspector.get_summary()
        return {"success": True, "summary": summary}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def search_code_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Search code, symbols, or files within the workspace."""
    try:
        ws = _get_workspace(args)
        searcher = CodeSearch(ws)
        query = args.get("query", "").strip()
        search_type = args.get("type", "text")  # "text", "file", "symbol"

        if search_type == "file":
            matches = searcher.search_files(query)
            return {"success": True, "files": matches}
        elif search_type == "symbol":
            matches = searcher.search_symbols(query)
            return {
                "success": True,
                "matches": [
                    {
                        "file": m.file_path,
                        "line": m.line_number,
                        "symbol": m.symbol_name,
                        "type": m.match_type,
                    }
                    for m in matches
                ],
            }
        else:
            matches = searcher.search_text(query)
            return {
                "success": True,
                "matches": [
                    {
                        "file": m.file_path,
                        "line": m.line_number,
                        "content": m.line_content,
                    }
                    for m in matches
                ],
            }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def patch_file_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Propose or apply a patch to a workspace file."""
    try:
        ws = _get_workspace(args)
        patcher = PatchManager(ws)
        file_path = args.get("file_path", "")
        target = args.get("target_snippet", "")
        replacement = args.get("replacement_snippet", "")
        reason = args.get("reason", "Code update")
        apply_mode = args.get("apply", False)

        if apply_mode:
            ok, diff, err = patcher.apply_patch(file_path, target, replacement, reason)
            return {"success": ok, "diff": diff, "error": err}
        else:
            ok, diff, err = patcher.propose_patch(file_path, target, replacement, reason)
            return {"success": ok, "diff": diff, "error": err}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def run_tests_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute tests in workspace and return structured results."""
    try:
        ws = _get_workspace(args)
        cmd_runner = CommandRunner(ws)
        runner = TestRunner(ws, cmd_runner)
        target = args.get("target_path")
        res = runner.run_tests(target_path=target)

        return {
            "success": res.all_passed,
            "passed": res.passed,
            "failed": res.failed,
            "skipped": res.skipped,
            "duration": res.duration_seconds,
            "failures": [
                {
                    "test": f.test_name,
                    "error": f.error_type,
                    "file": f.file_path,
                    "line": f.line_number,
                    "message": f.message,
                }
                for f in res.failures
            ],
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def build_coding_tools() -> List[ToolDefinition]:
    """Return all coding-related ToolDefinitions."""
    return [
        ToolDefinition(
            name="coding.inspect",
            description="Inspect workspace project structure, manifests, and map.",
            handler=inspect_workspace_tool,
            risk="low",
            capabilities=["coding", "inspection"],
            deterministic=True,
        ),
        ToolDefinition(
            name="coding.search",
            description="Search files, code text, or symbols inside the workspace.",
            handler=search_code_tool,
            risk="low",
            capabilities=["coding", "search"],
            deterministic=True,
        ),
        ToolDefinition(
            name="coding.patch",
            description="Propose or apply a targeted unified diff patch to a workspace file.",
            handler=patch_file_tool,
            risk="high",
            capabilities=["coding", "file_edit"],
            deterministic=True,
            requires_confirmation=True,
        ),
        ToolDefinition(
            name="coding.test",
            description="Run project tests inside the approved workspace and parse results.",
            handler=run_tests_tool,
            risk="medium",
            capabilities=["coding", "testing"],
            deterministic=True,
        ),
    ]
