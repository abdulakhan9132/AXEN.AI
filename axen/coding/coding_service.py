"""
axen/coding/coding_service.py
=============================
Unified Coding Service for AEXN AI Codex-Style Coding Agent.
Provides workspace management, file tree, safe editing, AST verification,
diff preview, patch application, rollback, test execution, and Think Engine integration.
"""

from __future__ import annotations

import ast
import logging
import os
from pathlib import Path
import subprocess
import threading
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .change_journal import ChangeJournal
from .code_search import CodeSearch
from .coding_agent import CodingAgent, CodingAgentResult, CodingTaskType
from .command_runner import CommandRunner
from .git_tracker import GitTracker
from .patch_manager import PatchManager
from .permissions import OperationType, PermissionLevel, PermissionPolicy
from .project_inspector import ProjectInspector
from .test_runner import TestResult, TestRunner
from .workspace import Workspace, WorkspaceSecurityError

logger = logging.getLogger("axen.coding.service")

# Extensions mapped to syntax modes for UI highlighting
SYNTAX_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".json": "json",
    ".html": "html",
    ".css": "css",
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".sh": "bash",
    ".bat": "batch",
    ".ps1": "powershell",
    ".txt": "plaintext",
    ".sql": "sql",
}

# Protected files that automated coding agent must never alter without explicit override
PROTECTED_PATTERNS = {
    ".env",
    ".env.local",
    ".env.production",
    "id_rsa",
    "id_ed25519",
    "credentials.json",
    "service_account.json",
    "master.key",
}


class CodingService:
    """Singleton service orchestrating the Codex-style coding assistant."""

    _instance: Optional["CodingService"] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, default_root: Optional[Path | str] = None) -> "CodingService":
        with cls._lock:
            if cls._instance is None:
                root = default_root or Path.cwd()
                cls._instance = cls(root_path=root)
            return cls._instance

    def __init__(self, root_path: Path | str, permission_level: PermissionLevel = PermissionLevel.EXECUTE) -> None:
        self._root_path = Path(root_path).resolve()
        self._permission_level = permission_level
        self._init_workspace()

    def _init_workspace(self) -> None:
        policy = PermissionPolicy(self._permission_level)
        self.workspace = Workspace(self._root_path, policy=policy)
        self.inspector = ProjectInspector(self.workspace)
        self.searcher = CodeSearch(self.workspace)
        self.journal = ChangeJournal()
        self.git_tracker = GitTracker(self.workspace)
        self.patch_mgr = PatchManager(self.workspace, self.journal, self.git_tracker)
        self.cmd_runner = CommandRunner(self.workspace)
        self.test_runner = TestRunner(self.workspace, self.cmd_runner)
        self.agent = CodingAgent(self.workspace, permission_level=self._permission_level)

    def set_workspace(self, new_path: Path | str, permission_level: Optional[PermissionLevel] = None) -> Dict[str, Any]:
        """Switch active workspace directory with strict validation."""
        target = Path(new_path).resolve()
        if not target.exists():
            raise FileNotFoundError(f"Path does not exist: {target}")
        if not target.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {target}")

        self._root_path = target
        if permission_level is not None:
            self._permission_level = permission_level

        self._init_workspace()
        logger.info("Switched coding workspace to: %s", self._root_path)
        return self.get_workspace_info()

    def get_workspace_info(self) -> Dict[str, Any]:
        """Return metadata about current workspace."""
        summary = self.inspector.get_summary()
        git_branch = self.git_tracker.get_current_branch() if self.git_tracker.is_git_repository else None
        return {
            "root_path": str(self.workspace.root_path),
            "workspace_name": self.workspace.root_path.name,
            "permission_level": self.workspace.policy.level.value,
            "is_git": self.git_tracker.is_git_repository,
            "git_branch": git_branch,
            "primary_language": summary.get("primary_language", "Unknown"),
            "test_runner": summary.get("test_runner", "pytest"),
            "total_files": summary.get("total_files", 0),
        }

    # -----------------------------------------------------------------------
    # File Tree & Navigation
    # -----------------------------------------------------------------------

    def get_file_tree(self, max_depth: int = 4) -> Dict[str, Any]:
        """Generate structured recursive file tree for the workspace explorer."""
        ignored_names = {
            ".git",
            "__pycache__",
            ".venv",
            "node_modules",
            ".pytest_cache",
            ".DS_Store",
            ".idea",
            ".vscode",
            "dist",
            "build",
            ".mypy_cache",
            ".tox",
        }

        def _build_node(current_path: Path, depth: int) -> Optional[Dict[str, Any]]:
            if depth > max_depth:
                return None

            try:
                rel_path = current_path.relative_to(self.workspace.root_path).as_posix()
            except ValueError:
                rel_path = current_path.name

            if current_path.is_file():
                ext = current_path.suffix.lower()
                return {
                    "name": current_path.name,
                    "path": rel_path if rel_path != "." else current_path.name,
                    "type": "file",
                    "extension": ext,
                    "syntax": SYNTAX_MAP.get(ext, "plaintext"),
                    "size": current_path.stat().st_size,
                }

            if current_path.is_dir():
                children = []
                try:
                    entries = sorted(
                        current_path.iterdir(),
                        key=lambda e: (not e.is_dir(), e.name.lower()),
                    )
                    for entry in entries:
                        if entry.name in ignored_names or entry.name.startswith(".system_"):
                            continue
                        node = _build_node(entry, depth + 1)
                        if node is not None:
                            children.append(node)
                except PermissionError:
                    pass

                return {
                    "name": current_path.name if rel_path != "." else self.workspace.root_path.name,
                    "path": rel_path if rel_path != "." else "",
                    "type": "directory",
                    "children": children,
                }

            return None

        tree = _build_node(self.workspace.root_path, depth=0)
        return tree or {"name": self.workspace.root_path.name, "type": "directory", "children": []}

    # -----------------------------------------------------------------------
    # File Read & Write with AST Verification
    # -----------------------------------------------------------------------

    def read_file(self, relative_path: str) -> Dict[str, Any]:
        """Safely read a file within the workspace."""
        clean_path = relative_path.replace("\\", "/").lstrip("/")
        resolved = self.workspace.resolve_path(clean_path)
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(f"File not found: {clean_path}")

        ext = resolved.suffix.lower()
        content = self.workspace.read_file(clean_path)
        lines = content.splitlines()

        return {
            "path": clean_path,
            "filename": resolved.name,
            "content": content,
            "lines_count": len(lines),
            "size_bytes": len(content.encode("utf-8")),
            "syntax": SYNTAX_MAP.get(ext, "plaintext"),
            "modified_time": resolved.stat().st_mtime,
            "read_only": resolved.name in PROTECTED_PATTERNS,
        }

    def save_file(
        self,
        relative_path: str,
        content: str,
        validate_syntax: bool = True,
        reason: str = "Manual edit in Coding Workspace",
    ) -> Dict[str, Any]:
        """Save a file with AST syntax validation and automatic backup."""
        clean_path = relative_path.replace("\\", "/").lstrip("/")
        resolved = self.workspace.resolve_path(clean_path)

        # Protect sensitive files
        if resolved.name in PROTECTED_PATTERNS:
            raise PermissionError(f"File '{resolved.name}' is a protected credential file and cannot be modified.")

        # Read original text for diff generation
        orig_content = ""
        if resolved.exists():
            orig_content = self.workspace.read_file(clean_path)

        # Pre-validate Python syntax if enabled
        syntax_valid = True
        syntax_error = ""
        if validate_syntax and clean_path.endswith(".py"):
            try:
                ast.parse(content)
            except SyntaxError as err:
                syntax_valid = False
                syntax_error = f"SyntaxError at line {err.lineno}: {err.msg}"
                return {
                    "success": False,
                    "error": syntax_error,
                    "syntax_valid": False,
                    "path": clean_path,
                }

        # Create backup before write
        if resolved.exists():
            self.workspace.create_backup(clean_path)

        try:
            self.workspace.write_file(clean_path, content)
            diff = self.patch_mgr.generate_diff(clean_path, orig_content, content)
            self.journal.record_change(
                file_path=clean_path,
                operation="modify" if orig_content else "create",
                reason=reason,
                diff=diff,
                status="applied",
            )
            self.workspace.remove_backup(clean_path)
            return {
                "success": True,
                "path": clean_path,
                "diff": diff,
                "syntax_valid": True,
                "lines_count": len(content.splitlines()),
            }
        except Exception as exc:
            if resolved.exists():
                self.workspace.restore_backup(clean_path)
            return {
                "success": False,
                "error": str(exc),
                "syntax_valid": syntax_valid,
                "path": clean_path,
            }

    # -----------------------------------------------------------------------
    # Git & Diff Operations
    # -----------------------------------------------------------------------

    def get_git_status(self) -> Dict[str, Any]:
        """Get git repository branch, modified files, and status."""
        if not self.git_tracker.is_git_repository:
            return {"is_git": False, "branch": None, "modified_files": [], "clean": True}

        branch = self.git_tracker.get_current_branch()
        modified = sorted(list(self.git_tracker.get_modified_files()))
        return {
            "is_git": True,
            "branch": branch,
            "modified_files": modified,
            "clean": len(modified) == 0,
            "modified_count": len(modified),
        }

    def get_diff(self, file_path: Optional[str] = None) -> str:
        """Return unstaged diff for workspace or specific file."""
        return self.git_tracker.get_diff(file_path)

    # -----------------------------------------------------------------------
    # Patch & Rollback
    # -----------------------------------------------------------------------

    def propose_patch(self, file_path: str, target_snippet: str, replacement_snippet: str) -> Dict[str, Any]:
        """Preview unified diff and check syntax without writing to disk."""
        valid, diff, err = self.patch_mgr.propose_patch(file_path, target_snippet, replacement_snippet)
        return {
            "valid": valid,
            "diff": diff,
            "error": err,
            "file_path": file_path,
        }

    def apply_patch(
        self,
        file_path: str,
        target_snippet: str,
        replacement_snippet: str,
        reason: str = "User approved patch",
    ) -> Dict[str, Any]:
        """Apply patch to file with automatic backup and rollback."""
        success, diff, err = self.patch_mgr.apply_patch(
            file_path, target_snippet, replacement_snippet, reason=reason, skip_git_check=True
        )
        return {
            "success": success,
            "diff": diff,
            "error": err,
            "file_path": file_path,
        }

    def rollback_file(self, file_path: str) -> Dict[str, Any]:
        """Rollback file to previous backup, journal original text, or git version."""
        clean_path = file_path.replace("\\", "/").lstrip("/")

        # 1. Try restoring from change journal original content
        orig_text = self.journal.get_latest_original_content(clean_path)
        if orig_text is not None:
            try:
                self.workspace.write_file(clean_path, orig_text)
                self.journal.record_change(
                    file_path=clean_path,
                    operation="revert",
                    reason="User requested undo/rollback",
                    diff="",
                    status="reverted",
                )
                return {"success": True, "method": "journal_restored", "file_path": clean_path}
            except Exception as e:
                logger.debug("Journal restore error: %s", e)

        # 2. Try workspace backup
        if self.workspace.restore_backup(clean_path):
            return {"success": True, "method": "backup_restored", "file_path": clean_path}

        # 3. Try git checkout if git is available
        if self.git_tracker.is_git_repository:
            try:
                res = subprocess.run(
                    ["git", "checkout", "--", clean_path],
                    cwd=str(self.workspace.root_path),
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                )
                if res.returncode == 0:
                    return {"success": True, "method": "git_checkout", "file_path": clean_path}
            except Exception as e:
                logger.debug("Git rollback failed: %s", e)

        return {"success": False, "error": "No backup or git version available to restore", "file_path": clean_path}

    # -----------------------------------------------------------------------
    # Testing & Error Analysis
    # -----------------------------------------------------------------------

    def run_tests(self, target: Optional[str] = None) -> Dict[str, Any]:
        """Execute test suite and parse structured failure info."""
        res: TestResult = self.test_runner.run_tests(target_path=target)
        failures_list = [
            {
                "test_name": f.test_name,
                "error_type": f.error_type,
                "file_path": f.file_path,
                "line_number": f.line_number,
                "message": f.message,
            }
            for f in res.failures
        ]
        return {
            "command": res.command,
            "exit_code": res.exit_code,
            "all_passed": res.all_passed,
            "passed": res.passed,
            "failed": res.failed,
            "skipped": res.skipped,
            "errors": res.errors,
            "duration_seconds": round(res.duration_seconds, 2),
            "failures": failures_list,
            "raw_stdout": res.raw_stdout[-3000:],
            "raw_stderr": res.raw_stderr[-3000:],
        }

    # -----------------------------------------------------------------------
    # Think Engine Integrated Coding Task
    # -----------------------------------------------------------------------

    def execute_task(
        self,
        request: str,
        think_mode: str = "balanced",
        state_callback: Optional[Callable[[str, str, Optional[Dict[str, Any]]], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        """
        Execute a coding task with Think Engine 2.0 modes:
        - FAST: Direct code explanation or review
        - BALANCED: Standard targeted fix or generation with verification
        - DEEP: Full multi-file plan, baseline test, patch, failure debug loop
        """
        mode = think_mode.lower().strip()
        task_type = self.agent.classify_task(request)

        # In FAST mode for explanation/question, bypass multi-step edits
        if mode == "fast" and task_type in (
            CodingTaskType.CODE_QUESTION,
            CodingTaskType.CODE_EXPLANATION,
            CodingTaskType.CODE_REVIEW,
            CodingTaskType.PROJECT_ANALYSIS,
        ):
            if state_callback:
                state_callback("ANALYZING", "Analyzing code in Fast mode...", {"mode": "fast"})
            res = self.agent.run_task(request, task_type=task_type, state_callback=state_callback, cancel_event=cancel_event)
            return {
                "success": res.success,
                "summary": res.summary,
                "task_type": res.task_type,
                "plan_steps": res.plan_steps,
                "files_modified": res.files_modified,
                "diffs": res.diffs,
                "think_mode": "fast",
                "tests_passed": res.tests_passed,
                "tests_failed": res.tests_failed,
            }

        # In BALANCED or DEEP mode: run full coding workflow
        if state_callback:
            state_callback("PLANNING", f"Planning task in {mode.upper()} mode...", {"mode": mode})

        res = self.agent.run_task(request, state_callback=state_callback, cancel_event=cancel_event)
        return {
            "success": res.success,
            "summary": res.summary,
            "task_type": res.task_type,
            "plan_steps": res.plan_steps,
            "files_searched": res.files_searched,
            "files_read": res.files_read,
            "files_modified": res.files_modified,
            "diffs": res.diffs,
            "tests_run": res.tests_run,
            "tests_passed": res.tests_passed,
            "tests_failed": res.tests_failed,
            "debug_iterations": res.debug_iterations,
            "findings": res.findings,
            "think_mode": mode,
        }
