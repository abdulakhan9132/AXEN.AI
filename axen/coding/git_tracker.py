"""
axen/coding/git_tracker.py
===========================
Read-only Git awareness and user-change protection for AXEN Coding Agent.
Ensures:
- Uncommitted user modifications are detected before file edits
- AXEN never executes destructive git commands (commit, push, hard reset, clean)
- Branch, status, and diff information are safely inspectable
"""

from __future__ import annotations

import logging
from pathlib import Path
import subprocess
from typing import List, Optional, Set

from .workspace import Workspace

logger = logging.getLogger("axen.coding.git")

# Destructive git commands that must never be run autonomously
DESTRUCTIVE_GIT_COMMANDS = frozenset({
    "commit", "push", "reset", "clean", "rebase", "merge",
    "cherry-pick", "revert", "tag", "branch -D", "checkout -f",
})


class GitSafetyError(PermissionError):
    """Raised when an automated action attempts a restricted or destructive Git command."""
    pass


class GitTracker:
    """Provides safe, read-only Git status and protects uncommitted user work."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self._is_repo = self._check_is_git_repo()

    @property
    def is_git_repository(self) -> bool:
        return self._is_repo

    def _run_git(self, args: List[str], timeout: float = 5.0) -> Optional[str]:
        """Execute a read-only git command confined to the workspace."""
        if not self._is_repo:
            return None

        # Check against destructive operations
        cmd_str = " ".join(args).lower()
        for forbidden in DESTRUCTIVE_GIT_COMMANDS:
            if forbidden in cmd_str:
                raise GitSafetyError(
                    f"Blocked destructive Git command: 'git {cmd_str}'. "
                    "AXEN does not perform automatic commits or destructive Git operations."
                )

        try:
            res = subprocess.run(
                ["git"] + args,
                cwd=str(self.workspace.root_path),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if res.returncode == 0:
                return res.stdout.strip()
            return None
        except (subprocess.SubprocessError, FileNotFoundError) as exc:
            logger.debug("Git command execution failed: %s", exc)
            return None

    def _check_is_git_repo(self) -> bool:
        """Check if workspace is a valid Git repository."""
        git_dir = self.workspace.root_path / ".git"
        if git_dir.exists():
            return True
        try:
            res = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=str(self.workspace.root_path),
                capture_output=True,
                text=True,
                timeout=3.0,
            )
            return res.returncode == 0 and res.stdout.strip() == "true"
        except Exception:
            return False

    def get_current_branch(self) -> str:
        """Return the current active git branch name."""
        branch = self._run_git(["branch", "--show-current"])
        return branch or "unknown"

    def get_modified_files(self) -> Set[str]:
        """Return relative paths of all files modified in working tree or staged."""
        output = self._run_git(["status", "--porcelain"])
        if not output:
            return set()

        modified = set()
        for line in output.splitlines():
            if len(line) >= 4:
                file_rel = line[3:].strip().replace("\\", "/")
                # In case of rename: 'orig -> renamed'
                if " -> " in file_rel:
                    file_rel = file_rel.split(" -> ")[-1]
                modified.add(file_rel)
        return modified

    def has_uncommitted_changes(self, file_path: Optional[str] = None) -> bool:
        """
        Check if the workspace or a specific file has uncommitted changes.
        Protects user's in-progress work from being overwritten.
        """
        modified = self.get_modified_files()
        if file_path is None:
            return len(modified) > 0

        norm_path = str(file_path).replace("\\", "/").strip("/")
        return norm_path in modified

    def get_diff(self, file_path: Optional[str] = None) -> str:
        """Return unstaged diff for workspace or specific file."""
        args = ["diff"]
        if file_path:
            args.extend(["--", file_path])
        return self._run_git(args) or ""
