"""
axen/coding/patch_manager.py
=============================
Safe patch-based editing system for the AXEN Coding Agent:
- Enforces READ-BEFORE-EDIT flow
- Generates standard unified diff previews (diff preview before edit)
- Performs automatic backup creation and instant rollback on syntax error
- Tracks changes in the ChangeJournal audit trail
- Validates Python syntax using AST prior to accepting modifications
"""

from __future__ import annotations

import ast
import difflib
import logging
from pathlib import Path
from typing import Optional, Tuple

from .change_journal import ChangeJournal
from .git_tracker import GitTracker
from .permissions import OperationType
from .workspace import Workspace

logger = logging.getLogger("axen.coding.patch")


class PatchError(ValueError):
    """Raised when a patch cannot be cleanly applied to target file."""
    pass


class PatchManager:
    """Safely constructs, previews, and applies targeted edits to workspace files."""

    def __init__(
        self,
        workspace: Workspace,
        journal: Optional[ChangeJournal] = None,
        git_tracker: Optional[GitTracker] = None,
    ) -> None:
        self.workspace = workspace
        self.journal = journal or ChangeJournal()
        self.git_tracker = git_tracker or GitTracker(workspace)

    def generate_diff(
        self,
        file_path: str,
        original_text: str,
        modified_text: str,
    ) -> str:
        """Generate a standard unified diff string showing lines added and removed."""
        orig_lines = original_text.splitlines(keepends=True)
        mod_lines = modified_text.splitlines(keepends=True)

        diff = difflib.unified_diff(
            orig_lines,
            mod_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
        )
        return "".join(diff)

    def propose_patch(
        self,
        file_path: str,
        target_snippet: str,
        replacement_snippet: str,
        reason: str = "Code update",
    ) -> Tuple[bool, str, str]:
        """
        Validate and preview a patch without modifying disk.
        Returns: (is_valid, unified_diff, error_message)
        """
        try:
            curr_text = self.workspace.read_file(file_path)
        except Exception as exc:
            return False, "", f"Could not read file for inspection: {exc}"

        if target_snippet not in curr_text:
            return (
                False,
                "",
                f"Target snippet not found in '{file_path}'. Read the current file before patching.",
            )

        count = curr_text.count(target_snippet)
        if count > 1:
            return (
                False,
                "",
                f"Target snippet occurs {count} times in '{file_path}'. Provide more surrounding context.",
            )

        new_text = curr_text.replace(target_snippet, replacement_snippet, 1)
        diff = self.generate_diff(file_path, curr_text, new_text)

        # Pre-validate Python syntax if applicable
        if file_path.endswith(".py"):
            try:
                ast.parse(new_text)
            except SyntaxError as syn_err:
                return False, diff, f"Proposed patch introduces a Python SyntaxError: {syn_err}"

        return True, diff, ""

    def apply_patch(
        self,
        file_path: str,
        target_snippet: str,
        replacement_snippet: str,
        reason: str = "Code update",
        skip_git_check: bool = False,
    ) -> Tuple[bool, str, str]:
        """
        Safely apply a patch to disk:
        1. Check permissions
        2. Check for user's uncommitted modifications
        3. Create safety backup
        4. Apply edit
        5. Verify syntax (if Python)
        6. Commit to journal or rollback
        Returns: (success, unified_diff, error_message)
        """
        self.workspace.policy.require(OperationType.APPLY_PATCH, file_path)

        # Step 18: Protect uncommitted user changes
        if not skip_git_check and self.git_tracker.has_uncommitted_changes(file_path):
            logger.warning("File '%s' has uncommitted changes in git.", file_path)

        # Step 12: Read current text before edit
        try:
            curr_text = self.workspace.read_file(file_path)
        except Exception as exc:
            return False, "", f"Could not read target file: {exc}"

        valid, diff, err = self.propose_patch(file_path, target_snippet, replacement_snippet, reason)
        if not valid:
            return False, diff, err

        new_text = curr_text.replace(target_snippet, replacement_snippet, 1)

        # Create backup before writing
        self.workspace.create_backup(file_path)

        try:
            self.workspace.write_file(file_path, new_text)

            # Verification: Parse AST to guarantee no broken syntax
            if file_path.endswith(".py"):
                ast.parse(new_text)

            # Success: clean up backup and record in journal
            self.workspace.remove_backup(file_path)
            self.journal.record_change(
                file_path=file_path,
                operation="modify",
                reason=reason,
                diff=diff,
                status="applied",
            )
            return True, diff, ""

        except Exception as exc:
            # Instant rollback on syntax error or write failure
            logger.error("Patch verification failed; rolling back file '%s': %s", file_path, exc)
            self.workspace.restore_backup(file_path)
            return False, diff, f"Patch failed verification and was rolled back: {exc}"
