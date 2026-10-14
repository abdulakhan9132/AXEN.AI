"""
axen/coding/workspace.py
========================
Controlled Project Workspace for AXEN Software Engineering Agent.

Enforces strict directory boundary confinement:
- No directory traversal (../) escape
- No absolute path escape
- No symlink / junction escape
- No system directory access (C:\\Windows, C:\\Program Files, C:\\, etc.)
- All file operations are validated against the approved workspace root
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any, List, Optional, Tuple

from .permissions import OperationType, PermissionPolicy


# Disallowed system root and sensitive paths (case-insensitive prefixes)
_BLOCKED_SYSTEM_PATHS = (
    r"c:\windows",
    r"c:\program files",
    r"c:\program files (x86)",
    r"c:\system volume information",
    r"c:\$recycle.bin",
    r"c:\recovery",
    r"c:\boot",
    r"/etc",
    r"/usr",
    r"/var",
    r"/bin",
    r"/sbin",
    r"/sys",
    r"/proc",
)


class WorkspaceSecurityError(PermissionError):
    """Raised when an operation attempts to escape or violate the workspace boundary."""
    pass


class Workspace:
    """Represents a validated, user-approved project directory."""

    def __init__(
        self,
        root_path: Path | str,
        policy: Optional[PermissionPolicy] = None,
        max_file_size_bytes: int = 500 * 1024,  # 500 KB limit for safe reads
    ) -> None:
        raw_path = Path(root_path).resolve()
        if not raw_path.exists():
            raise FileNotFoundError(f"Workspace path does not exist: {raw_path}")
        if not raw_path.is_dir():
            raise NotADirectoryError(f"Workspace root must be a directory: {raw_path}")

        # Block entire drive root or sensitive system directories from becoming a workspace
        norm_str = str(raw_path).lower().rstrip("\\/")
        if len(norm_str) <= 3 and norm_str.endswith(":"):
            raise WorkspaceSecurityError(f"Root drive '{raw_path}' cannot be an approved workspace.")

        for blocked in _BLOCKED_SYSTEM_PATHS:
            if norm_str.startswith(blocked):
                raise WorkspaceSecurityError(f"System directory '{raw_path}' cannot be an approved workspace.")

        self.root_path: Path = raw_path
        self.policy: PermissionPolicy = policy or PermissionPolicy()
        self.max_file_size_bytes = max_file_size_bytes

    def resolve_path(self, relative_or_absolute: Path | str) -> Path:
        """
        Safely resolve a path within the workspace.
        Raises WorkspaceSecurityError if the path escapes the workspace root.
        """
        p = Path(relative_or_absolute)
        if p.is_absolute():
            resolved = p.resolve()
        else:
            resolved = (self.root_path / p).resolve()

        # Strict containment check
        try:
            # Check if resolved is relative to root_path
            resolved.relative_to(self.root_path)
        except ValueError:
            raise WorkspaceSecurityError(
                f"Security Violation: Path '{relative_or_absolute}' escapes the approved "
                f"workspace '{self.root_path}'."
            )

        # Symlink target resolution check
        if resolved.is_symlink():
            target = resolved.resolve()
            try:
                target.relative_to(self.root_path)
            except ValueError:
                raise WorkspaceSecurityError(
                    f"Security Violation: Symlink '{relative_or_absolute}' points outside "
                    f"the approved workspace to '{target}'."
                )

        return resolved

    def is_inside(self, path: Path | str) -> bool:
        """Return True if path is safely inside the approved workspace."""
        try:
            self.resolve_path(path)
            return True
        except (WorkspaceSecurityError, ValueError):
            return False

    def read_file(self, path: Path | str) -> str:
        """Read text from a workspace file safely."""
        self.policy.require(OperationType.READ, str(path))
        target = self.resolve_path(path)

        if not target.exists():
            raise FileNotFoundError(f"File not found in workspace: {path}")
        if target.is_dir():
            raise IsADirectoryError(f"Path is a directory: {path}")

        file_size = target.stat().st_size
        if file_size > self.max_file_size_bytes:
            raise ValueError(
                f"File '{target.name}' size ({file_size} bytes) exceeds safety limit "
                f"({self.max_file_size_bytes} bytes)."
            )

        return target.read_text(encoding="utf-8", errors="replace")

    def write_file(self, path: Path | str, content: str, create_parents: bool = True) -> int:
        """Write text to a workspace file safely."""
        self.policy.require(OperationType.WRITE_FILE, str(path))
        target = self.resolve_path(path)

        if create_parents and not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)

        target.write_text(content, encoding="utf-8")
        return len(content)

    def delete_file(self, path: Path | str) -> bool:
        """Delete a file inside the workspace safely."""
        self.policy.require(OperationType.DELETE_FILE, str(path))
        target = self.resolve_path(path)

        if not target.exists():
            return False
        if target.is_dir():
            raise IsADirectoryError(f"Use directory removal for directories: {path}")

        target.unlink()
        return True

    def create_backup(self, path: Path | str) -> Optional[Path]:
        """Create a safety backup copy of a file before modification."""
        target = self.resolve_path(path)
        if not target.exists() or target.is_dir():
            return None

        backup_path = target.with_suffix(target.suffix + ".axen_backup")
        shutil.copy2(target, backup_path)
        return backup_path

    def restore_backup(self, path: Path | str) -> bool:
        """Restore a file from its safety backup."""
        target = self.resolve_path(path)
        backup_path = target.with_suffix(target.suffix + ".axen_backup")
        if not backup_path.exists():
            return False

        shutil.copy2(backup_path, target)
        backup_path.unlink()
        return True

    def remove_backup(self, path: Path | str) -> None:
        """Clean up the safety backup file."""
        target = self.resolve_path(path)
        backup_path = target.with_suffix(target.suffix + ".axen_backup")
        if backup_path.exists():
            backup_path.unlink()

    def list_files(
        self,
        sub_dir: Optional[Path | str] = None,
        include_ignored: bool = False,
        max_files: Optional[int] = None,
    ) -> List[Path]:
        """List files in workspace, automatically filtering noisy/internal folders."""
        start_dir = self.resolve_path(sub_dir) if sub_dir else self.root_path
        results: list[Path] = []

        ignore_dirs = {
            ".git", ".venv", "venv", "__pycache__", "node_modules",
            ".pytest_cache", ".mypy_cache", "dist", "build", ".egg-info",
            ".agents", ".codex", ".gsd-staging", "%SystemDrive%",
        }

        for root, dirs, files in os.walk(start_dir):
            dirs[:] = [d for d in dirs if self.is_inside(Path(root) / d)]
            if not include_ignored:
                # Prune noisy directories in-place
                dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.endswith(".egg-info")]

            for f in sorted(files):
                # Filter out backup files and compiled binaries
                if f.endswith((".axen_backup", ".pyc", ".pyo", ".exe", ".dll", ".so", ".dylib")):
                    continue
                full_path = Path(root) / f
                if not self.is_inside(full_path):
                    continue
                results.append(full_path)
                if max_files and len(results) >= max_files:
                    return results

        return results
