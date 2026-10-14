"""
axen/coding/change_journal.py
==============================
Structured change journal and audit trail for the AXEN Coding Agent.
Records engineering actions taken during coding sessions:
- Which files changed
- Nature of operation (modify, create, delete)
- Diff preview
- Associated tests
- Verification outcome
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from typing import Any, Dict, List, Optional


@dataclass
class ChangeRecord:
    file_path: str
    operation: str  # "modify", "create", "delete"
    reason: str
    diff: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "applied"  # "proposed", "applied", "reverted"
    tests_run: List[str] = field(default_factory=list)
    verified: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


class ChangeJournal:
    """Maintains an audit trail of all changes applied during a coding session."""

    def __init__(self) -> None:
        self.records: List[ChangeRecord] = []

    def record_change(
        self,
        file_path: str,
        operation: str,
        reason: str,
        diff: str,
        status: str = "applied",
        tests_run: Optional[List[str]] = None,
        verified: bool = False,
        original_content: Optional[str] = None,
    ) -> ChangeRecord:
        """Add a change record to the journal."""
        details = {}
        if original_content is not None:
            details["original_content"] = original_content

        rec = ChangeRecord(
            file_path=file_path,
            operation=operation,
            reason=reason,
            diff=diff,
            status=status,
            tests_run=tests_run or [],
            verified=verified,
            details=details,
        )
        self.records.append(rec)
        return rec

    def get_latest_original_content(self, file_path: str) -> Optional[str]:
        """Retrieve original content for a file from the most recent change record."""
        clean = file_path.replace("\\", "/").strip("/")
        for rec in reversed(self.records):
            if rec.file_path.replace("\\", "/").strip("/") == clean and "original_content" in rec.details:
                return rec.details["original_content"]
        return None

    def mark_verified(self, file_path: str, tests_passed: Optional[List[str]] = None) -> None:
        """Mark the most recent change to a file as verified."""
        for rec in reversed(self.records):
            if rec.file_path == file_path:
                rec.verified = True
                if tests_passed:
                    rec.tests_run.extend(tests_passed)
                break

    def get_summary(self) -> str:
        """Format a human-readable engineering summary of changes."""
        if not self.records:
            return "No files were modified."

        lines = ["### Changes Applied:"]
        for r in self.records:
            status_symbol = "✓" if r.verified else "•"
            lines.append(f"{status_symbol} {r.file_path} ({r.operation}) — {r.reason}")

        return "\n".join(lines)

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Return serialized list of records."""
        return [asdict(r) for r in self.records]
