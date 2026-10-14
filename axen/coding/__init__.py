"""
axen/coding/__init__.py
========================
AXEN Software Engineering / Coding Agent Module:
- Workspace & Security
- Permissions & Policies
- Project Inspection & Project Map
- Local Code Search & Symbol Indexing
- Context Selection & Management
- Git Awareness & User Modification Protection
- Controlled Command Runner
- Test Runner & Failure Diagnosis
- Patch-based Editing & Unified Diff Previews
- Change Journal Audit Trail
- Master CodingAgent
"""

from .change_journal import ChangeJournal, ChangeRecord
from .code_context import CodeContextManager, FileContext
from .code_search import CodeSearch, SearchMatch
from .coding_agent import CodingAgent, CodingAgentResult, CodingTaskType
from .command_runner import CommandResult, CommandRunner, CommandSecurityError
from .git_tracker import GitSafetyError, GitTracker
from .patch_manager import PatchError, PatchManager
from .permissions import OperationType, PermissionLevel, PermissionPolicy
from .project_inspector import ProjectInspector
from .test_runner import FailureInfo, TestResult, TestRunner
from .workspace import Workspace, WorkspaceSecurityError
from .coding_service import CodingService

__all__ = [
    "Workspace",
    "WorkspaceSecurityError",
    "PermissionLevel",
    "PermissionPolicy",
    "OperationType",
    "ProjectInspector",
    "CodeSearch",
    "SearchMatch",
    "CodeContextManager",
    "FileContext",
    "GitTracker",
    "GitSafetyError",
    "CommandRunner",
    "CommandResult",
    "CommandSecurityError",
    "TestRunner",
    "TestResult",
    "FailureInfo",
    "PatchManager",
    "PatchError",
    "ChangeJournal",
    "ChangeRecord",
    "CodingAgent",
    "CodingAgentResult",
    "CodingTaskType",
    "CodingService",
]
