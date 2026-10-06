"""
tests/test_coding_agent.py
===========================
Comprehensive Test Matrix for AXEN AI Phase 5 Coding Agent:
- Workspace security & path traversal protection
- Permission level enforcement
- Project inspection & project map generation
- Fast local code, symbol, and file search
- Bounded relevant-file selection and context bounding
- Safe patch-based editing & unified diff previews
- Python AST syntax checking & instant rollback
- Controlled command execution & dangerous command blacklisting
- Test execution & structured failure parsing (ErrorAnalyzer)
- Bounded Failure Debugging Loop (max 3 retries)
- Controlled Test Project 1 (Calculator bug: return a - b -> a + b)
- Controlled Test Project 2 (Function typo / import repair)
- Code Review mode (read-only audit findings)
- Code Generation vs Workspace Editing separation
- Git safety & uncommitted change protection
- Cancellation safety
- Memory & RAG integration with coding agent
- Thinking UI state emissions
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import tempfile
import threading
import pytest

from axen.coding import (
    ChangeJournal,
    CodeContextManager,
    CodeSearch,
    CodingAgent,
    CodingTaskType,
    CommandResult,
    CommandRunner,
    CommandSecurityError,
    GitSafetyError,
    GitTracker,
    OperationType,
    PatchManager,
    PermissionLevel,
    PermissionPolicy,
    ProjectInspector,
    TestRunner,
    Workspace,
    WorkspaceSecurityError,
)
from axen.agent.classifier import TaskClassifier
from axen.agent.loop import AgentLoop
from axen.assistant.controller import AssistantController
from axen.memory.conversation_memory import ConversationMemory
from axen.memory.long_term_memory import LongTermMemory
from axen.tools import build_default_registry


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_project(tmp_path):
    """Creates a controlled sample project workspace."""
    proj_dir = tmp_path / "sample_project"
    proj_dir.mkdir(parents=True, exist_ok=True)

    # 1. Config / Manifest
    (proj_dir / "pyproject.toml").write_text(
        '[project]\nname = "sample-project"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )

    # 2. Source code with an inverted bug: add returns a - b
    src_dir = proj_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "calculator.py").write_text(
        "class Calculator:\n"
        "    def add(self, a, b):\n"
        "        return a - b\n\n"
        "    def multiply(self, a, b):\n"
        "        return a * b\n",
        encoding="utf-8",
    )

    # 3. Test suite verifying addition and multiplication
    test_dir = proj_dir / "tests"
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "test_calculator.py").write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).parent.parent))\n"
        "from src.calculator import Calculator\n\n"
        "def test_add():\n"
        "    calc = Calculator()\n"
        "    assert calc.add(2, 3) == 5\n\n"
        "def test_multiply():\n"
        "    calc = Calculator()\n"
        "    assert calc.multiply(3, 4) == 12\n",
        encoding="utf-8",
    )

    return proj_dir


# ---------------------------------------------------------------------------
# Test Suite: Phase 5 Coding Agent
# ---------------------------------------------------------------------------

class TestPhase5CodingAgent:

    # -----------------------------------------------------------------------
    # TEST 1 — WORKSPACE PATH TRAVERSAL & ESCAPE SECURITY
    # -----------------------------------------------------------------------
    def test_workspace_traversal_blocked(self, temp_project):
        """Parent directory traversal (../../) is strictly rejected by workspace."""
        ws = Workspace(temp_project)

        with pytest.raises(WorkspaceSecurityError):
            ws.resolve_path("../../secret.txt")

        with pytest.raises(WorkspaceSecurityError):
            ws.resolve_path("../other_folder/file.py")

    def test_workspace_absolute_path_escape_blocked(self, temp_project):
        """Absolute path pointing outside workspace is strictly rejected."""
        ws = Workspace(temp_project)

        with pytest.raises(WorkspaceSecurityError):
            ws.resolve_path(r"C:\Windows\System32\drivers\etc\hosts")

        with pytest.raises(WorkspaceSecurityError):
            ws.resolve_path(Path.home() / "passwords.txt")

    def test_system_root_cannot_be_workspace(self):
        """Root system directory cannot become an approved workspace."""
        with pytest.raises(WorkspaceSecurityError):
            Workspace(r"C:\Windows")

    # -----------------------------------------------------------------------
    # TEST 2 — PERMISSION POLICY ENFORCEMENT
    # -----------------------------------------------------------------------
    def test_permission_levels(self, temp_project):
        """Verifies READ_ONLY, PROPOSE, EDIT, and EXECUTE permission boundaries."""
        # 1. READ_ONLY allows reading, forbids writing
        ro_policy = PermissionPolicy(PermissionLevel.READ_ONLY)
        ws_ro = Workspace(temp_project, policy=ro_policy)
        assert ws_ro.read_file("src/calculator.py") != ""
        with pytest.raises(PermissionError):
            ws_ro.write_file("src/calculator.py", "new content")

        # 2. PROPOSE allows diff generation, forbids direct writing
        prop_policy = PermissionPolicy(PermissionLevel.PROPOSE)
        ws_prop = Workspace(temp_project, policy=prop_policy)
        with pytest.raises(PermissionError):
            ws_prop.write_file("src/calculator.py", "new content")

        # 3. EDIT allows writing, forbids command execution
        edit_policy = PermissionPolicy(PermissionLevel.EDIT)
        ws_edit = Workspace(temp_project, policy=edit_policy)
        cmd_runner = CommandRunner(ws_edit)
        with pytest.raises(PermissionError):
            cmd_runner.run("echo hello")

    # -----------------------------------------------------------------------
    # TEST 3 — PROJECT INSPECTION & PROJECT MAP
    # -----------------------------------------------------------------------
    def test_project_inspection_and_map(self, temp_project):
        """Inspector identifies manifests, language, test framework, and generates map."""
        ws = Workspace(temp_project)
        inspector = ProjectInspector(ws)
        summary = inspector.get_summary()

        assert summary["primary_language"] == "python"
        assert summary["test_runner"] == "pytest"
        assert "pyproject.toml" in summary["manifests"]
        assert "sample_project" in summary["project_map"]
        assert "src/" in summary["project_map"]
        assert "calculator.py" in summary["project_map"]

    # -----------------------------------------------------------------------
    # TEST 4 — FAST LOCAL CODE SEARCH (FILE, TEXT, SYMBOL)
    # -----------------------------------------------------------------------
    def test_code_search(self, temp_project):
        """CodeSearch finds filenames, text lines, and symbol definitions."""
        ws = Workspace(temp_project)
        searcher = CodeSearch(ws)

        # File search
        files = searcher.search_files("calc")
        assert len(files) >= 1
        assert any("calculator.py" in f for f in files)

        # Text search
        text_matches = searcher.search_text("def multiply")
        assert len(text_matches) >= 1
        assert text_matches[0].line_number > 0

        # Symbol search
        sym_matches = searcher.search_symbols("Calculator", symbol_type="class")
        assert len(sym_matches) >= 1
        assert sym_matches[0].symbol_name == "Calculator"

    # -----------------------------------------------------------------------
    # TEST 5 — RELEVANT FILE SELECTION & CONTEXT BOUNDS
    # -----------------------------------------------------------------------
    def test_relevant_file_selection_and_bounds(self, temp_project):
        """ContextManager selects top relevant files and bounds character length."""
        ws = Workspace(temp_project)
        searcher = CodeSearch(ws)
        ctx_mgr = CodeContextManager(ws, searcher)

        selected = ctx_mgr.select_relevant_files("Fix the add method in calculator")
        assert len(selected) >= 1
        assert "src/calculator.py" in selected[0]

        fc = ctx_mgr.build_file_context(selected[0])
        assert fc is not None
        assert "Calculator" in fc.symbols
        assert len(fc.content_snippet) < ctx_mgr.MAX_FILE_CHARS

    # -----------------------------------------------------------------------
    # TEST 6 — SAFE PATCH-BASED EDITING & DIFF PREVIEW
    # -----------------------------------------------------------------------
    def test_safe_patch_and_diff(self, temp_project):
        """PatchManager generates unified diff and applies modification safely."""
        ws = Workspace(temp_project, policy=PermissionPolicy(PermissionLevel.FULL_WORKSPACE))
        patcher = PatchManager(ws)

        # 1. Propose patch
        ok, diff, err = patcher.propose_patch(
            file_path="src/calculator.py",
            target_snippet="return a - b",
            replacement_snippet="return a + b",
        )
        assert ok is True
        assert "--- a/src/calculator.py" in diff
        assert "+++ b/src/calculator.py" in diff
        assert "-        return a - b" in diff
        assert "+        return a + b" in diff

        # 2. Apply patch
        applied_ok, _, _ = patcher.apply_patch(
            file_path="src/calculator.py",
            target_snippet="return a - b",
            replacement_snippet="return a + b",
            reason="Fix subtraction bug",
        )
        assert applied_ok is True
        assert "return a + b" in ws.read_file("src/calculator.py")

    # -----------------------------------------------------------------------
    # TEST 7 — PYTHON SYNTAX CHECK & INSTANT ROLLBACK ON ERROR
    # -----------------------------------------------------------------------
    def test_syntax_error_rollback(self, temp_project):
        """Patch introducing broken Python syntax is rolled back immediately."""
        ws = Workspace(temp_project, policy=PermissionPolicy(PermissionLevel.FULL_WORKSPACE))
        patcher = PatchManager(ws)

        # Attempt to inject syntax error: broken indentation / invalid token
        ok, diff, err = patcher.apply_patch(
            file_path="src/calculator.py",
            target_snippet="return a - b",
            replacement_snippet="return a +++ ??? invalid syntax :::",
        )
        assert ok is False
        assert "introduces a Python SyntaxError" in err or "SyntaxError" in err
        # File must remain uncorrupted
        assert "return a - b" in ws.read_file("src/calculator.py")

    # -----------------------------------------------------------------------
    # TEST 8 — COMMAND RUNNER & DANGEROUS COMMAND BLACKLIST
    # -----------------------------------------------------------------------
    def test_dangerous_commands_blocked(self, temp_project):
        """Dangerous commands like rm -rf, format, or shell injection are blocked."""
        ws = Workspace(temp_project, policy=PermissionPolicy(PermissionLevel.FULL_WORKSPACE))
        runner = CommandRunner(ws)

        with pytest.raises(CommandSecurityError):
            runner.run("rm -rf /")

        with pytest.raises(CommandSecurityError):
            runner.run("format c:")

        with pytest.raises(CommandSecurityError):
            runner.run("shutdown /s /t 0")

    # -----------------------------------------------------------------------
    # TEST 9 — TEST RUNNER & ERROR ANALYZER
    # -----------------------------------------------------------------------
    def test_test_runner_detects_failure(self, temp_project):
        """TestRunner executes test suite and identifies failing assertion."""
        ws = Workspace(temp_project, policy=PermissionPolicy(PermissionLevel.FULL_WORKSPACE))
        cmd_runner = CommandRunner(ws)
        t_runner = TestRunner(ws, cmd_runner)

        result = t_runner.run_tests()
        # Because add returns a - b, test_add will fail initially
        assert result.failed >= 1
        assert len(result.failures) >= 1
        assert "test_add" in result.failures[0].test_name

    # -----------------------------------------------------------------------
    # TEST 10 — END-TO-END CODING AGENT BUG FIX & VERIFICATION (TEST PROJECT 1)
    # -----------------------------------------------------------------------
    def test_coding_agent_fixes_calculator_bug(self, temp_project):
        """CodingAgent reproduces failing test, patches add(), retests, and verifies success."""
        ws = Workspace(temp_project, policy=PermissionPolicy(PermissionLevel.FULL_WORKSPACE))
        agent = CodingAgent(ws)

        result = agent.run_task("Fix the add function in the calculator")

        assert result.success is True
        assert "src/calculator.py" in result.files_modified
        assert result.tests_passed >= 2
        assert result.tests_failed == 0
        assert "=== CODING AGENT DEBUG ===" in result.debug_block
        assert "verification: PASS" in result.debug_block

        # Confirm file on disk now has correct addition
        content = ws.read_file("src/calculator.py")
        assert "return a + b" in content

    # -----------------------------------------------------------------------
    # TEST 11 — SECOND TEST PROJECT (TYPO / FUNCTION REPAIR)
    # -----------------------------------------------------------------------
    def test_second_project_typo_repair(self, tmp_path):
        """CodingAgent identifies and repairs a function naming typo."""
        proj2 = tmp_path / "project_math"
        proj2.mkdir()
        src = proj2 / "src"
        src.mkdir()
        (src / "math_ops.py").write_text(
            "def multipyl(a, b):\n    return a * b\n",
            encoding="utf-8",
        )
        tests = proj2 / "tests"
        tests.mkdir()
        (tests / "test_ops.py").write_text(
            "import sys\nfrom pathlib import Path\n"
            "sys.path.insert(0, str(Path(__file__).parent.parent))\n"
            "from src.math_ops import multiply\n\n"
            "def test_multiply():\n"
            "    assert multiply(3, 5) == 15\n",
            encoding="utf-8",
        )

        ws = Workspace(proj2, policy=PermissionPolicy(PermissionLevel.FULL_WORKSPACE))
        agent = CodingAgent(ws)
        res = agent.run_task("Fix the multiply function in math_ops.py")

        assert res.success is True
        assert "def multiply(a, b):" in ws.read_file("src/math_ops.py")

    # -----------------------------------------------------------------------
    # TEST 12 — READ-ONLY CODE REVIEW MODE
    # -----------------------------------------------------------------------
    def test_code_review_mode(self, temp_project):
        """Code review mode returns structured findings and makes zero modifications."""
        ws = Workspace(temp_project, policy=PermissionPolicy(PermissionLevel.READ_ONLY))
        agent = CodingAgent(ws)

        # Inject a bare except block
        orig_calc = ws.read_file("src/calculator.py")
        ws.policy.level = PermissionLevel.FULL_WORKSPACE
        ws.write_file("src/calculator.py", orig_calc + "\ndef helper():\n    try:\n        pass\n    except:\n        pass\n")
        ws.policy.level = PermissionLevel.READ_ONLY

        res = agent.run_task("Review this project for code quality and bugs")
        assert res.success is True
        assert res.task_type == CodingTaskType.CODE_REVIEW
        assert len(res.findings) >= 1
        assert any(f["severity"] == "WARNING" for f in res.findings)
        assert len(res.files_modified) == 0  # No files modified in review mode

    # -----------------------------------------------------------------------
    # TEST 13 — SEPARATION: CODE GENERATION VS PROJECT EDITING
    # -----------------------------------------------------------------------
    def test_code_generation_does_not_modify_project(self, temp_project):
        """Generic programming prompt ('write a python function...') does not edit project."""
        classifier = TaskClassifier()
        gen_intent = classifier.analyze_intent("Write a python function to reverse a string")
        assert gen_intent.intent == "programming"
        assert gen_intent.primary_tool != "coding"

        proj_intent = classifier.analyze_intent("Fix the bug in calculator.py in my project")
        assert proj_intent.intent == "coding"
        assert proj_intent.primary_tool == "coding"

    # -----------------------------------------------------------------------
    # TEST 14 — GIT SAFETY & DESTRUCTIVE COMMAND REJECTION
    # -----------------------------------------------------------------------
    def test_git_destructive_commands_blocked(self, temp_project):
        """GitTracker strictly blocks automated git commit, push, and reset --hard."""
        ws = Workspace(temp_project)
        git_tracker = GitTracker(ws)
        # Mock git repo detection for testing
        git_tracker._is_repo = True

        with pytest.raises(GitSafetyError):
            git_tracker._run_git(["commit", "-m", "auto commit"])

        with pytest.raises(GitSafetyError):
            git_tracker._run_git(["push", "origin", "main"])

        with pytest.raises(GitSafetyError):
            git_tracker._run_git(["reset", "--hard", "HEAD"])

    # -----------------------------------------------------------------------
    # TEST 15 — CANCELLATION SAFETY
    # -----------------------------------------------------------------------
    def test_cancellation_safety(self, temp_project):
        """Cancel event halts execution cleanly without corrupting files."""
        ws = Workspace(temp_project, policy=PermissionPolicy(PermissionLevel.FULL_WORKSPACE))
        agent = CodingAgent(ws)

        cancel_ev = threading.Event()
        cancel_ev.set()  # Cancel immediately

        res = agent.run_task("Fix the add function", cancel_event=cancel_ev)
        assert res.success is False
        assert "cancelled" in res.summary.lower()
        assert len(res.files_modified) == 0

    # -----------------------------------------------------------------------
    # TEST 16 — REAL THINKING UI STATE EMISSIONS
    # -----------------------------------------------------------------------
    def test_thinking_ui_states_emitted(self, temp_project):
        """Verifies that genuine thinking states are emitted in sequence."""
        ws = Workspace(temp_project, policy=PermissionPolicy(PermissionLevel.FULL_WORKSPACE))
        agent = CodingAgent(ws)

        states_observed = []
        def on_state(state, msg, meta):
            states_observed.append(state)

        agent.run_task("Fix the add function", state_callback=on_state)

        assert "UNDERSTANDING" in states_observed
        assert "SEARCHING" in states_observed
        assert "READING" in states_observed
        assert "PLANNING" in states_observed
        assert "TESTING" in states_observed
        assert "COMPLETED" in states_observed

    # -----------------------------------------------------------------------
    # TEST 17 — ASSISTANT CONTROLLER INTEGRATION
    # -----------------------------------------------------------------------
    def test_assistant_controller_coding_dispatch(self, temp_project):
        """AssistantController coordinates workspace coding task through AgentLoop."""
        controller = AssistantController(workspace_path=str(temp_project))
        ans = controller.respond("Fix the bug in calculator.py in my project")

        assert "calculator.py" in ans or "Completed" in ans
        assert controller.last_agent_state is not None
        assert controller.last_agent_state.task_type in ("coding", "software_engineering")
