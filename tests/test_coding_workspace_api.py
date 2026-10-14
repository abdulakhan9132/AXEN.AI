"""
tests/test_coding_workspace_api.py
==================================
Comprehensive API Integration tests for Phase 5 Codex-Style Coding Workspace:
- Workspace info and switching
- File tree generation
- File reading with path traversal prevention
- File saving with AST syntax checking
- Code search (text, symbol, file)
- Git status & diff
- Patch preview and application
- File rollback/undo
- Test runner invocation
- AI coding chat execution with Think Engine modes
"""

import os
from pathlib import Path
import shutil
import tempfile
import pytest
from starlette.testclient import TestClient

from app.ui.app import app
from axen.coding.coding_service import CodingService
from axen.coding.permissions import PermissionLevel


@pytest.fixture
def test_workspace():
    """Create a temporary git-initialized workspace with test files."""
    d = tempfile.mkdtemp(prefix="axen_coding_ws_")
    ws_path = Path(d)

    # Initialize a dummy calculator file and test
    calc_file = ws_path / "calculator.py"
    calc_file.write_text("def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n", encoding="utf-8")

    test_file = ws_path / "test_calc.py"
    test_file.write_text("from calculator import add, sub\n\ndef test_add():\n    assert add(2, 3) == 5\n\ndef test_sub():\n    assert sub(5, 2) == 3\n", encoding="utf-8")

    # Set as active coding workspace
    service = CodingService.get_instance()
    service.set_workspace(ws_path, permission_level=PermissionLevel.EXECUTE)

    yield ws_path

    # Teardown
    shutil.rmtree(d, ignore_errors=True)
    # Restore original workspace to current directory
    service.set_workspace(Path.cwd(), permission_level=PermissionLevel.EXECUTE)


def test_workspace_info_endpoint(test_workspace):
    client = TestClient(app)
    res = client.get("/api/coding/workspace")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    ws = data["workspace"]
    assert ws["workspace_name"] == test_workspace.name
    assert ws["permission_level"] == "EXECUTE"
    assert ws["total_files"] >= 2


def test_file_tree_endpoint(test_workspace):
    client = TestClient(app)
    res = client.get("/api/coding/tree?max_depth=3")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    tree = data["tree"]
    assert tree["type"] == "directory"
    child_names = [c["name"] for c in tree.get("children", [])]
    assert "calculator.py" in child_names
    assert "test_calc.py" in child_names


def test_file_read_endpoint(test_workspace):
    client = TestClient(app)
    # 1. Valid read
    res = client.get("/api/coding/file?path=calculator.py")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "def add(a, b):" in data["file"]["content"]
    assert data["file"]["syntax"] == "python"

    # 2. Path traversal attack blocked
    res_escape = client.get("/api/coding/file?path=../../windows/win.ini")
    assert res_escape.status_code in (400, 403)


def test_file_save_with_ast_syntax_check(test_workspace):
    client = TestClient(app)

    # 1. Valid Python save
    valid_code = "def multiply(a, b):\n    return a * b\n"
    res = client.post("/api/coding/file/save", json={"path": "math_ops.py", "content": valid_code, "validate_syntax": True})
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    assert (test_workspace / "math_ops.py").exists()

    # 2. Invalid Python save (broken syntax)
    broken_code = "def broken(\n  syntax error here"
    res_err = client.post("/api/coding/file/save", json={"path": "broken.py", "content": broken_code, "validate_syntax": True})
    assert res_err.status_code == 200
    assert res_err.json()["status"] == "error"
    assert res_err.json()["result"]["syntax_valid"] is False
    assert not (test_workspace / "broken.py").exists()


def test_patch_preview_and_apply(test_workspace):
    client = TestClient(app)

    # Preview patch
    target = "return a - b"
    replacement = "return a - b  # verified"
    preview_res = client.post("/api/coding/patch/preview", json={
        "file_path": "calculator.py",
        "target": target,
        "replacement": replacement,
    })
    assert preview_res.status_code == 200
    patch_data = preview_res.json()["patch"]
    assert patch_data["valid"] is True
    assert "+    return a - b  # verified" in patch_data["diff"]

    # Apply patch
    apply_res = client.post("/api/coding/patch/apply", json={
        "file_path": "calculator.py",
        "target": target,
        "replacement": replacement,
    })
    assert apply_res.status_code == 200
    assert apply_res.json()["status"] == "ok"

    # Verify disk content updated
    updated_content = (test_workspace / "calculator.py").read_text(encoding="utf-8")
    assert "return a - b  # verified" in updated_content

    # Rollback
    undo_res = client.post("/api/coding/undo", json={"file_path": "calculator.py"})
    assert undo_res.status_code == 200
    # Note: undo restored to backup if available
    assert undo_res.json()["status"] == "ok"


def test_test_runner_endpoint(test_workspace):
    client = TestClient(app)
    res = client.post("/api/coding/test", json={"target": "test_calc.py"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    test_run = data["test_run"]
    assert test_run["passed"] == 2
    assert test_run["all_passed"] is True


def test_code_search_endpoint(test_workspace):
    client = TestClient(app)
    res = client.post("/api/coding/search", json={"query": "def add", "type": "text"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["count"] >= 1
    assert any("calculator.py" in m["file"] for m in data["matches"])


def test_coding_chat_endpoint_fast_mode(test_workspace):
    client = TestClient(app)
    res = client.post("/api/coding/chat", json={
        "message": "Explain the functions in calculator.py",
        "think_mode": "fast",
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    task_res = data["task_result"]
    assert task_res["think_mode"] == "fast"
    assert len(data["timeline"]) >= 1
