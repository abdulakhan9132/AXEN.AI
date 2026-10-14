"""
tests/test_existing_regression.py
=================================
Regression tests verifying existing core systems remain operational and unchanged:
- ToolRegistry lifecycle
- Verifier tool result verification
- MemoryStore SQLite compatibility
"""

import pytest
from axen.agent.verifier import Verifier
from axen.memory.memory_models import MemoryItem, MemoryType
from axen.memory.memory_store import SQLiteMemoryStore
from axen.tools.tool_registry import ToolRegistry


def test_existing_tool_registry_intact():
    registry = ToolRegistry()
    tools = registry.list_tools()
    assert isinstance(tools, list)
    names = registry.names()
    assert isinstance(names, list)


def test_existing_verifier_intact():
    verifier = Verifier()
    # Test arithmetic tool result verification
    result = verifier.verify_tool_result(
        tool_name="calculator",
        arguments={"expression": "2 + 2"},
        result={"success": True, "result": 4, "display": 4},
    )
    assert result["passed"] is True
    assert result["domain"] == "arithmetic"


def test_existing_memory_store_intact():
    import shutil
    import tempfile
    from pathlib import Path

    tmp_dir = Path(tempfile.mkdtemp(prefix="axen_reg_test_"))
    try:
        db_file = tmp_dir / "test_memory.db"
        mem_store = SQLiteMemoryStore(db_path=db_file)
        item = MemoryItem(
            key="user_name",
            value="Alice",
            memory_type=MemoryType.FACT,
        )
        mem_store.add(item)
        fetched = mem_store.get_by_key("user_name")
        assert fetched is not None
        assert fetched.value == "Alice"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
