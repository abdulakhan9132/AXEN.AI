from __future__ import annotations

from app.agent.memory import MemoryStore
from app.agent.tools import ToolDefinition


def read_memory(args: dict) -> dict:
    query = args.get("query", "")
    category = args.get("category")
    store = MemoryStore()
    if query:
        matches = store.search(query, category=category, limit=10)
    elif category:
        matches = store.category(category, limit=10)
    else:
        matches = store.list_recent(limit=10)
    return {"success": True, "items": matches}


def build_memory_tool() -> ToolDefinition:
    return ToolDefinition(
        name="memory.read",
        description="Read the agent's persistent memory store.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}, "category": {"type": "string"}},
        },
        output_schema={"type": "object", "properties": {"items": {"type": "array"}}, "additionalProperties": True},
        risk="low",
        handler=read_memory,
    )
