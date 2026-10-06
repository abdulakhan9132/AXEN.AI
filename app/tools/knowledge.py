from __future__ import annotations
import sys
from pathlib import Path
from app.agent.tools import ToolDefinition

# Ensure knowledge module can be imported
KNOWLEDGE_DIR = Path(__file__).parent.parent.parent / "knowledge"
if str(KNOWLEDGE_DIR) not in sys.path:
    sys.path.insert(0, str(KNOWLEDGE_DIR.parent))

def search_knowledge_tool(args: dict) -> dict:
    query = args.get("query", "")
    top_k = args.get("top_k", 5)
    
    try:
        from knowledge.retrieve import search_knowledge
        results = search_knowledge(query, top_k)
        return {"success": True, "results": results}
    except Exception as e:
        return {"success": False, "error": str(e)}

def build_knowledge_tool() -> ToolDefinition:
    return ToolDefinition(
        name="search_knowledge",
        description="Search the local codebase and documents via semantic search to find relevant context.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query or concept to find."},
                "top_k": {"type": "integer", "description": "Number of results to return (default 5)."}
            },
            "required": ["query"]
        },
        output_schema={"type": "object", "properties": {"results": {"type": "array"}}, "additionalProperties": True},
        risk="low",
        handler=search_knowledge_tool,
    )
