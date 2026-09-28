from __future__ import annotations

from app.agent.tools import ToolDefinition


def snapshot(args: dict) -> dict:
    url = args.get("url", "")
    return {
        "success": True,
        "url": url,
        "title": "Browser snapshot placeholder",
        "ready": True,
        "message": "Browser automation is planned for Phase 2 and is intentionally disabled in Phase 1.",
    }


def build_browser_tool() -> ToolDefinition:
    return ToolDefinition(
        name="browser.snapshot",
        description="Returns a placeholder snapshot for browser state until browser automation is enabled.",
        input_schema={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
        output_schema={"type": "object", "properties": {"title": {"type": "string"}, "ready": {"type": "boolean"}}},
        risk="medium",
        handler=snapshot,
    )
