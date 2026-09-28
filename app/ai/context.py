from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ContextBundle:
    recent_context: List[str] = field(default_factory=list)
    relevant_memory: List[str] = field(default_factory=list)
    current_task: Optional[str] = None
    current_plan: Optional[Dict[str, Any]] = None
    tool_results: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)


class ContextManager:
    """Manage context windows without dumping the entire session into the model."""

    def __init__(self, max_recent: int = 8, max_memory: int = 6, max_tool_results: int = 6):
        self.max_recent = max_recent
        self.max_memory = max_memory
        self.max_tool_results = max_tool_results

    def build_context(
        self,
        *,
        task: Optional[str],
        plan: Optional[Dict[str, Any]],
        memory: List[str],
        tool_results: List[str],
        constraints: Optional[List[str]] = None,
    ) -> ContextBundle:
        bundle = ContextBundle(
            recent_context=self._truncate(task, self.max_recent),
            relevant_memory=self._truncate(memory, self.max_memory),
            current_task=task,
            current_plan=plan,
            tool_results=self._truncate(tool_results, self.max_tool_results),
            constraints=list(constraints or []),
        )
        return bundle

    def summarize_older_context(self, older_messages: List[str], keep: int = 3) -> List[str]:
        if len(older_messages) <= keep:
            return older_messages
        return older_messages[-keep:]

    @staticmethod
    def _truncate(items: List[str], limit: int) -> List[str]:
        if not items:
            return []
        return items[-limit:]

    def to_dict(self, bundle: ContextBundle) -> Dict[str, Any]:
        return {
            "recent_context": bundle.recent_context,
            "relevant_memory": bundle.relevant_memory,
            "current_task": bundle.current_task,
            "current_plan": bundle.current_plan,
            "tool_results": bundle.tool_results,
            "constraints": bundle.constraints,
        }
