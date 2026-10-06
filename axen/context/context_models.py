"""
axen/context/context_models.py
==============================
Data models and schemas for AXEN Context Engine.

Defines:
- ContextPriority: Ordered hierarchy from System constraints to External Web data
- ContextItem: Individual typed contextual element
- ContextRequest: Input parameters specifying request and query bounds
- AssembledContext: Final structured result containing isolated sections and debug metrics
- ContextDebugInfo: Structured observability record
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional


class ContextPriority(IntEnum):
    """
    Context Priority Hierarchy:
    1. System/application constraints
    2. Current user message
    3. Current task state
    4. Project instructions
    5. Relevant conversation context
    6. Relevant project memory
    7. Relevant global memory
    8. Relevant files (DATA)
    9. Relevant knowledge (DATA)
    10. External web evidence (DATA)
    """
    SYSTEM = 1
    CURRENT_USER_MESSAGE = 2
    TASK_STATE = 3
    PROJECT_INSTRUCTIONS = 4
    CONVERSATION_CONTEXT = 5
    PROJECT_MEMORY = 6
    GLOBAL_MEMORY = 7
    FILES = 8
    KNOWLEDGE = 9
    WEB_EVIDENCE = 10


@dataclass
class ContextItem:
    """A single piece of context evaluated by the Context Engine."""
    source_type: str  # "system" | "project_instruction" | "conversation" | "project_memory" | "global_memory" | "file" | "knowledge" | "web"
    content: str
    priority: int = ContextPriority.GLOBAL_MEMORY
    is_trusted: bool = True  # True for system, user prompt, project instructions; False for files, knowledge, web
    identifier: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    estimated_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.estimated_tokens and self.content:
            self.estimated_tokens = max(1, len(self.content) // 4)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContextRequest:
    """Specification of what context is needed for a turn."""
    user_request: str
    task_type: str = "conversation"
    conversation_id: Optional[str] = None
    project_id: Optional[str] = None
    attachments: List[str] = field(default_factory=list)
    user_id: str = "default_user"
    max_tokens: int = 2000
    allow_web: bool = False
    debug: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContextDebugInfo:
    """Debug metrics and candidates for transparency."""
    request_id: str = ""
    query: str = ""
    task_type: str = ""
    project_id: Optional[str] = None
    is_fast_path: bool = False
    retrieval_reasons: List[str] = field(default_factory=list)
    retrieved_memory_count: int = 0
    retrieved_memories: List[Dict[str, Any]] = field(default_factory=list)
    project_memory_count: int = 0
    global_memory_count: int = 0
    files_included_count: int = 0
    knowledge_chunks_count: int = 0
    web_evidence_count: int = 0
    total_tokens_estimated: int = 0
    timings: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def format_debug_block(self) -> str:
        """Format human-readable debug output."""
        lines = [
            "=== CONTEXT ENGINE DEBUG ===",
            f"Query: {self.query[:80]}",
            f"Task Type: {self.task_type}",
            f"Fast-Path Activated: {self.is_fast_path}",
            f"Project ID: {self.project_id or 'None'}",
            f"Global Memories: {self.global_memory_count}",
            f"Project Memories: {self.project_memory_count}",
            f"Files Count: {self.files_included_count}",
            f"Knowledge Chunks: {self.knowledge_chunks_count}",
            f"Web Evidence: {self.web_evidence_count}",
            f"Estimated Tokens: {self.total_tokens_estimated}",
        ]
        if self.retrieved_memories:
            lines.append("Selected Memories:")
            for m in self.retrieved_memories:
                lines.append(f"  - [{m.get('scope', 'global')}] {m.get('key')}: {m.get('value')[:60]} (score: {m.get('score', 0):.2f})")
        if self.timings:
            time_strs = [f"{k}={v*1000:.1f}ms" for k, v in self.timings.items()]
            lines.append(f"Timings: {', '.join(time_strs)}")
        lines.append("=== END CONTEXT ENGINE DEBUG ===")
        return "\n".join(lines)


@dataclass
class AssembledContext:
    """The final assembled context package consumed by AgentLoop and model."""
    system_instruction: str = ""
    project_instructions: str = ""
    current_task: str = ""
    conversation_summary: str = ""
    recent_messages: List[Dict[str, str]] = field(default_factory=list)
    project_memories: List[Dict[str, Any]] = field(default_factory=list)
    global_memories: List[Dict[str, Any]] = field(default_factory=list)
    file_contexts: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_contexts: List[Dict[str, Any]] = field(default_factory=list)
    web_contexts: List[Dict[str, Any]] = field(default_factory=list)
    formatted_prompt: str = ""
    messages_for_model: List[Dict[str, str]] = field(default_factory=list)
    debug_info: Optional[ContextDebugInfo] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.debug_info:
            d["debug_info"] = self.debug_info.to_dict()
        return d
