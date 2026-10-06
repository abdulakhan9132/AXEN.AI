import threading
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional

class ContextPriority(IntEnum):
    SYSTEM = 1
    CURRENT_USER_MESSAGE = 2
    CONVERSATION_CONTEXT_RECENT = 3
    PROJECT_MEMORY = 4
    GLOBAL_MEMORY = 5
    PROJECT_CONTEXT = 6
    KNOWLEDGE = 7
    FILE = 8
    OLDER_CONVERSATION = 9
    TOOL_OBSERVATIONS = 10
    METADATA = 11

@dataclass
class ContextComponent:
    """A single piece of context evaluated by the Context Engine."""
    source: str
    content: str
    priority: ContextPriority
    budget_tokens: int = 0
    relevance_score: float = 1.0
    required: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def estimate_tokens(self) -> int:
        # A simple estimation: 1 token ~ 4 characters
        return max(1, len(self.content) // 4)

@dataclass
class ContextRequest:
    """The incoming request for context assembly."""
    system_instructions: List[ContextComponent] = field(default_factory=list)
    current_message: Optional[ContextComponent] = None
    conversation_history: List[ContextComponent] = field(default_factory=list)
    long_term_memories: List[ContextComponent] = field(default_factory=list)
    project_context: List[ContextComponent] = field(default_factory=list)
    retrieved_knowledge: List[ContextComponent] = field(default_factory=list)
    file_context: List[ContextComponent] = field(default_factory=list)
    tool_observations: List[ContextComponent] = field(default_factory=list)
    runtime_metadata: List[ContextComponent] = field(default_factory=list)
    constraints: List[ContextComponent] = field(default_factory=list)

@dataclass
class ModelCapability:
    """Interface for model capability configuration (PHASE 3C)."""
    context_length: int
    max_output_tokens: int
    tokenizer: str
    provider: str
    model_id: str

@dataclass
class AssembledContext:
    """The final structured context package consumed by the model."""
    system_instruction: str = ""
    messages_for_model: List[Dict[str, str]] = field(default_factory=list)
    summary_injected: bool = False
    tokens_used: int = 0
    
@dataclass
class ModelRequest:
    """The contract for all model providers (PHASE 3G)."""
    context: AssembledContext
    generation_settings: Dict[str, Any]
    session_metadata: Dict[str, Any]
    cancel_event: Optional[threading.Event] = None
