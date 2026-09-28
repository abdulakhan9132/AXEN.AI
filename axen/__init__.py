"""
axen/__init__.py
================
AXEN — Local AI Assistant, Phase 2.

Public surface area:
    from axen import AssistantController, HFLocalModel, HFModelConfig
    from axen.memory import ConversationMemory, LongTermMemory
    from axen.tools import ToolRegistry, build_default_registry
"""

from .assistant.controller import AssistantController
from .model.hf_model import HFLocalModel, HFModelConfig
from .memory.conversation_memory import ConversationMemory
from .memory.long_term_memory import LongTermMemory
from .tools import ToolRegistry, build_default_registry
from .agent import AgentState, AgentLoop, TaskClassifier, Planner, Verifier

__version__ = "3.0.0"
__all__ = [
    "AssistantController",
    "HFLocalModel",
    "HFModelConfig",
    "ConversationMemory",
    "LongTermMemory",
    "ToolRegistry",
    "build_default_registry",
    "AgentState",
    "AgentLoop",
    "TaskClassifier",
    "Planner",
    "Verifier",
]
