"""
axen/agent/__init__.py
======================
Phase 3 Agentic Reasoning Loop package.
"""

from .state import AgentState
from .classifier import TaskClassifier
from .intent import IntentInfo
from .context_manager import ContextManager
from .planner import Planner
from .verifier import Verifier, VERIFIED, FAILED, UNCERTAIN, NOT_APPLICABLE
from .loop import AgentLoop, MAX_STEPS, MAX_TOOL_ATTEMPTS
from .think_config import (
    ThinkMode,
    ThinkBudget,
    ThinkModeConfig,
    ThinkEngineConfig,
    ThinkModeSelector,
)
from .think_metrics import (
    ThinkRequestMetrics,
    ToolCallMetric,
    PhaseMetric,
    ThinkLogger,
)
from .think_engine import (
    ThinkEngine,
    Subtask,
    DecomposedObjective,
    SubtaskDecomposer,
    ClaimVerificationReport,
)

__all__ = [
    "AgentState",
    "TaskClassifier",
    "IntentInfo",
    "ContextManager",
    "Planner",
    "Verifier",
    "VERIFIED",
    "FAILED",
    "UNCERTAIN",
    "NOT_APPLICABLE",
    "AgentLoop",
    "MAX_STEPS",
    "MAX_TOOL_ATTEMPTS",
    "ThinkMode",
    "ThinkBudget",
    "ThinkModeConfig",
    "ThinkEngineConfig",
    "ThinkModeSelector",
    "ThinkRequestMetrics",
    "ToolCallMetric",
    "PhaseMetric",
    "ThinkLogger",
    "ThinkEngine",
    "Subtask",
    "DecomposedObjective",
    "SubtaskDecomposer",
    "ClaimVerificationReport",
]

