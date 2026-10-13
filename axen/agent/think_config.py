"""
axen/agent/think_config.py
===========================
Think Engine 2.0 — Configuration, Mode Definitions, and Budget Management.

Defines:
- ThinkMode enum (FAST, BALANCED, DEEP)
- ThinkBudget dataclass (token, time, iteration, tool-call limits)
- ThinkModeConfig per-mode behavior flags
- ThinkEngineConfig loaded from configs/think_engine.yaml
- ThinkModeSelector: maps (intent, complexity) → ThinkMode
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("axen.agent.think_config")

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ThinkMode(str, Enum):
    """Thinking depth modes for the reasoning engine."""
    FAST = "fast"
    BALANCED = "balanced"
    DEEP = "deep"

    @classmethod
    def from_string(cls, value: str) -> ThinkMode:
        """Case-insensitive lookup with fallback to BALANCED."""
        try:
            return cls(value.lower().strip())
        except (ValueError, AttributeError):
            return cls.BALANCED


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


@dataclass
class ThinkBudget:
    """Resource budget for a single request under a specific thinking mode."""
    max_tokens: int = 4096
    max_runtime_seconds: float = 45.0
    max_iterations: int = 4
    max_tool_calls: int = 4
    max_retries: int = 1

    def remaining_tokens(self, used: int) -> int:
        return max(0, self.max_tokens - used)

    def remaining_iterations(self, used: int) -> int:
        return max(0, self.max_iterations - used)

    def remaining_tool_calls(self, used: int) -> int:
        return max(0, self.max_tool_calls - used)

    def is_token_budget_exhausted(self, used: int) -> bool:
        return used >= self.max_tokens

    def is_time_budget_exhausted(self, elapsed: float) -> bool:
        return elapsed >= self.max_runtime_seconds

    def is_iteration_budget_exhausted(self, used: int) -> bool:
        return used >= self.max_iterations

    def is_tool_budget_exhausted(self, used: int) -> bool:
        return used >= self.max_tool_calls

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "max_runtime_seconds": self.max_runtime_seconds,
            "max_iterations": self.max_iterations,
            "max_tool_calls": self.max_tool_calls,
            "max_retries": self.max_retries,
        }


# ---------------------------------------------------------------------------
# Mode Configuration
# ---------------------------------------------------------------------------


@dataclass
class ThinkModeConfig:
    """Behavior configuration for a single thinking mode."""
    mode: ThinkMode
    description: str = ""
    budget: ThinkBudget = field(default_factory=ThinkBudget)
    skip_planning: bool = False
    skip_verification: bool = False
    skip_memory_retrieval: bool = False
    skip_context_assembly: bool = False
    allow_replan: bool = False
    max_replan_attempts: int = 0
    allow_tool_use: bool = True
    enable_subtask_decomposition: bool = False
    enable_claim_verification: bool = False
    complexity_match: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "description": self.description,
            "budget": self.budget.to_dict(),
            "skip_planning": self.skip_planning,
            "skip_verification": self.skip_verification,
            "skip_memory_retrieval": self.skip_memory_retrieval,
            "skip_context_assembly": self.skip_context_assembly,
            "allow_replan": self.allow_replan,
            "max_replan_attempts": self.max_replan_attempts,
            "allow_tool_use": self.allow_tool_use,
            "enable_subtask_decomposition": self.enable_subtask_decomposition,
            "enable_claim_verification": self.enable_claim_verification,
        }


# ---------------------------------------------------------------------------
# Default mode configs (used when YAML is missing)
# ---------------------------------------------------------------------------

_DEFAULT_FAST = ThinkModeConfig(
    mode=ThinkMode.FAST,
    description="Direct response with minimal overhead",
    budget=ThinkBudget(max_tokens=1024, max_runtime_seconds=10.0, max_iterations=1, max_tool_calls=1, max_retries=0),
    skip_planning=True,
    allow_replan=False,
    complexity_match=["LOW"],
)

_DEFAULT_BALANCED = ThinkModeConfig(
    mode=ThinkMode.BALANCED,
    description="Task analysis, limited planning, and basic verification",
    budget=ThinkBudget(max_tokens=4096, max_runtime_seconds=45.0, max_iterations=4, max_tool_calls=4, max_retries=1),
    allow_replan=True,
    max_replan_attempts=1,
    complexity_match=["MEDIUM"],
)

_DEFAULT_DEEP = ThinkModeConfig(
    mode=ThinkMode.DEEP,
    description="Multi-stage planning, retrieval, execution, verification, and correction",
    budget=ThinkBudget(max_tokens=16384, max_runtime_seconds=120.0, max_iterations=8, max_tool_calls=8, max_retries=3),
    allow_replan=True,
    max_replan_attempts=3,
    enable_subtask_decomposition=True,
    enable_claim_verification=True,
    complexity_match=["HIGH"],
)


# ---------------------------------------------------------------------------
# Engine Config (loaded from YAML)
# ---------------------------------------------------------------------------


@dataclass
class ThinkEngineConfig:
    """Top-level Think Engine configuration."""
    enabled: bool = True
    auto_select: bool = True
    default_mode: ThinkMode = ThinkMode.BALANCED
    modes: Dict[ThinkMode, ThinkModeConfig] = field(default_factory=dict)
    intent_overrides: Dict[str, ThinkMode] = field(default_factory=dict)
    global_max_tokens: int = 16384
    global_max_runtime_seconds: float = 120.0
    global_max_iterations: int = 12
    global_max_tool_calls: int = 10
    global_max_retries: int = 3
    observability_enabled: bool = True
    redact_fields: List[str] = field(default_factory=lambda: ["api_key", "password", "token", "secret"])

    def __post_init__(self) -> None:
        if not self.modes:
            self.modes = {
                ThinkMode.FAST: _DEFAULT_FAST,
                ThinkMode.BALANCED: _DEFAULT_BALANCED,
                ThinkMode.DEEP: _DEFAULT_DEEP,
            }

    def get_mode_config(self, mode: ThinkMode) -> ThinkModeConfig:
        """Get the configuration for a specific mode, with fallback."""
        return self.modes.get(mode, self.modes.get(ThinkMode.BALANCED, _DEFAULT_BALANCED))

    @classmethod
    def from_yaml(cls, path: Optional[str] = None) -> ThinkEngineConfig:
        """Load configuration from YAML file."""
        yaml_path = Path(path) if path else Path("configs/think_engine.yaml")
        if not yaml_path.exists():
            logger.info("Think Engine config not found at %s, using defaults.", yaml_path)
            return cls()

        try:
            import yaml
            raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load Think Engine config from %s: %s", yaml_path, exc)
            return cls()

        if not raw:
            return cls()

        te = raw.get("think_engine", {})
        modes_raw = raw.get("modes", {})
        intent_raw = raw.get("intent_overrides", {})
        obs_raw = raw.get("observability", {})
        glim = te.get("global_limits", {})

        # Parse modes
        modes: Dict[ThinkMode, ThinkModeConfig] = {}
        for mode_key, mode_data in modes_raw.items():
            try:
                tm = ThinkMode.from_string(mode_key)
                budget = ThinkBudget(
                    max_tokens=mode_data.get("max_tokens", 4096),
                    max_runtime_seconds=float(mode_data.get("max_runtime_seconds", 45.0)),
                    max_iterations=mode_data.get("max_iterations", 4),
                    max_tool_calls=mode_data.get("max_tool_calls", 4),
                    max_retries=mode_data.get("max_retries", 1),
                )
                mc = ThinkModeConfig(
                    mode=tm,
                    description=mode_data.get("description", ""),
                    budget=budget,
                    skip_planning=mode_data.get("skip_planning", False),
                    skip_verification=mode_data.get("skip_verification", False),
                    skip_memory_retrieval=mode_data.get("skip_memory_retrieval", False),
                    skip_context_assembly=mode_data.get("skip_context_assembly", False),
                    allow_replan=mode_data.get("allow_replan", False),
                    max_replan_attempts=mode_data.get("max_replan_attempts", 0),
                    allow_tool_use=mode_data.get("allow_tool_use", True),
                    enable_subtask_decomposition=mode_data.get("enable_subtask_decomposition", False),
                    enable_claim_verification=mode_data.get("enable_claim_verification", False),
                    complexity_match=mode_data.get("complexity_match", []),
                )
                modes[tm] = mc
            except Exception as exc:
                logger.warning("Failed to parse mode '%s': %s", mode_key, exc)

        # Parse intent overrides
        intent_overrides: Dict[str, ThinkMode] = {}
        for intent_key, mode_str in intent_raw.items():
            intent_overrides[intent_key] = ThinkMode.from_string(mode_str)

        return cls(
            enabled=te.get("enabled", True),
            auto_select=te.get("auto_select", True),
            default_mode=ThinkMode.from_string(te.get("default_mode", "balanced")),
            modes=modes if modes else {ThinkMode.FAST: _DEFAULT_FAST, ThinkMode.BALANCED: _DEFAULT_BALANCED, ThinkMode.DEEP: _DEFAULT_DEEP},
            intent_overrides=intent_overrides,
            global_max_tokens=glim.get("max_total_tokens", 16384),
            global_max_runtime_seconds=float(glim.get("max_runtime_seconds", 120.0)),
            global_max_iterations=glim.get("max_iterations", 12),
            global_max_tool_calls=glim.get("max_tool_calls", 10),
            global_max_retries=glim.get("max_retries", 3),
            observability_enabled=obs_raw.get("enabled", True),
            redact_fields=obs_raw.get("redact_fields", ["api_key", "password", "token", "secret"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "auto_select": self.auto_select,
            "default_mode": self.default_mode.value,
            "modes": {m.value: c.to_dict() for m, c in self.modes.items()},
            "intent_overrides": {k: v.value for k, v in self.intent_overrides.items()},
        }


# ---------------------------------------------------------------------------
# Mode Selector
# ---------------------------------------------------------------------------


class ThinkModeSelector:
    """
    Selects the appropriate thinking mode based on task classification.

    Priority order:
    1. Explicit override (user or API request)
    2. Intent-based override (from config)
    3. Complexity-based match (from config)
    4. Default mode
    """

    def __init__(self, config: Optional[ThinkEngineConfig] = None) -> None:
        self.config = config or ThinkEngineConfig()

    def select(
        self,
        query_or_intent: Optional[str] = None,
        complexity: Optional[str] = None,
        *,
        user_request: Optional[str] = None,
        intent_info: Optional[Any] = None,
        override: Optional[Any] = None,
        requires_tool: bool = False,
        requires_multiple_steps: bool = False,
    ) -> Tuple[ThinkMode, str]:
        """
        Select the thinking mode for a request.

        Returns:
            Tuple of (ThinkMode, rationale_string)
        """
        import re

        # Determine user query string
        req_text = user_request or (query_or_intent if intent_info is not None else "")
        req_lower = (req_text or "").lower()

        # Determine classified intent & complexity
        intent_str = "conversation"
        complexity_str = "LOW"
        multi_step = requires_multiple_steps

        if intent_info is not None:
            intent_str = getattr(intent_info, "intent", "conversation")
            complexity_str = getattr(intent_info, "complexity", "LOW")
        elif query_or_intent is not None and complexity is not None:
            intent_str = query_or_intent
            complexity_str = complexity
        elif query_or_intent is not None:
            intent_str = query_or_intent

        # Multi-step heuristics from prompt if text provided
        if req_text and not multi_step:
            if re.search(r"\b(?:and\s+then|then)\b", req_lower):
                multi_step = True
            elif re.search(r"\b(?:find|search|read)\b.*?\b(?:calculate|compute|solve|average|sum)\b", req_lower):
                multi_step = True
            elif req_lower.count(" and ") >= 2 or ";" in req_text or (req_lower.count(",") >= 2 and " and " in req_lower):
                multi_step = True


        # 1. Explicit override
        if override:
            chosen = ThinkMode.from_string(str(override))
            return chosen, f"Explicit override to {chosen.value.upper()}"

        if not self.config.auto_select:
            return self.config.default_mode, "Default mode (auto_select disabled)"

        # 2. Intent-based override
        if intent_str in self.config.intent_overrides:
            chosen = self.config.intent_overrides[intent_str]
            return chosen, f"Intent override for '{intent_str}' -> {chosen.value.upper()}"

        # 3. Multi-step tasks go DEEP
        if multi_step:
            return ThinkMode.DEEP, "Multi-step complex workflow requires DEEP mode"

        # 4. Complexity-based match
        comp_upper = str(complexity_str).upper()
        for mode, mode_config in self.config.modes.items():
            if comp_upper in [c.upper() for c in mode_config.complexity_match]:
                return mode, f"Complexity {comp_upper} matched {mode.value.upper()} mode"

        # 5. Default fallback
        return self.config.default_mode, f"Default fallback mode ({self.config.default_mode.value.upper()})"


    def get_budget(self, mode: ThinkMode) -> ThinkBudget:
        """Get the budget for a specific mode."""
        return self.config.get_mode_config(mode).budget

    def get_mode_config(self, mode: ThinkMode) -> ThinkModeConfig:
        """Get the full configuration for a specific mode."""
        return self.config.get_mode_config(mode)
