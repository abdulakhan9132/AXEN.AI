"""
axen/agent/tool_intent.py
=========================
Canonical Structured Tool Intent Schema, Validator, and Brain Decision Interface.

This module provides:
1. Canonical ToolIntent schema and BrainDecision representation.
2. ToolIntentParser for parsing structured model outputs (JSON, markdown blocks).
3. ToolIntentValidator enforcing schema, permissions, parameter types, sandbox boundaries,
   and resource limits before tool invocation.
4. BrainDecisionAdapter bridging AXEN Foundation Model, native tool intent, and fallback planner.
5. Strict protection preventing private chain-of-thought exposure.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ToolIntentType(str, Enum):
    """Canonical types of intent produced by AXEN brain."""
    TOOL_CALL = "tool_call"
    MULTIPLE_TOOL_CALLS = "multiple_tool_calls"
    FINAL_RESPONSE = "final_response"
    CLARIFICATION = "clarification"
    REFLECTION = "reflection"
    ABORT = "abort"


class DecisionSource(str, Enum):
    """Tracks the origin of the agent decision for observability and metrics."""
    BRAIN_NATIVE = "brain_native"              # Native structured call from model
    BRAIN_STRUCTURED = "brain_structured"      # Parsed from structured text block
    DETERMINISTIC_RULE = "deterministic_rule"  # Deterministic intent match
    PLANNER_FALLBACK = "planner_fallback"      # Existing planner fallback
    SPECIALIST_QWEN = "specialist_qwen"        # Qwen coding specialist only


# ---------------------------------------------------------------------------
# Canonical Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ToolIntent:
    """
    Canonical representation of structured tool intent.
    NOTE: The 'reason' field must contain only a concise operational summary,
    NEVER private chain-of-thought.
    """
    type: ToolIntentType = ToolIntentType.TOOL_CALL
    tool: Optional[str] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    confidence: float = 1.0
    calls: List[ToolIntent] = field(default_factory=list)
    execution_id: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.type, str):
            try:
                self.type = ToolIntentType(self.type)
            except ValueError:
                self.type = ToolIntentType.TOOL_CALL

        # Sanitize reason field against private chain-of-thought leakage
        if self.reason:
            self.reason = ToolIntent.sanitize_reason(self.reason)

        if not self.execution_id and self.type == ToolIntentType.TOOL_CALL:
            self.execution_id = f"exec_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def sanitize_reason(reason_text: str) -> str:
        """Strip internal reasoning or multi-step thought prefixes."""
        cleaned = reason_text.strip()
        # Disallow internal monologue patterns
        patterns = [
            r"^(?:thinking|thought|internal monologue|i think|let me think|first i consider)[\s\:\-]+",
            r"(?:hidden reasoning|chain of thought)[\s\:\-]*",
        ]
        for p in patterns:
            cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE).strip()
        # Keep operational reason concise (max 200 chars)
        return cleaned[:200]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable representation."""
        data: Dict[str, Any] = {
            "type": self.type.value if hasattr(self.type, "value") else str(self.type),
            "tool": self.tool,
            "arguments": self.arguments,
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
        }
        if self.execution_id:
            data["execution_id"] = self.execution_id
        if self.calls:
            data["calls"] = [c.to_dict() for c in self.calls]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ToolIntent:
        """Construct ToolIntent from dictionary."""
        intent_type_str = data.get("type", "tool_call")
        try:
            intent_type = ToolIntentType(intent_type_str)
        except ValueError:
            intent_type = ToolIntentType.TOOL_CALL

        calls = []
        if "calls" in data and isinstance(data["calls"], list):
            calls = [cls.from_dict(c) for c in data["calls"] if isinstance(c, dict)]

        return cls(
            type=intent_type,
            tool=data.get("tool"),
            arguments=data.get("arguments") or {},
            reason=str(data.get("reason") or ""),
            confidence=float(data.get("confidence", 1.0)),
            calls=calls,
            execution_id=data.get("execution_id"),
        )


@dataclass
class BrainDecision:
    """
    Unified decision emitted by the AXEN Brain or its structured adapter.
    """
    decision_type: ToolIntentType
    content: Optional[str] = None
    tool_intent: Optional[ToolIntent] = None
    confidence: float = 1.0
    decision_source: DecisionSource = DecisionSource.BRAIN_NATIVE
    operational_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_type": self.decision_type.value,
            "content": self.content,
            "tool_intent": self.tool_intent.to_dict() if self.tool_intent else None,
            "confidence": round(self.confidence, 4),
            "decision_source": self.decision_source.value,
            "operational_reason": self.operational_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BrainDecision:
        dtype = ToolIntentType(data.get("decision_type", "final_response"))
        dsource = DecisionSource(data.get("decision_source", "brain_native"))
        t_intent = ToolIntent.from_dict(data["tool_intent"]) if data.get("tool_intent") else None
        return cls(
            decision_type=dtype,
            content=data.get("content"),
            tool_intent=t_intent,
            confidence=float(data.get("confidence", 1.0)),
            decision_source=dsource,
            operational_reason=str(data.get("operational_reason") or ""),
        )


# ---------------------------------------------------------------------------
# Tool Intent Parser
# ---------------------------------------------------------------------------

class ToolIntentParser:
    """
    Extracts structured ToolIntent from model output safely.
    Handles:
    1. Valid JSON objects directly
    2. JSON inside markdown code blocks (```json ... ```)
    3. Custom tool call tags (<tool_call>...</tool_call>)
    4. Text with implicit final response
    """

    _JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
    _TOOL_TAG_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL | re.IGNORECASE)

    @classmethod
    def parse(cls, raw_output: str) -> BrainDecision:
        """Parse raw model string into BrainDecision."""
        text = (raw_output or "").strip()
        if not text:
            return BrainDecision(
                decision_type=ToolIntentType.FINAL_RESPONSE,
                content="",
                confidence=1.0,
                decision_source=DecisionSource.BRAIN_NATIVE,
            )

        # 1. Attempt extraction from ```json ... ``` code fence
        match = cls._JSON_BLOCK_PATTERN.search(text)
        if match:
            parsed = cls._try_parse_json(match.group(1))
            if parsed:
                intent = ToolIntent.from_dict(parsed)
                return BrainDecision(
                    decision_type=intent.type,
                    tool_intent=intent,
                    content=text,
                    confidence=intent.confidence,
                    decision_source=DecisionSource.BRAIN_STRUCTURED,
                    operational_reason=intent.reason,
                )

        # 2. Attempt extraction from <tool_call>...</tool_call> tags
        match_tag = cls._TOOL_TAG_PATTERN.search(text)
        if match_tag:
            parsed = cls._try_parse_json(match_tag.group(1))
            if parsed:
                intent = ToolIntent.from_dict(parsed)
                return BrainDecision(
                    decision_type=intent.type,
                    tool_intent=intent,
                    content=text,
                    confidence=intent.confidence,
                    decision_source=DecisionSource.BRAIN_STRUCTURED,
                    operational_reason=intent.reason,
                )

        # 3. Direct JSON object string check
        if text.startswith("{") and text.endswith("}"):
            parsed = cls._try_parse_json(text)
            if parsed and "type" in parsed:
                intent = ToolIntent.from_dict(parsed)
                return BrainDecision(
                    decision_type=intent.type,
                    tool_intent=intent,
                    content=text,
                    confidence=intent.confidence,
                    decision_source=DecisionSource.BRAIN_NATIVE,
                    operational_reason=intent.reason,
                )

        # 4. Fallback: standard conversational response
        return BrainDecision(
            decision_type=ToolIntentType.FINAL_RESPONSE,
            content=text,
            confidence=0.9,
            decision_source=DecisionSource.BRAIN_NATIVE,
        )

    @staticmethod
    def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
        try:
            val = json.loads(text)
            if isinstance(val, dict):
                return val
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# Tool Intent Validator
# ---------------------------------------------------------------------------

class ToolIntentValidator:
    """
    Enforces strict pre-execution validation on model tool requests.
    Validates:
    - Schema conformity
    - Tool existence in registry
    - Argument types and required parameters
    - Permissions and Project boundaries
    - Payload size bounds
    - Absence of leaked private thoughts in operational reasons
    """

    MAX_PAYLOAD_BYTES = 500_000  # 500 KB limit for input payload

    @classmethod
    def validate(
        cls,
        intent: ToolIntent,
        registry: Any,
        context: Optional[Any] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate ToolIntent before execution.
        Returns (is_valid, error_reason).
        """
        if not isinstance(intent, ToolIntent):
            return False, f"Expected ToolIntent object, got {type(intent).__name__}."

        if intent.type == ToolIntentType.FINAL_RESPONSE:
            return True, None

        if intent.type == ToolIntentType.CLARIFICATION:
            return True, None

        if intent.type == ToolIntentType.ABORT:
            return True, None

        if intent.type == ToolIntentType.MULTIPLE_TOOL_CALLS:
            if not intent.calls:
                return False, "Multiple tool calls intent contains empty calls list."
            for idx, sub_call in enumerate(intent.calls, 1):
                ok, err = cls.validate(sub_call, registry, context=context)
                if not ok:
                    return False, f"Sub-call #{idx} validation failed: {err}"
            return True, None

        # Singular tool call
        if intent.type == ToolIntentType.TOOL_CALL:
            if not intent.tool:
                return False, "Tool intent missing 'tool' name."

            # Check registry existence
            if registry is None:
                return False, "Tool registry unavailable for validation."

            tool_def = registry.get(intent.tool) if hasattr(registry, "get") else None
            if tool_def is None:
                return False, f"Unknown tool: '{intent.tool}' not found in registry."

            if hasattr(tool_def, "enabled") and not tool_def.enabled:
                return False, f"Tool '{intent.tool}' is disabled."

            # Check arguments format
            if not isinstance(intent.arguments, dict):
                return False, f"Tool arguments must be a dictionary, got {type(intent.arguments).__name__}."

            # Check size limit
            try:
                raw_bytes = len(json.dumps(intent.arguments, default=str).encode("utf-8"))
                if raw_bytes > cls.MAX_PAYLOAD_BYTES:
                    return False, f"Arguments payload exceeds {cls.MAX_PAYLOAD_BYTES} bytes limit ({raw_bytes} bytes)."
            except Exception:
                pass

            # Delegate to ToolVerifier for schema and permission checks
            from axen.tools.verifier import ToolVerifier
            is_valid, v_err = ToolVerifier.validate_tool_request(
                tool=tool_def,
                tool_name=intent.tool,
                arguments=intent.arguments,
                context=context,
            )
            if not is_valid:
                return False, v_err

        return True, None


# ---------------------------------------------------------------------------
# Brain Decision Adapter
# ---------------------------------------------------------------------------

class BrainDecisionAdapter:
    """
    Coordinates decision making between the AXEN Foundation Model,
    structured intent validation, and fallback mechanisms.
    Tracks decision source metrics across runs.
    """

    def __init__(self, registry: Optional[Any] = None) -> None:
        self.registry = registry
        self._metrics: Dict[str, int] = {
            DecisionSource.BRAIN_NATIVE.value: 0,
            DecisionSource.BRAIN_STRUCTURED.value: 0,
            DecisionSource.DETERMINISTIC_RULE.value: 0,
            DecisionSource.PLANNER_FALLBACK.value: 0,
            DecisionSource.SPECIALIST_QWEN.value: 0,
        }

    @property
    def metrics(self) -> Dict[str, int]:
        return dict(self._metrics)

    def record_decision(self, source: DecisionSource) -> None:
        """Record decision source metric."""
        key = source.value if hasattr(source, "value") else str(source)
        self._metrics[key] = self._metrics.get(key, 0) + 1

    def decide(
        self,
        user_request: str,
        model: Optional[Any] = None,
        context: Optional[Any] = None,
        prior_observations: Optional[List[Dict[str, Any]]] = None,
    ) -> BrainDecision:
        """
        Primary decision workflow:
        1. Query Foundation Model if available for native/structured tool intent.
        2. Parse structured intent via ToolIntentParser.
        3. Validate intent via ToolIntentValidator.
        4. If valid, return decision.
        5. If model output is invalid or model unavailable, cleanly return planner fallback.
        """
        # If model is loaded, attempt native structured decision
        if model is not None and getattr(model, "is_loaded", False):
            try:
                system_prompt = (
                    "You are AXEN. To use a tool, output a JSON object with:\n"
                    '{"type": "tool_call", "tool": "<name>", "arguments": {...}, "reason": "<operational reason>"}\n'
                    'Or for a direct answer, output {"type": "final_response"}.'
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_request},
                ]
                # Invoke model chat/generation
                if hasattr(model, "chat"):
                    raw_out = model.chat(messages, max_tokens=150)
                elif hasattr(model, "generate"):
                    raw_out = model.generate(user_request, max_new_tokens=150)
                else:
                    raw_out = ""

                decision = ToolIntentParser.parse(raw_out)
                if decision.tool_intent:
                    is_valid, err = ToolIntentValidator.validate(
                        decision.tool_intent,
                        registry=self.registry,
                        context=context,
                    )
                    if is_valid:
                        self.record_decision(decision.decision_source)
                        return decision
                    else:
                        logger.warning("Native tool intent invalid: %s. Falling back to planner.", err)
            except Exception as exc:
                logger.debug("Native brain generation error: %s", exc)

        # Fallback to planner
        self.record_decision(DecisionSource.PLANNER_FALLBACK)
        return BrainDecision(
            decision_type=ToolIntentType.FINAL_RESPONSE,
            content=None,
            confidence=0.8,
            decision_source=DecisionSource.PLANNER_FALLBACK,
            operational_reason="Fallback to deterministic planner",
        )
