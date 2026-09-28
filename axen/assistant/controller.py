"""
axen/assistant/controller.py
=============================
The main AXEN assistant controller — Phase 2 Local Claude-like AI.

This is the brain that ties everything together:
  1. Memory  — ConversationMemory (short-term) + LongTermMemory (facts)
  2. Model   — HFLocalModel (local PyTorch/Transformers inference)
  3. Tools   — ToolRegistry (calculator, python executor, etc.)
  4. Router  — decides when to use a tool vs. answer directly

Design principles:
  - NEVER fabricate tool results. Tools return real output.
  - NEVER pretend the model ran when it didn't load.
  - NEVER expose internal system prompts to the user.
  - Short-circuit to tools for deterministic tasks (arithmetic, code).
  - Use the language model only for creative/conversational responses.

The controller is MODEL-AGNOSTIC at the interface level:
  If the HF model is not loaded, it falls back gracefully with a clear message.
  The Ollama backend (used by math/code reasoning) is entirely separate.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from axen.memory.conversation_memory import ConversationMemory
from axen.memory.long_term_memory import LongTermMemory
from axen.model.hf_model import HFLocalModel, HFModelConfig
from axen.tools import ToolRegistry, build_default_registry
from axen.agent import AgentLoop, AgentState

logger = logging.getLogger("axen.assistant")


# -----------------------------------------------------------------------
# Tool-trigger patterns (simple keyword-based dispatch)
# These trigger tool use WITHOUT requiring the LLM to decide.
# -----------------------------------------------------------------------

# Arithmetic: "what is 1234 * 567?", "what is 25 multiplied by 48?", "calculate (3+5)**2"
_CALC_PATTERN = re.compile(
    r"\b(calculate|compute|what\s+is|eval)\b.*?[\d\+\-\*\/\(\)\^\.]+|"
    r"\b[\d\.]+\s*(?:multiplied\s+by|times|divided\s+by|plus|minus|\*|\+|\/|\-|\*\*)\s*[\d\.]+|"
    r"^\s*[\d\+\-\*\/\(\)\^\.% \t]+\s*$",
    re.IGNORECASE | re.DOTALL,
)

# Symbolic math / advanced math: derivative, integral, gcd, primes
_MATH_SOLVER_PATTERN = re.compile(
    r"\b(derivative\s+of|integral\s+of|limit\s+of|gcd\s+of|greatest\s+common\s+divisor|is\s+\d+\s+prime|euler\s+phi|modular\s+inverse)\b|"
    r"^\s*solve\s+[a-z0-9\^\+\-\*\/\s=]+$",
    re.IGNORECASE,
)

# File read: "read this file: <path>", "read file <path>"
_FILE_PATTERN = re.compile(
    r"\b(?:read|show|view|open)\s+(?:this\s+)?file[:\s]+([^\s,;]+)",
    re.IGNORECASE,
)

# Python execution: "run this code", "execute python"
_PYTHON_PATTERN = re.compile(
    r"\b(run|execute|evaluate)\b.*\bpython\b|"
    r"```python",
    re.IGNORECASE | re.DOTALL,
)

# Memory recall: "do you remember", "what do you know about X"
_RECALL_PATTERN = re.compile(
    r"\b(remember|recall|do you know|you mentioned|earlier you said)\b",
    re.IGNORECASE,
)

# Memory store: "remember that", "please note", "save this"
_REMEMBER_PATTERN = re.compile(
    r"\b(remember\s+that|please\s+note|save\s+that|note\s+that|keep\s+in\s+mind)\b",
    re.IGNORECASE,
)


def _normalize_math_text(text: str) -> str:
    """Convert natural language arithmetic operators to standard symbols."""
    t = text
    t = re.sub(r"\bmultiplied\s+by\b", "*", t, flags=re.IGNORECASE)
    t = re.sub(r"\btimes\b", "*", t, flags=re.IGNORECASE)
    t = re.sub(r"\bdivided\s+by\b", "/", t, flags=re.IGNORECASE)
    t = re.sub(r"\bplus\b", "+", t, flags=re.IGNORECASE)
    t = re.sub(r"\bminus\b", "-", t, flags=re.IGNORECASE)
    t = re.sub(r"\bto\s+the\s+power\s+of\b", "**", t, flags=re.IGNORECASE)
    return t


def _extract_math_expression(text: str) -> Optional[str]:
    """
    Try to extract a clean arithmetic expression from user text.
    Returns None if no clear expression is found.
    """
    normalized = _normalize_math_text(text)
    # Look for an expression starting with digit/paren and ending with digit/paren
    match = re.search(r"\(?\s*\d[\d\s\+\-\*\/\(\)\^\.%]*\d\)?", normalized)
    if match:
        candidate = match.group(0).strip()
        # Require at least one operator and at least 2 digits
        if re.search(r"\d.*[+\-*/^%]\s*.*\d|\d.*\*\*.*\d", candidate):
            return candidate.replace("^", "**")
    return None


def _extract_python_code(text: str) -> Optional[str]:
    """Extract Python code from a fenced code block."""
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Look for indented block after "execute" / "run"
    match = re.search(r"(?:run|execute):?\s*\n((?:[ \t].+\n?)+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


class AssistantController:
    """
    Central controller for AXEN Phase 2.

    Coordinates conversation memory, tool dispatch, and language model inference.

    Example:
        controller = AssistantController()
        controller.load_model()
        response = controller.respond("What is 123 * 456?")
        print(response)
    """

    def __init__(
        self,
        model: Optional[HFLocalModel] = None,
        tool_registry: Optional[ToolRegistry] = None,
        conversation_memory: Optional[ConversationMemory] = None,
        long_term_memory: Optional[LongTermMemory] = None,
        config: Optional[HFModelConfig] = None,
    ) -> None:
        self.model = model or HFLocalModel(config=config or HFModelConfig.from_yaml())
        self.tools = tool_registry or build_default_registry()
        self.memory = conversation_memory or ConversationMemory()
        self.long_memory = long_term_memory or LongTermMemory()
        self._model_loaded: bool = False
        self.last_debug_info: dict[str, Any] = {}
        self.agent_loop = AgentLoop(
            tools=self.tools,
            memory=self.memory,
            long_memory=self.long_memory,
            model=self.model,
        )
        self.last_agent_state: Optional[AgentState] = None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the local HF model into GPU/CPU memory."""
        logger.info("Loading AXEN local model ...")
        self.model.load()
        self._model_loaded = True
        logger.info("AXEN local model ready.")

    def unload_model(self) -> None:
        """Release the model from memory."""
        if self._model_loaded:
            self.model.unload()
            self._model_loaded = False

    @property
    def model_ready(self) -> bool:
        return self._model_loaded and self.model.is_loaded

    # ------------------------------------------------------------------
    # Main response method
    # ------------------------------------------------------------------

    def respond(
        self,
        user_input: str,
        *,
        verbose: bool = False,
        debug: bool = False,
    ) -> str:
        """
        Generate a response to the user's input via the agentic reasoning loop.

        Steps:
          1. Store user message in conversation memory.
          2. Execute agent loop (understand, classify, plan, act, observe, verify, respond).
          3. Populate structured debug metadata.
          4. Store assistant response in conversation memory.
          5. Return final answer.
        """
        if not user_input.strip():
            return "I didn't catch that. Could you type your message again?"

        # 1. Store user message
        self.memory.add_user(user_input)

        # 2. Run agentic reasoning loop
        response, state = self.agent_loop.run(user_input, verbose=verbose, debug=debug)
        self.last_agent_state = state

        # 3. Populate debug info
        first_tool_res = state.tool_results[0] if state.tool_results else {}
        stats = state.metadata.get("model_stats", {})
        messages = self.memory.to_messages()

        self.last_debug_info = {
            "task_type": state.task_type,
            "tool_decision": "tool_required" if state.metadata.get("tool_required") else "no_tool",
            "selected_tool": state.metadata.get("selected_tool"),
            "tool_result": first_tool_res.get("result", first_tool_res.get("error", "none")),
            "verification": state.verification.get("status") if state.verification else ("passed" if state.status == "completed" else "none"),
            "final_status": state.status,
            "route": "tool" if state.tool_calls else "model",
            "tool_used": bool(state.tool_calls),
            "input": user_input,
            "model": getattr(self.model, "model_name", "unknown"),
            "device": getattr(self.model, "_device", "unknown"),
            "message_count": len(messages),
            "roles": stats.get("roles", [m["role"] for m in messages]),
            "input_tokens": stats.get("input_tokens", 0),
            "generated_tokens": stats.get("generated_tokens", 0),
        }

        # 4. Store and return
        self.memory.add_assistant(response)
        return response

    # ------------------------------------------------------------------
    # Identity Guardrails
    # ------------------------------------------------------------------

    def _apply_identity_guardrails(self, user_input: str, response: str) -> str:
        """
        Ensure AXEN never invents a fictional company, human job, personal hobbies, or runaway greeting monologue.
        """
        cleaned = response.strip()
        lower_q = user_input.lower().strip()
        lower_r = cleaned.lower()

        # For simple greetings ('hi', 'hello', etc.), return concise greeting
        if re.match(r"^(hello|hi|hey|greetings|good\s+(morning|afternoon|evening|day)|howdy|sup)[\s\.\!\?]*$", lower_q):
            cleaned = cleaned.split("\n\n")[0].strip()
            cleaned = cleaned.split("\n")[0].strip()
            return cleaned

        # Identity questions: "who are you", "what are you"
        if re.search(r"\b(?:who\s+are\s+you|what\s+are\s+you|what\s+is\s+your\s+name)\b", lower_q):
            if any(phrase in lower_r for phrase in ["work for", "employed by", "founded by", "company called", "job search", "hobbies"]):
                return "I am AXEN, a local AI assistant running on your computer."

        # Company / employer questions
        if re.search(r"\b(?:what|which)\s+company\s+(?:do\s+you|are\s+you)\b|\bwho\s+do\s+you\s+work\s+for\b|\byour\s+employer\b", lower_q):
            if any(phrase in lower_r for phrase in ["i work for", "employed by", "founded by", "company called", "joined the company"]):
                return "I don't work for a company. I'm AXEN, a local AI assistant running on your computer."

        # Hobbies questions
        if re.search(r"\b(?:do\s+you\s+have|what\s+are\s+your)\s+hobbies\b", lower_q):
            if any(phrase in lower_r for phrase in ["i like to", "i enjoy playing", "i love playing", "my hobbies are", "guitar", "hiking", "watching movies"]):
                return "I don't have personal hobbies. I'm an AI assistant running locally on your computer."

        return cleaned

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _try_tool_dispatch(self, text: str, verbose: bool = False) -> Optional[str]:
        """
        Check if the user's message clearly calls for a tool.
        Returns a formatted response string, or None if no tool matched.
        """
        # ---- File reading ----
        file_match = _FILE_PATTERN.search(text)
        if file_match:
            path_arg = file_match.group(1).strip().strip("'\"")
            res = self.tools.call("file.read", {"path": path_arg})
            if res.success:
                content = res.output.get("content", "")
                return f"Contents of `{path_arg}`:\n\n```\n{content}\n```"
            return f"Could not read file: {res.error}"

        # ---- Symbolic math / MathSolver ----
        if _MATH_SOLVER_PATTERN.search(text):
            try:
                from reasoning.math.solver import MathSolver
                solver = MathSolver()
                result = solver.solve(text)
                if result.verified:
                    return f"{result.answer}"
            except Exception as exc:
                logger.debug("MathSolver failed to solve %r: %s", text, exc)

        # ---- Calculator ----
        if _CALC_PATTERN.search(text):
            expr = _extract_math_expression(text)
            if expr:
                result = self.tools.call("calculator.evaluate", {"expression": expr})
                if result.success:
                    output = result.output
                    val = output.get("display", str(output.get("result", "?")))
                    prefix = f"[Tool: calculator] `{expr}` = " if verbose else ""
                    return f"{prefix}**{val}**"
                # Calculator failed — fall through to LLM
                if verbose:
                    return f"[Tool: calculator] Error: {result.error}"

        # ---- Python execution ----
        if _PYTHON_PATTERN.search(text):
            code = _extract_python_code(text)
            if code:
                result = self.tools.call("python.execute", {"code": code})
                if result.success:
                    out = result.output
                    stdout = out.get("stdout", "")
                    stderr = out.get("stderr", "")
                    lines = [f"```\n{stdout}\n```" if stdout else "_(no output)_"]
                    if stderr:
                        lines.append(f"**stderr:**\n```\n{stderr}\n```")
                    return "I ran your Python code:\n\n" + "\n".join(lines)
                return f"Code execution failed: {result.error}"

        return None

    # ------------------------------------------------------------------
    # Memory operations
    # ------------------------------------------------------------------

    def _try_memory_ops(self, text: str) -> Optional[str]:
        """Handle explicit memory store/recall requests."""
        if _REMEMBER_PATTERN.search(text):
            # Extract the fact after "remember that" / "note that" etc.
            match = re.search(
                r"(?:remember\s+that|please\s+note|save\s+that|note\s+that|keep\s+in\s+mind)[,:\s]+(.+)",
                text,
                re.IGNORECASE,
            )
            if match:
                fact = match.group(1).strip().rstrip(".")
                # Use the first 40 chars as the key
                key = re.sub(r"[^a-z0-9_]", "_", fact[:40].lower()).strip("_")
                self.long_memory.remember(key, fact)
                return f"Got it — I've saved that: \"{fact}\""

        if _RECALL_PATTERN.search(text):
            all_facts = self.long_memory.all_facts()
            if not all_facts:
                return "I don't have any saved notes about you yet."
            lines = [f"- **{k}**: {v}" for k, v in list(all_facts.items())[-10:]]
            return "Here's what I remember:\n" + "\n".join(lines)

        return None

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def _build_messages_with_memory(self) -> list[dict]:
        """
        Build the full message list for the model, injecting relevant
        long-term memory facts into the system prompt.
        """
        messages = self.memory.to_messages()
        facts = self.long_memory.all_facts()
        if facts:
            # Inject user-saved facts as context
            fact_lines = "\n".join(f"- {k}: {v}" for k, v in facts.items())
            system_addendum = f"\n\nUser facts saved by user:\n{fact_lines}"
            if messages and messages[0]["role"] == "system":
                messages[0] = {
                    "role": "system",
                    "content": messages[0]["content"] + system_addendum,
                }
        return messages

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def reset_conversation(self) -> None:
        """Clear the short-term conversation memory."""
        self.memory.clear()

    def status(self) -> dict[str, Any]:
        """Return a summary of the controller's current state."""
        return {
            "model": repr(self.model),
            "model_loaded": self.model_ready,
            "conversation_turns": self.memory.turn_count,
            "memory_tokens_estimate": self.memory.token_estimate,
            "long_term_facts": len(self.long_memory),
            "tools": self.tools.names(),
        }

    def __repr__(self) -> str:
        return (
            f"AssistantController(model={self.model.model_name!r}, "
            f"loaded={self.model_ready}, "
            f"turns={self.memory.turn_count})"
        )
