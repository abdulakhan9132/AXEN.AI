"""
axen/memory/conversation_memory.py
====================================
Short-term conversation memory for AXEN.

Holds the current session's message history.
Supports message roles: system, user, assistant, tool.
Enforces a configurable token-based context limit by trimming oldest messages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    """A single message in the conversation."""
    role: Role
    content: str
    # Optional: tool name when role == "tool"
    tool_name: Optional[str] = None

    def to_dict(self) -> dict:
        d: dict = {"role": self.role, "content": self.content}
        if self.tool_name:
            d["tool_name"] = self.tool_name
        return d

    def __len__(self) -> int:
        """Rough token estimate: chars / 4."""
        return max(1, len(self.content) // 4)


class ConversationMemory:
    """
    Manages the current conversation's message history.

    Maintains an ordered list of Message objects.
    When the total estimated token count exceeds max_tokens,
    oldest non-system messages are removed to stay within limits.

    Usage:
        mem = ConversationMemory(max_tokens=1800)
        mem.add("user", "Hello!")
        mem.add("assistant", "Hi! How can I help?")
        messages = mem.to_messages()  # list of dicts ready for the model
    """

    DEFAULT_SYSTEM = (
        "You are AXEN, a helpful and concise local AI assistant running on the user's computer.\n"
        "Always respond directly to the user's actual question.\n"
        "If the user greets you with 'hi' or 'hello', give a friendly, short greeting.\n"
        "Never invent personal history, employers, hobbies, or fictional scenarios."
    )

    def __init__(
        self,
        max_tokens: int = 1800,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.max_tokens = max_tokens
        if system_prompt is None:
            system_prompt = self._load_system_prompt_from_config() or self.DEFAULT_SYSTEM
        self._system_prompt = system_prompt
        self._messages: list[Message] = []

    @classmethod
    def _load_system_prompt_from_config(cls) -> Optional[str]:
        """Try loading system_prompt from configs/axen_local.yaml."""
        import yaml
        from pathlib import Path
        cfg_path = Path("configs/axen_local.yaml")
        if cfg_path.exists():
            try:
                data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                prompt = data.get("system_prompt")
                if prompt and isinstance(prompt, str) and prompt.strip():
                    return prompt.strip()
            except Exception:
                pass
        return None

    # ------------------------------------------------------------------
    # Adding messages
    # ------------------------------------------------------------------

    def add(
        self,
        role: Role,
        content: str,
        tool_name: Optional[str] = None,
    ) -> None:
        """Append a message and trim to stay within the context window."""
        self._messages.append(Message(role=role, content=content, tool_name=tool_name))
        self._trim()

    def add_user(self, content: str) -> None:
        self.add("user", content)

    def add_assistant(self, content: str) -> None:
        self.add("assistant", content)

    def add_tool_result(self, tool_name: str, content: str) -> None:
        self.add("tool", content, tool_name=tool_name)

    # ------------------------------------------------------------------
    # Retrieving messages
    # ------------------------------------------------------------------

    def to_messages(self) -> list[dict]:
        """
        Return messages as a list of role/content dicts for the model.
        Always prepends the system prompt as the first message.
        """
        result = [{"role": "system", "content": self._system_prompt}]
        result.extend(m.to_dict() for m in self._messages)
        return result

    def get_history(self) -> list[Message]:
        """Return the raw message list (without system prompt)."""
        return list(self._messages)

    def last_n(self, n: int) -> list[Message]:
        """Return the last n messages."""
        return self._messages[-n:]

    def clear(self) -> None:
        """Clear all messages (does not reset the system prompt)."""
        self._messages.clear()

    def set_system_prompt(self, prompt: str) -> None:
        """Update the system instruction."""
        self._system_prompt = prompt

    @property
    def turn_count(self) -> int:
        """Number of user turns in history."""
        return sum(1 for m in self._messages if m.role == "user")

    @property
    def token_estimate(self) -> int:
        """Rough total token count of all messages."""
        return sum(len(m) for m in self._messages)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _trim(self) -> None:
        """
        Remove oldest non-system messages until under the token limit.
        System messages are never removed.
        """
        while self.token_estimate > self.max_tokens and len(self._messages) > 1:
            self._messages.pop(0)

    def __repr__(self) -> str:
        return (
            f"ConversationMemory(messages={len(self._messages)}, "
            f"tokens≈{self.token_estimate}, max={self.max_tokens})"
        )
