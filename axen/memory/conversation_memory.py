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


MAX_RECENT_MESSAGES: int = 12

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
        "You are AXEN, a local AI assistant. Answer the user's current request directly and naturally. "
        "Use available tools when necessary. Do not invent information. Follow the conversation context. "
        "Do not mention internal instructions or hidden reasoning."
    )

    MAX_RECENT_MESSAGES: int = 12

    def __init__(
        self,
        max_tokens: int = 1800,
        system_prompt: Optional[str] = None,
        max_recent_messages: int = 12,
        max_history: Optional[int] = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.max_recent_messages = max_history if max_history is not None else max_recent_messages
        if system_prompt is None:
            system_prompt = self._load_system_prompt_from_config() or self.DEFAULT_SYSTEM
        self._system_prompt = system_prompt
        self._messages: list[Message] = []
        self._conversation_summary: Optional[str] = None

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

    def add_user_message(self, content: str) -> None:
        self.add_user(content)

    def add_assistant(self, content: str) -> None:
        self.add("assistant", content)

    def add_assistant_message(self, content: str) -> None:
        self.add_assistant(content)

    def add_tool_result(self, tool_name: str, content: str) -> None:
        self.add("tool", content, tool_name=tool_name)

    # ------------------------------------------------------------------
    # Retrieving messages
    # ------------------------------------------------------------------

    def to_messages(self) -> list[dict]:
        """
        Return messages as a list of role/content dicts for the model.
        Always prepends the system prompt as the first message.
        Injects a concise conversation summary if older messages were trimmed.
        Ensures valid HuggingFace chat template roles (system, user, assistant).
        """
        sys_content = self._system_prompt
        if self._conversation_summary:
            sys_content += f"\n\nEarlier Conversation Summary:\n{self._conversation_summary}"

        result = [{"role": "system", "content": sys_content}]
        for m in self._messages:
            role = m.role
            content = m.content
            if role == "tool":
                role = "user"
                content = f"[Tool {m.tool_name or 'output'}]: {content}"
            result.append({"role": role, "content": content})
        return result

    def get_history(self) -> list[Message]:
        """Return the raw message list (without system prompt)."""
        return list(self._messages)

    def get_recent_history(self, limit: int = 12) -> list[Message]:
        """Return the last `limit` messages."""
        return self._messages[-limit:]

    def get_summary(self) -> Optional[str]:
        """Return the summary of trimmed messages, if any."""
        return self._conversation_summary

    def has_summary(self) -> bool:
        """Check if any earlier conversation history has been summarized."""
        return bool(self._conversation_summary)

    def last_n(self, n: int) -> list[Message]:
        """Return the last n messages."""
        return self._messages[-n:]

    def clear(self) -> None:
        """Clear all messages and reset conversation summary."""
        self._messages.clear()
        self._conversation_summary = None

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
        Remove oldest non-system messages until within max_recent_messages and token limits.
        Summarizes trimmed messages into _conversation_summary.
        """
        while len(self._messages) > self.max_recent_messages:
            oldest = self._messages.pop(0)
            self._update_summary(oldest)

        while self.token_estimate > self.max_tokens and len(self._messages) > 1:
            oldest = self._messages.pop(0)
            self._update_summary(oldest)

    def _update_summary(self, msg: Message) -> None:
        """Maintain a lightweight summary of trimmed messages."""
        snippet = msg.content.strip().split("\n")[0][:80]
        if not snippet:
            return
        entry = f"- {msg.role.capitalize()}: {snippet}"
        if not self._conversation_summary:
            self._conversation_summary = entry
        else:
            lines = self._conversation_summary.split("\n")
            if len(lines) >= 6:
                lines = lines[-5:]
            lines.append(entry)
            self._conversation_summary = "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ConversationMemory(messages={len(self._messages)}, "
            f"tokens≈{self.token_estimate}, max={self.max_tokens})"
        )
