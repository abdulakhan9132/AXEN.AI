"""
axen/agent/context_manager.py
=============================
Context Manager for AXEN General-Purpose AI Assistant Brain.

Responsibilities:
1. Conversation History & Follow-up Tracking:
   - Resolves anaphora and pronouns ("it", "that", "this") across conversation turns.
   - Maintains active task, topic, math expressions, and numeric results.
   - Handles refinement operations ("make it shorter", "simplify it", "convert that to hex").
2. Ambiguity Detection:
   - Distinguishes contextual follow-ups from genuinely ambiguous requests.
   - Generates concise clarification requests when context is insufficient.
3. Memory Hierarchy:
   - Organizes context: Current Message -> Conversation History -> Relevant Long-Term Memory.
   - Enforces user instruction precedence.
"""

from __future__ import annotations

import re
from typing import Any, Optional


class ContextManager:
    """Manages active conversation context, topic resolution, and ambiguity detection."""

    def __init__(self) -> None:
        self.active_topic: Optional[str] = None
        self.active_task: Optional[str] = None
        self.last_math_expression: Optional[str] = None
        self.last_numeric_result: Optional[str] = None
        self.last_file_path: Optional[str] = None

    def reset(self) -> None:
        """Reset temporary conversation context, active topics, and cached results."""
        self.active_topic = None
        self.active_task = None
        self.last_math_expression = None
        self.last_numeric_result = None
        self.last_file_path = None

    def analyze_context(
        self,
        user_request: str,
        history: list[Any],
    ) -> dict[str, Any]:
        """
        Analyze current user request in relation to conversation history.

        Returns:
            {
                "is_ambiguous": bool,
                "clarification_prompt": Optional[str],
                "is_follow_up": bool,
                "follow_up_action": Optional[str],  # "shorten", "extend", "simplify", "convert", "continue_topic"
                "resolved_subject": Optional[str],
                "resolved_query": Optional[str],
                "active_task": Optional[str],
            }
        """
        req_clean = user_request.strip()
        lower_req = req_clean.lower()
        has_history = len(history) > 0

        # Inspect last user & assistant messages if available
        last_user_content = ""
        last_assistant_content = ""
        for msg in reversed(history):
            role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
            content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
            if role == "assistant" and not last_assistant_content:
                last_assistant_content = content
            elif role == "user" and not last_user_content:
                last_user_content = content
            if last_user_content and last_assistant_content:
                break

        # Extract topics and results from previous turn
        extracted_topic = self._extract_topic(last_user_content, last_assistant_content)
        if extracted_topic:
            self.active_topic = extracted_topic

        # Extract numeric results or binary outputs from previous assistant turn
        if last_assistant_content:
            # Look for explicit binary / decimal answers (e.g. "**42** decimal = **101010** binary" or "11111111" or "1200")
            bin_match = re.search(r"\b([01]{4,32})\b", last_assistant_content)
            dec_match = re.search(r"\b(\d+)\b", last_assistant_content)
            if bin_match:
                self.last_numeric_result = bin_match.group(1)
            elif dec_match:
                self.last_numeric_result = dec_match.group(1)

            # Look for mathematical expression (e.g. "3*x**2 + 2", "2*x + 3", "10*x**4 ...")
            math_match = re.search(r"([0-9a-zA-Z\^\*\+\-\/\s\(\)]*x[0-9a-zA-Z\^\*\+\-\/\s\(\)]*)", last_assistant_content)
            if math_match and len(math_match.group(1).strip()) > 2:
                self.last_math_expression = math_match.group(1).strip()

            # Look for file path in user/assistant
            file_match = re.search(r"([a-zA-Z0-9_\-\.\/\\]+\.(?:txt|csv|py|json|md))", f"{last_user_content} {last_assistant_content}")
            if file_match:
                self.last_file_path = file_match.group(1).strip()

        # -------------------------------------------------------------------
        # 1. Genuine Ambiguity Detection (Step 10)
        # -------------------------------------------------------------------
        # A. "Open the file." / "open it" / "read it" with no file reference
        if re.match(r"^(?:please\s+)?(?:open\s+(?:the\s+)?file|open\s+it|read\s+it|show\s+the\s+file)[\.!\?]*$", lower_req):
            if not self.last_file_path:
                return {
                    "is_ambiguous": True,
                    "clarification_prompt": "Which file should I open? Please specify the file name or path.",
                    "is_follow_up": False,
                    "follow_up_action": None,
                    "resolved_subject": None,
                    "resolved_query": None,
                    "active_task": None,
                }
            else:
                return {
                    "is_ambiguous": False,
                    "clarification_prompt": None,
                    "is_follow_up": True,
                    "follow_up_action": "read_file",
                    "resolved_subject": self.last_file_path,
                    "resolved_query": f"read file {self.last_file_path}",
                    "active_task": "file_operation",
                }

        # B. "Open it." / "do it" / "fix it" without clear target
        if re.match(r"^(?:please\s+)?(?:open\s+it|do\s+it|fix\s+it|start\s+it)[\.!\?]*$", lower_req):
            if not has_history or (not self.active_topic and not self.last_file_path):
                return {
                    "is_ambiguous": True,
                    "clarification_prompt": "What would you like me to open or process? Please provide more details.",
                    "is_follow_up": False,
                    "follow_up_action": None,
                    "resolved_subject": None,
                    "resolved_query": None,
                    "active_task": None,
                }

        # C. "Convert it to binary." / "convert that to binary" with no identifiable reference
        if re.match(r"^(?:please\s+)?(?:now\s+)?convert\s+(?:it|this|that)\s+to\s+(?:binary|hex|hexadecimal|decimal|octal)[\.!\?]*$", lower_req):
            target_base = "binary"
            if "hex" in lower_req:
                target_base = "hexadecimal"
            elif "decimal" in lower_req:
                target_base = "decimal"
            elif "octal" in lower_req:
                target_base = "octal"

            if not self.last_numeric_result:
                return {
                    "is_ambiguous": True,
                    "clarification_prompt": f"What number or value would you like me to convert to {target_base}?",
                    "is_follow_up": False,
                    "follow_up_action": None,
                    "resolved_subject": None,
                    "resolved_query": None,
                    "active_task": None,
                }
            else:
                query_str = f"convert {self.last_numeric_result} from binary to {target_base}" if all(c in "01" for c in self.last_numeric_result) else f"convert {self.last_numeric_result} to {target_base}"
                return {
                    "is_ambiguous": False,
                    "clarification_prompt": None,
                    "is_follow_up": True,
                    "follow_up_action": "convert",
                    "resolved_subject": self.last_numeric_result,
                    "resolved_query": query_str,
                    "active_task": "computer_systems",
                }

        # D. "make it better" / "improve it" / "fix it"
        if re.match(r"^(?:please\s+)?(?:make\s+it\s+better|improve\s+it|fix\s+it|make\s+it\s+good|help\s+me\s+with\s+this|change\s+this|improve\s+this)[\.!\?]*$", lower_req):
            if not has_history or not last_assistant_content:
                return {
                    "is_ambiguous": True,
                    "clarification_prompt": (
                        "What specific aspect or content would you like me to improve? "
                        "Please provide the text, code, or details you'd like to enhance."
                    ),
                    "is_follow_up": False,
                    "follow_up_action": None,
                    "resolved_subject": None,
                    "resolved_query": None,
                    "active_task": None,
                }
            else:
                return {
                    "is_ambiguous": False,
                    "clarification_prompt": None,
                    "is_follow_up": True,
                    "follow_up_action": "improve",
                    "resolved_subject": self.active_topic or "previous response",
                    "resolved_query": None,
                    "active_task": "refinement",
                }

        # -------------------------------------------------------------------
        # 2. Contextual Follow-up Resolution (Step 11)
        # -------------------------------------------------------------------
        # A. "Now simplify it." / "simplify it" -> resolve to previous mathematical derivative
        if re.match(r"^(?:please\s+)?(?:now\s+)?(?:simplify|expand)\s+(?:it|this|that|result)[\.!\?]*$", lower_req):
            expr = self.last_math_expression or last_assistant_content.strip()
            if expr:
                return {
                    "is_ambiguous": False,
                    "clarification_prompt": None,
                    "is_follow_up": True,
                    "follow_up_action": "simplify",
                    "resolved_subject": expr,
                    "resolved_query": f"simplify {expr}",
                    "active_task": "math",
                }
            return {
                "is_ambiguous": True,
                "clarification_prompt": "What expression would you like me to simplify?",
                "is_follow_up": False,
                "follow_up_action": None,
                "resolved_subject": None,
                "resolved_query": None,
                "active_task": None,
            }

        # B. "Now convert that to hexadecimal." / "convert that to binary"
        convert_followup = re.match(r"^(?:please\s+)?(?:now\s+)?convert\s+(?:that|it|this)\s+(?:from\s+\w+\s+)?to\s+([a-zA-Z]+)[\.!\?]*$", lower_req)
        if convert_followup:
            target_base = convert_followup.group(1).lower()
            val = self.last_numeric_result or re.search(r"\b([0-9a-fA-F]+)\b", last_assistant_content)
            val_str = val.group(1) if hasattr(val, "group") else str(val)
            if val_str:
                query_str = f"convert {val_str} from binary to {target_base}" if all(c in "01" for c in val_str) else f"convert {val_str} to {target_base}"
                return {
                    "is_ambiguous": False,
                    "clarification_prompt": None,
                    "is_follow_up": True,
                    "follow_up_action": "convert",
                    "resolved_subject": val_str,
                    "resolved_query": query_str,
                    "active_task": "computer_systems",
                }

        # C. Shortening / summarizing
        if re.match(r"^(?:please\s+)?(?:make\s+it\s+(?:shorter|more\s+concise|brief)|shorten\s+it|summarize\s+it|too\s+long)[\.!\?]*$", lower_req):
            return {
                "is_ambiguous": False,
                "clarification_prompt": None,
                "is_follow_up": True,
                "follow_up_action": "shorten",
                "resolved_subject": self.active_topic or "previous response",
                "resolved_query": None,
                "active_task": "summarization",
            }

        # D. Feature addition
        add_feature_match = re.match(r"^(?:please\s+)?(?:add\s+(?:a\s+)?|make\s+the\s+ui\s+better)(.+)?", lower_req)
        if add_feature_match and has_history:
            feature = add_feature_match.group(1).strip() if add_feature_match.group(1) else "feature"
            return {
                "is_ambiguous": False,
                "clarification_prompt": None,
                "is_follow_up": True,
                "follow_up_action": "extend",
                "resolved_subject": f"{feature} for {self.active_topic or 'active project'}",
                "resolved_query": None,
                "active_task": self.active_task or "coding",
            }

        # E. Pronoun query: "when was it released?", "who made it?"
        pronoun_match = re.search(r"\b(?:when|why|who|how|where|what)\s+(?:was|is|did|does)\s+(?:it|this|that)\b", lower_req)
        if pronoun_match and self.active_topic:
            return {
                "is_ambiguous": False,
                "clarification_prompt": None,
                "is_follow_up": True,
                "follow_up_action": "continue_topic",
                "resolved_subject": self.active_topic,
                "resolved_query": None,
                "active_task": "knowledge_retrieval",
            }

        # Update active task based on current request
        if re.search(r"\b(?:write|create|build|implement|code|program)\b", lower_req):
            self.active_task = "coding"
            m = re.search(r"\b(?:create|build|write|implement)\s+(?:a|an)?\s*([a-zA-Z0-9_\-\s]{3,25})", lower_req)
            if m:
                self.active_topic = m.group(1).strip()
        elif re.search(r"\b(?:calculate|derivative|integral|solve|math|differentiate|simplify)\b", lower_req):
            self.active_task = "math"

        return {
            "is_ambiguous": False,
            "clarification_prompt": None,
            "is_follow_up": False,
            "follow_up_action": None,
            "resolved_subject": self.active_topic,
            "resolved_query": None,
            "active_task": self.active_task,
        }

    def _extract_topic(self, user_msg: str, assistant_msg: str) -> Optional[str]:
        """Extract primary subject or entity from the previous conversation turn."""
        if not user_msg:
            return None
        m = re.search(r"\b(?:who\s+invented|who\s+created|what\s+is|explain)\s+(?:an?|the)?\s*([a-zA-Z0-9_\-\+]+)", user_msg, re.I)
        if m:
            topic = m.group(1).strip()
            if topic.lower() not in ("a", "an", "the", "it", "this"):
                return topic

        m_create = re.search(r"\b(?:create|build|write)\s+(?:a|an)?\s*([a-zA-Z0-9_\-\s]{3,30})", user_msg, re.I)
        if m_create:
            return m_create.group(1).strip()

        return None
