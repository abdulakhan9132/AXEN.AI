"""
app/ui/backend_service.py
=========================
Bridge between the modern AXEN Web UI and the existing AXEN backend.

Integrates:
- AssistantController (Phase 2 & Phase 3)
- Local TinyLlama PyTorch / CUDA inference
- Deterministic Tool Execution (Calculator, MathSolver, Python, File)
- Conversation memory & persistent session storage
- Long-term memory
- Real-time status reporting ("Calculating...", "Running Python...", "Verifying result...")
"""

from __future__ import annotations

import json
import logging
import re
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from axen import AssistantController, HFModelConfig
from axen.agent.state import AgentState

logger = logging.getLogger("axen.ui.backend")
CONVERSATIONS_FILE = ROOT / "data" / "axen_conversations.json"


def _format_tool_status(task_type: str, selected_tool: Optional[str]) -> str:
    """Return a concise status message without exposing chain-of-thought."""
    if selected_tool == "calculator" or task_type == "calculation":
        return "Calculating..."
    elif selected_tool == "python" or task_type == "code_execution":
        return "Running Python..."
    elif selected_tool == "math_solver" or task_type == "math":
        return "Analyzing math problem..."
    elif selected_tool == "file" or task_type == "file_read":
        return "Reading file..."
    elif selected_tool == "memory" or task_type in ("memory_recall", "memory_store"):
        return "Accessing memory..."
    elif task_type in ("complex_reasoning", "multistep"):
        return "Analyzing and planning..."
    return "Thinking..."


class AxenBackendService:
    """
    Manages the live AssistantController and session persistence for the web UI.
    Guarantees thread-safe access and zero cloud dependencies.
    """

    _instance: Optional["AxenBackendService"] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "AxenBackendService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self) -> None:
        self.config = HFModelConfig.from_yaml()
        self.controller = AssistantController(config=self.config)
        self.active_session_id: str = str(uuid.uuid4())
        self.conversations: Dict[str, Dict[str, Any]] = {}
        self.model_loading: bool = False
        self.model_load_error: Optional[str] = None
        self._load_conversations_from_disk()

        # If a session wasn't loaded, create a fresh default one
        if not self.conversations:
            self.new_session()
        else:
            # Set active session to most recent
            sorted_convs = sorted(
                self.conversations.values(),
                key=lambda x: x.get("updated_at", ""),
                reverse=True,
            )
            self.active_session_id = sorted_convs[0]["id"]

    # -----------------------------------------------------------------------
    # Model Loading (Async / Background)
    # -----------------------------------------------------------------------

    def start_model_load_in_background(self) -> None:
        """Asynchronously load TinyLlama into CUDA/CPU without blocking server startup."""
        if self.controller.model_ready or self.model_loading:
            return

        def _worker():
            self.model_loading = True
            try:
                logger.info("Background model loading started...")
                self.controller.load_model()
                logger.info("Background model loading completed successfully.")
            except Exception as e:
                logger.error("Background model loading error: %s", e)
                self.model_load_error = str(e)
            finally:
                self.model_loading = False

        thread = threading.Thread(target=_worker, daemon=True, name="AxenModelLoader")
        thread.start()

    # -----------------------------------------------------------------------
    # Chat Dispatch with Status Events
    # -----------------------------------------------------------------------

    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Process a user message through AssistantController.
        Returns full structured response including tool/verification metadata.
        """
        sid = session_id or self.active_session_id
        if sid not in self.conversations:
            sid = self.new_session(title=message[:30].strip())

        conv = self.conversations[sid]
        
        # Determine likely tool to give immediate UI status feedback
        classified_type, tool_req, sel_tool = self.controller.agent_loop.classifier.classify(message)
        initial_status = _format_tool_status(classified_type, sel_tool)
        if status_callback:
            status_callback(initial_status)

        # Record user message in session
        user_msg = {
            "id": str(uuid.uuid4()),
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat(),
        }
        conv["messages"].append(user_msg)

        # Update title if it's the first message
        if len(conv["messages"]) == 1 or conv.get("title") == "New Chat":
            clean_title = message.strip().split("\n")[0][:36].strip()
            if clean_title:
                conv["title"] = clean_title

        # Execute through AssistantController
        start_time = time.perf_counter()
        response_text = self.controller.respond(message)
        elapsed_sec = time.perf_counter() - start_time

        state: Optional[AgentState] = getattr(self.controller, "last_agent_state", None)
        debug_info = getattr(self.controller, "last_debug_info", {})

        task_type = state.task_type if state else classified_type
        tool_used = bool(state.tool_calls) if state else bool(debug_info.get("tool_used"))
        selected_tool = (
            state.metadata.get("selected_tool")
            if state
            else debug_info.get("selected_tool")
        )
        tool_status = _format_tool_status(task_type, selected_tool)
        verification = state.verification if state else debug_info.get("verification")

        assistant_msg = {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "task_type": task_type,
                "tool_used": tool_used,
                "selected_tool": selected_tool,
                "tool_status": tool_status,
                "verification": verification,
                "elapsed_sec": round(elapsed_sec, 2),
                "model": getattr(self.controller.model, "model_name", "TinyLlama"),
                "device": getattr(self.controller.model, "_device", "cpu"),
            },
        }
        conv["messages"].append(assistant_msg)
        conv["updated_at"] = datetime.now().isoformat()
        self._save_conversations_to_disk()

        return {
            "session_id": sid,
            "response": response_text,
            "task_type": task_type,
            "tool_used": tool_used,
            "tool_name": selected_tool,
            "tool_status": tool_status,
            "verification": verification,
            "status": state.status if state else "completed",
            "elapsed": round(elapsed_sec, 2),
            "device": getattr(self.controller.model, "_device", "cpu"),
            "model_ready": self.controller.model_ready,
        }

    # -----------------------------------------------------------------------
    # Session Management
    # -----------------------------------------------------------------------

    def new_session(self, title: str = "New Chat") -> str:
        """Create a fresh conversation session and reset controller short-term memory."""
        self.controller.reset_conversation()
        new_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        self.conversations[new_id] = {
            "id": new_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }
        self.active_session_id = new_id
        self._save_conversations_to_disk()
        return new_id

    def switch_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Switch active conversation, restoring previous turns to memory if needed."""
        if session_id not in self.conversations:
            return None
        self.active_session_id = session_id
        self.controller.reset_conversation()
        # Re-populate short-term memory with this session's history
        for msg in self.conversations[session_id].get("messages", []):
            if msg.get("role") == "user":
                self.controller.memory.add_user(msg.get("content", ""))
            elif msg.get("role") == "assistant":
                self.controller.memory.add_assistant(msg.get("content", ""))
        return self.conversations[session_id]

    def delete_session(self, session_id: str) -> bool:
        """Delete a conversation from history."""
        if session_id in self.conversations:
            del self.conversations[session_id]
            if self.active_session_id == session_id:
                if self.conversations:
                    self.active_session_id = list(self.conversations.keys())[0]
                    self.switch_session(self.active_session_id)
                else:
                    self.new_session()
            self._save_conversations_to_disk()
            return True
        return False

    def rename_session(self, session_id: str, new_title: str) -> bool:
        """Update conversation title."""
        if session_id in self.conversations and new_title.strip():
            self.conversations[session_id]["title"] = new_title.strip()
            self.conversations[session_id]["updated_at"] = datetime.now().isoformat()
            self._save_conversations_to_disk()
            return True
        return False

    def list_sessions(self, search: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all conversations ordered by recent update, with optional search filter."""
        items = list(self.conversations.values())
        if search:
            q = search.lower().strip()
            filtered = []
            for item in items:
                match_title = q in item.get("title", "").lower()
                match_msgs = any(
                    q in m.get("content", "").lower()
                    for m in item.get("messages", [])
                )
                if match_title or match_msgs:
                    filtered.append(item)
            items = filtered

        items.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return [
            {
                "id": c["id"],
                "title": c.get("title", "New Chat"),
                "created_at": c.get("created_at"),
                "updated_at": c.get("updated_at"),
                "message_count": len(c.get("messages", [])),
                "is_active": c["id"] == self.active_session_id,
            }
            for c in items
        ]

    def get_active_session(self) -> Dict[str, Any]:
        """Return the active conversation data."""
        if self.active_session_id not in self.conversations:
            self.new_session()
        return self.conversations[self.active_session_id]

    # -----------------------------------------------------------------------
    # Status & Diagnostics
    # -----------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return system health, model status, and device metadata."""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            cuda_device = torch.cuda.get_device_name(0) if cuda_available else "N/A"
        except ImportError:
            cuda_available = False
            cuda_device = "N/A"

        st = self.controller.status()
        return {
            "status": "ready",
            "model_name": self.config.name,
            "model_loaded": self.controller.model_ready,
            "model_loading": self.model_loading,
            "model_error": self.model_load_error,
            "cuda_available": cuda_available,
            "cuda_device": cuda_device,
            "device": getattr(self.controller.model, "_device", "cpu"),
            "conversation_turns": st.get("conversation_turns", 0),
            "long_term_facts": st.get("long_term_facts", 0),
            "tools": st.get("tools", []),
            "active_session_id": self.active_session_id,
            "sessions_count": len(self.conversations),
        }

    # -----------------------------------------------------------------------
    # Disk Persistence
    # -----------------------------------------------------------------------

    def _load_conversations_from_disk(self) -> None:
        if CONVERSATIONS_FILE.exists():
            try:
                data = json.loads(CONVERSATIONS_FILE.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self.conversations = data
            except Exception as e:
                logger.warning("Could not read conversations file: %s", e)
                self.conversations = {}

    def _save_conversations_to_disk(self) -> None:
        try:
            CONVERSATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
            CONVERSATIONS_FILE.write_text(
                json.dumps(self.conversations, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error("Could not write conversations file: %s", e)
