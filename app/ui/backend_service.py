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
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.model_loading: bool = False
        self.model_load_error: Optional[str] = None
        self._chat_lock = threading.Lock()
        self._foreground_requests = 0
        from axen.projects.project_manager import ProjectManager
        self.project_manager = ProjectManager.get_instance()
        self.controller.project_manager = self.project_manager
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
            self._sync_controller_memory(self.active_session_id)

    # -----------------------------------------------------------------------
    # Task Tracking & Cancellation
    # -----------------------------------------------------------------------

    def cancel_request(self, request_id: str) -> bool:
        """Cancel an in-flight agent request without stopping the server."""
        with self._lock:
            task = self.active_tasks.get(request_id)
            if task and task.get("cancel_event"):
                task["cancel_event"].set()
                logger.info("Cancelled request task: %s", request_id)
                return True
        return False

    def cleanup_task(self, request_id: str) -> None:
        """Remove a completed or aborted task from tracking."""
        with self._lock:
            self.active_tasks.pop(request_id, None)

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

    def run_foreground(self, function, *args, **kwargs):
        with self._lock:
            self._foreground_requests += 1
        try:
            return function(*args, **kwargs)
        finally:
            with self._lock:
                self._foreground_requests -= 1

    def unified_response(self, message, manager, *, current_file=None, attachments=None,
                         session_id=None, project_id=None, persist_messages=True, state_callback=None):
        """One AEXN request dispatcher, reusing local models, Think Engine and tools."""
        from app.ai.router import ModelRouter
        from app.ai.chat_adapter import ProviderChatAdapter, CODING_INSTRUCTIONS
        kind = ModelRouter.task_kind(message)
        with self._chat_lock:
            history = list(self.conversations.get(session_id, {}).get('messages', []))
        # The frontend may have saved this user turn already. Don't duplicate it.
        if history and history[-1].get('role') == 'user' and history[-1].get('content') == message:
            history = history[:-1]
        previous = next((turn for turn in reversed(history) if turn.get('role') == 'assistant'), {})
        new_code_request = bool(re.search(r"\b(give|write|generate|create|build)\b.*\b(code|program|script|page|website|app|example)\b|\b(simple|new|another|different)\b.*\b(code|program|script|example|page)\b", message, re.I))
        following_code = bool(not new_code_request and re.search(r"\b(change|update|add|remove|make|fix|use|replace)\b", message, re.I)
                              and '```' in previous.get('content', ''))
        if kind == 'conversation' and following_code:
            kind = 'coding'
        def finish(result):
            if persist_messages:
                sid = session_id if session_id in self.conversations else self.new_session(title=message[:36], project_id=project_id)
                with self._chat_lock:
                    conv = self.conversations[sid]
                    for role, text in [('user', message), ('assistant', result['response'])]:
                        conv['messages'].append({'id': str(uuid.uuid4()), 'role': role, 'content': text,
                                                 'capability': result.get('capability') if role == 'assistant' else None,
                                                 'timestamp': datetime.now().isoformat()})
                    conv['updated_at'] = datetime.now().isoformat()
                    self._save_conversations_to_disk()
                result['session_id'] = sid
            self._queue_answer_review(message, result)
            return result
        if kind == "document" and not attachments:
            return finish({"response": "Attach the document you want me to analyze, then send your question.", "capability": "document"})
        if kind == "image":
            from image_brain import ImageBrain
            if state_callback:
                state_callback("GENERATING", "Generating your image", {})
            result = ImageBrain().generate(message)
            paths = result.get("paths", [])
            if not paths:
                raise RuntimeError("Image pipeline returned no image")
            filename = Path(paths[0]).name
            return finish({"response": f"![Generated image](/images/{filename})", "capability": "image"})
        if kind == "coding":
            from app.context.models import AssembledContext, ModelRequest
            from axen.agent.intent import IntentInfo
            from axen.agent.think_engine import ThinkEngine
            think_engine = ThinkEngine()
            complexity = 'HIGH' if re.search(r"\b(complex|full.?stack|multi.?file)\b", message, re.I) else 'MEDIUM'
            metrics, config = think_engine.start_request(message,
                IntentInfo(intent='coding', complexity=complexity, requires_reasoning=True))
            if state_callback:
                state_callback('PLANNING', 'Preparing your code', {})
            messages = [{'role': 'system', 'content': CODING_INSTRUCTIONS}]
            relevant_history = history[-2:] if following_code else []
            messages += [{'role': turn['role'], 'content': turn['content']} for turn in relevant_history
                         if turn.get('role') in ('user', 'assistant')]
            messages.append({'role': 'user', 'content': message})
            try:
                answer = manager.get_model('router').generate(ModelRequest(
                    AssembledContext(messages_for_model=messages),
                    {'task_type': 'coding', 'max_tokens': min(4096, config.budget.max_tokens),
                     'timeout': config.budget.max_runtime_seconds}, {}))
                if not isinstance(answer, str) or not answer.strip():
                    raise RuntimeError('Coding model returned an empty response')
                think_engine.finish_request(metrics)
            except Exception:
                think_engine.finish_request(metrics, status='failed')
                raise
            return finish({'response': answer, 'capability': 'coding_example'})
        return self.chat(message, session_id=session_id, project_id=project_id,
            attachments=attachments, think_mode=None, state_callback=state_callback,
            inference_model=ProviderChatAdapter(manager.get_model("router"), "AEXN AI"),
            persist_messages=persist_messages)

    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        request_id: Optional[str] = None,
        attachments: Optional[List[str]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        state_callback: Optional[Callable[[str, str, Optional[Dict[str, Any]]], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        think_mode: Optional[str] = None,
        inference_model: Optional[Any] = None,
        persist_messages: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a user message through AssistantController.
        Returns full structured response including tool/verification metadata.
        """
        sid = session_id or self.active_session_id
        if sid not in self.conversations:
            sid = self.new_session(title=message[:30].strip(), project_id=project_id)

        self.active_session_id = sid
        conv = self.conversations[sid]
        eff_project_id = project_id or conv.get("project_id")
        if project_id and not conv.get("project_id"):
            conv["project_id"] = project_id

        with self._chat_lock:
            # Isolate session history: synchronize controller memory with this session
            self._sync_controller_memory(sid, exclude_last_user=not persist_messages)

            if request_id:
                with self._lock:
                    if cancel_event is None:
                        cancel_event = threading.Event()
                    self.active_tasks[request_id] = {
                        "cancel_event": cancel_event,
                        "session_id": sid,
                        "start_time": time.time(),
                    }

            # Combine state and status callbacks
            def _combined_callback(st_name: str, disp_status: str, meta: Optional[Dict[str, Any]] = None):
                if state_callback:
                    try:
                        import inspect
                        sig = inspect.signature(state_callback)
                        if len(sig.parameters) == 1:
                            state_callback({"state": st_name, "text": disp_status, "meta": meta or {}})
                        else:
                            state_callback(st_name, disp_status, meta or {})
                    except Exception as cb_err:
                        try:
                            state_callback({"state": st_name, "text": disp_status, "meta": meta or {}})
                        except Exception:
                            logger.debug("state_callback error: %s", cb_err)
                if status_callback:
                    try:
                        status_callback(disp_status)
                    except Exception as cb_err:
                        logger.debug("status_callback error: %s", cb_err)

            # Record user message in session
            user_msg = {
                "id": str(uuid.uuid4()),
                "role": "user",
                "content": message,
                "attachments": attachments or [],
                "timestamp": datetime.now().isoformat(),
            }
            if persist_messages:
                conv["messages"].append(user_msg)

            # Update title if it's the first message
            if len(conv["messages"]) == 1 or conv.get("title") == "New Chat":
                clean_title = message.strip().split("\n")[0][:36].strip()
                if clean_title:
                    conv["title"] = clean_title

            # Execute through AssistantController
            previous_model = self.controller.model
            try:
                if inference_model is not None:
                    self.controller.model = inference_model
                start_time = time.perf_counter()
                response_text = self.controller.respond(
                    message,
                    attachments=attachments,
                    project_id=eff_project_id,
                    state_callback=_combined_callback,
                    cancel_event=cancel_event,
                    request_id=request_id,
                    session_id=sid,
                    think_mode=think_mode,
                )
                elapsed_sec = time.perf_counter() - start_time
            finally:
                self.controller.model = previous_model
                if request_id:
                    self.cleanup_task(request_id)

        state: Optional[AgentState] = getattr(self.controller, "last_agent_state", None)
        debug_info = getattr(self.controller, "last_debug_info", {})

        task_type = state.task_type if state else "unknown"
        tool_used = bool(state.tool_calls) if state else bool(debug_info.get("tool_used"))
        selected_tool = (
            state.metadata.get("selected_tool")
            if state
            else debug_info.get("selected_tool")
        )
        tool_status = _format_tool_status(task_type, selected_tool)
        verification = state.verification if state else debug_info.get("verification")
        meta_dict = state.metadata if state else {}
        # Measured task telemetry can select authorized learning curricula.
        # A low-confidence signal alone is never an improvement measurement.
        if state and "python" in message.lower():
            try:
                from axen.evolution.learning_worker import get_worker
                from axen.evolution.learning_store import record_task_observation
                import hashlib
                record_task_observation(get_worker().store, skill="Python debugging",
                    task_failed=state.status == "failed",
                    low_confidence=isinstance(meta_dict.get("confidence"), (int, float)) and meta_dict["confidence"] < 0.5,
                    execution_failed=any(not r.get("success", True) for r in state.tool_results),
                    request_hash=hashlib.sha256(message.encode()).hexdigest())
            except (OSError, ValueError, TypeError) as exc:
                logger.warning("Learning telemetry unavailable: %s", type(exc).__name__)
        think_mode_used = meta_dict.get("think_mode", think_mode or "auto")
        think_metrics = meta_dict.get("think_metrics", {})
        claim_verification = meta_dict.get("claim_verification", {})

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
                "think_mode": think_mode_used,
                "think_metrics": think_metrics,
                "claim_verification": claim_verification,
            },
        }
        if persist_messages:
            conv["messages"].append(assistant_msg)
        conv["updated_at"] = datetime.now().isoformat()
        self._save_conversations_to_disk()

        result = {
            "session_id": sid,
            "project_id": eff_project_id,
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
            "think_mode": think_mode_used,
            "think_metrics": think_metrics,
            "claim_verification": claim_verification,
        }
        self._queue_answer_review(message, result)
        return result

    def _queue_answer_review(self, message, result):
        """Persist a bounded review job; inference stays on the answer agent."""
        try:
            from axen.evolution.background_agent import get_background_agent
            get_background_agent().observe(message, result.get('response', ''), {
                'verification': result.get('verification'),
                'capability': result.get('capability') or result.get('task_type'),
                'selected_tool': result.get('tool_name'),
            })
        except Exception as exc:
            logger.warning('Background answer review unavailable: %s', type(exc).__name__)

    # -----------------------------------------------------------------------
    # Session Management
    # -----------------------------------------------------------------------

    def new_session(self, title: str = "New Chat", project_id: Optional[str] = None) -> str:
        """Create a fresh conversation session and reset controller short-term memory."""
        self.controller.reset_conversation()
        new_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        self.conversations[new_id] = {
            "id": new_id,
            "project_id": project_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }
        self.active_session_id = new_id
        self._save_conversations_to_disk()
        return new_id

    def _sync_controller_memory(self, session_id: str, exclude_last_user: bool = False) -> None:
        """Synchronize controller short-term memory to exclusively contain this session's history."""
        self.controller.reset_conversation()
        conv = self.conversations.get(session_id)
        if not conv:
            return
        messages = conv.get("messages", [])
        if exclude_last_user and messages and messages[-1].get("role") == "user":
            messages = messages[:-1]
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if not content:
                continue
            if role == "user":
                self.controller.memory.add_user(content)
            elif role == "assistant":
                self.controller.memory.add_assistant(content)

    def switch_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Switch active conversation, restoring previous turns to memory if needed."""
        if session_id not in self.conversations:
            return None
        self.active_session_id = session_id
        self._sync_controller_memory(session_id)
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

    def list_sessions(self, search: Optional[str] = None, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List conversations ordered by recent update, filtered by search query or project_id."""
        items = list(self.conversations.values())
        if project_id == "all":
            pass
        elif project_id:
            items = [c for c in items if c.get("project_id") == project_id]
        else:
            items = [c for c in items if not c.get("project_id")]

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
                "project_id": c.get("project_id"),
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
            "dtype": st.get("dtype", "unknown"),
            "vram_allocated_mb": st.get("vram_allocated_mb", 0.0),
            "vram_reserved_mb": st.get("vram_reserved_mb", 0.0),
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

    # -----------------------------------------------------------------------
    # Memory Management Layer (Step 14 & 15)
    # -----------------------------------------------------------------------

    def list_memories(self, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """List all active persistent memories."""
        if hasattr(self.controller.long_memory, "manager"):
            items = self.controller.long_memory.manager.store.list_active(user_id=user_id)
            return [item.to_dict() for item in items]
        facts = self.controller.long_memory.all_facts()
        return [{"key": k, "value": v, "memory_type": "fact", "is_active": True} for k, v in facts.items()]

    def add_memory(self, key: str, value: str, memory_type: str = "fact", user_id: str = "default_user") -> Dict[str, Any]:
        """Save or update a memory item."""
        if hasattr(self.controller.long_memory, "manager"):
            ok, msg, mem_id = self.controller.long_memory.manager.remember(
                key=key, value=value, category=memory_type, user_id=user_id
            )
            return {"success": ok, "message": msg, "id": mem_id}
        self.controller.long_memory.remember(key, value, category=memory_type)
        return {"success": True, "message": "Saved memory."}

    def delete_memory(self, item_id: int) -> bool:
        """Deactivate memory by row ID."""
        if hasattr(self.controller.long_memory, "manager"):
            return self.controller.long_memory.manager.store.delete_by_id(item_id, hard=False)
        return False

    def clear_memories(self, user_id: str = "default_user") -> bool:
        """Clear all active memories for user."""
        self.controller.long_memory.clear()
        return True

    def export_memories(self, user_id: str = "default_user") -> Dict[str, Any]:
        """Export memories as JSON."""
        if hasattr(self.controller.long_memory, "manager"):
            return self.controller.long_memory.manager.export_memories(user_id=user_id)
        facts = self.controller.long_memory.all_facts()
        return {"memory_version": 1, "user_id": user_id, "memories": [{"key": k, "value": v} for k, v in facts.items()]}

    def import_memories(self, data: Dict[str, Any], user_id: str = "default_user") -> int:
        """Import memories from JSON dictionary."""
        if hasattr(self.controller.long_memory, "manager"):
            return self.controller.long_memory.manager.import_memories(data, user_id=user_id)
        mem_list = data.get("memories", [])
        for m in mem_list:
            if "key" in m and "value" in m:
                self.controller.long_memory.remember(m["key"], m["value"])
        return len(mem_list)

    # -----------------------------------------------------------------------
    # Knowledge Base / RAG Management Layer (Step 17 & 25)
    # -----------------------------------------------------------------------

    def list_knowledge_documents(self, collection_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all indexed documents in knowledge base."""
        if hasattr(self.controller, "knowledge_manager") and self.controller.knowledge_manager:
            docs = self.controller.knowledge_manager.list_documents(collection_id=collection_id)
            return [d.to_dict() for d in docs]
        return []

    def list_knowledge_collections(self) -> List[Dict[str, Any]]:
        """List all collections in knowledge base."""
        if hasattr(self.controller, "knowledge_manager") and self.controller.knowledge_manager:
            colls = self.controller.knowledge_manager.list_collections()
            return [c.to_dict() for c in colls]
        return []

    def ingest_knowledge_file(self, file_path: str, collection_id: str = "default") -> Dict[str, Any]:
        """Ingest a local document into knowledge base."""
        if hasattr(self.controller, "knowledge_manager") and self.controller.knowledge_manager:
            res = self.controller.knowledge_manager.add_document(file_path, collection_id=collection_id)
            return res.to_dict()
        return {"success": False, "error": "Knowledge manager not available"}

    def delete_knowledge_document(self, document_id: str) -> bool:
        """Delete a document from knowledge base."""
        if hasattr(self.controller, "knowledge_manager") and self.controller.knowledge_manager:
            return self.controller.knowledge_manager.remove_document(document_id)
        return False

    def clear_knowledge_collection(self, collection_id: str) -> int:
        """Clear all documents from a knowledge collection."""
        if hasattr(self.controller, "knowledge_manager") and self.controller.knowledge_manager:
            return self.controller.knowledge_manager.clear_collection(collection_id)
        return 0

    def search_knowledge(self, query: str, collection_id: Optional[str] = None, top_k: int = 4) -> List[Dict[str, Any]]:
        """Search knowledge base with hybrid retrieval."""
        if hasattr(self.controller, "knowledge_manager") and self.controller.knowledge_manager:
            return self.controller.knowledge_manager.search(query=query, collection_id=collection_id, top_k=top_k)
        return []

    # -----------------------------------------------------------------------
    # File Storage & Upload Management APIs
    # -----------------------------------------------------------------------

    def upload_file(self, filename: str, file_bytes: bytes, mime_type: Optional[str] = None) -> Dict[str, Any]:
        """Upload and extract text from a file into local storage."""
        from axen.storage.file_manager import FileManager
        fobj = FileManager.get_instance().upload_and_process(filename, file_bytes, mime_type=mime_type)
        return fobj.to_dict()

    def list_uploaded_files(self) -> List[Dict[str, Any]]:
        """List all stored uploaded files."""
        from axen.storage.file_manager import FileManager
        files = FileManager.get_instance().list_files()
        return [f.to_dict() for f in files]

    def get_uploaded_file(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get uploaded file details by file ID."""
        from axen.storage.file_manager import FileManager
        fobj = FileManager.get_instance().get_file(file_id)
        return fobj.to_dict() if fobj else None

    def delete_uploaded_file(self, file_id: str) -> bool:
        """Delete an uploaded file from storage."""
        from axen.storage.file_manager import FileManager
        return FileManager.get_instance().delete_file(file_id)

    def ingest_uploaded_file_to_knowledge(self, file_id: str, collection_id: str = "default") -> Dict[str, Any]:
        """Ingest an uploaded file into persistent KnowledgeManager."""
        from axen.storage.file_manager import FileManager
        km = getattr(self.controller, "knowledge_manager", None)
        if not km:
            return {"success": False, "error": "Knowledge manager not available"}
        return FileManager.get_instance().ingest_into_knowledge(file_id, km, collection_id=collection_id)

    # -----------------------------------------------------------------------
    # Project & Workspace APIs
    # -----------------------------------------------------------------------

    def list_projects(self, include_archived: bool = False, search: Optional[str] = None) -> List[Dict[str, Any]]:
        """List projects with message counts and file counts."""
        projs = self.project_manager.list_projects(include_archived=include_archived, search=search)
        results = []
        for p in projs:
            d = p.to_dict()
            d["chat_count"] = sum(1 for c in self.conversations.values() if c.get("project_id") == p.id)
            d["file_count"] = len(self.project_manager.store.list_project_file_ids(p.id))
            results.append(d)
        return results

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get full project details including scoped stats."""
        p = self.project_manager.get_project(project_id)
        if not p:
            return None
        d = p.to_dict()
        d["chat_count"] = sum(1 for c in self.conversations.values() if c.get("project_id") == p.id)
        d["file_count"] = len(self.project_manager.store.list_project_file_ids(p.id))
        d["memories_count"] = len(self.project_manager.list_project_memories(p.id))
        d["collections_count"] = len(self.project_manager.list_project_collections(p.id))
        return d

    def create_project(
        self,
        name: str,
        description: str = "",
        instructions: str = "",
        icon: str = "folder",
        settings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a new project workspace."""
        p = self.project_manager.create_project(
            name=name,
            description=description,
            instructions=instructions,
            icon=icon,
            settings=settings,
        )
        return p.to_dict()

    def update_project(self, project_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update an existing project workspace."""
        p = self.project_manager.update_project(project_id, updates)
        return p.to_dict() if p else None

    def delete_project(self, project_id: str, delete_physical_files: bool = False) -> bool:
        """Delete a project workspace and optionally delete associated physical files."""
        for c in self.conversations.values():
            if c.get("project_id") == project_id:
                c["project_id"] = None
        self._save_conversations_to_disk()
        return self.project_manager.delete_project(project_id, delete_physical_files=delete_physical_files)

    def archive_project(self, project_id: str) -> bool:
        """Archive a project workspace."""
        return self.project_manager.archive_project(project_id)

    def restore_project(self, project_id: str) -> bool:
        """Restore an archived project workspace."""
        return self.project_manager.restore_project(project_id)

    def list_project_memories(self, project_id: str) -> List[Dict[str, Any]]:
        """List scoped memories for a project."""
        return self.project_manager.list_project_memories(project_id)

    def add_project_memory(self, project_id: str, key: str, value: str, memory_type: str = "fact") -> Dict[str, Any]:
        """Add or update a memory item scoped to a project."""
        mid = self.project_manager.add_project_memory(project_id, key=key, value=value, memory_type=memory_type)
        return {"success": True, "id": mid, "key": key, "value": value}

    def delete_project_memory(self, project_id: str, key: str) -> bool:
        """Deactivate a project-scoped memory."""
        return self.project_manager.delete_project_memory(project_id, key)

    def list_project_files(self, project_id: str) -> List[Dict[str, Any]]:
        """List files linked to a project."""
        files = self.project_manager.list_project_files(project_id)
        return [f.to_dict() for f in files]

    def add_project_file(self, project_id: str, file_id: str) -> bool:
        """Link an uploaded file to a project."""
        return self.project_manager.add_file_to_project(project_id, file_id)

    attach_file_to_project = add_project_file

    def remove_project_file(self, project_id: str, file_id: str, delete_physical: bool = False) -> bool:
        """Unlink file from project and optionally delete physical file."""
        return self.project_manager.remove_file_from_project(project_id, file_id, delete_physical=delete_physical)

    def list_project_knowledge(self, project_id: str) -> List[str]:
        """List knowledge collections associated with a project."""
        return self.project_manager.list_project_collections(project_id)

    def associate_project_knowledge(self, project_id: str, collection_id: str) -> bool:
        """Associate a knowledge collection with a project."""
        return self.project_manager.associate_knowledge(project_id, collection_id)

    # -----------------------------------------------------------------------
    # Unified Memory 2.0 Management APIs
    # -----------------------------------------------------------------------

    def list_memories(
        self,
        scope: Optional[str] = None,
        project_id: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List memories filtered by scope, project, or text search."""
        mem_mgr = getattr(self.controller._long_memory, "manager", None)
        if not mem_mgr:
            return []
        if search and search.strip():
            items = mem_mgr.search(search.strip(), scope=scope, project_id=project_id)
            return items
        items = mem_mgr.list_memories(is_active=True, scope=scope, project_id=project_id)
        return [m.to_dict() for m in items]

    def add_memory(
        self,
        key: str,
        value: str,
        scope: str = "global",
        project_id: Optional[str] = None,
        memory_type: str = "fact",
    ) -> Dict[str, Any]:
        """Add a persistent memory under the specified scope."""
        mem_mgr = getattr(self.controller._long_memory, "manager", None)
        if not mem_mgr:
            return {"success": False, "error": "Memory manager not available"}
        item = mem_mgr.remember(
            key=key,
            value=value,
            scope=scope,
            project_id=project_id,
            memory_type=memory_type,
            source="explicit_user",
        )
        if item:
            return {"success": True, "memory": item.to_dict()}
        return {"success": False, "error": "Memory rejected by policy or validation"}

    def delete_memory(
        self,
        key: str,
        scope: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> bool:
        """Delete / deactivate a memory item by key and scope."""
        mem_mgr = getattr(self.controller._long_memory, "manager", None)
        if not mem_mgr:
            return False
        return mem_mgr.forget(key, scope=scope, project_id=project_id)

    def clear_memories(
        self,
        scope: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> bool:
        """Clear memories within a given scope or project."""
        mem_mgr = getattr(self.controller._long_memory, "manager", None)
        if not mem_mgr:
            return False
        mem_mgr.clear(scope=scope, project_id=project_id)
        return True


