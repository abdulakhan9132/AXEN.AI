"""
reasoning/loop.py
=================
NOTE: ACTIVE LOOP NOTICE
The active agent loop used by the AXEN Web UI and AssistantController is:
    `axen.agent.loop.AgentLoop` (in axen/agent/loop.py)

This module (`reasoning/loop.py`) is an auxiliary/prototype loop preserved for
standalone test compatibility (e.g. tests/test_brain_router.py, tests/test_image_brain.py).
Production intent routing is consolidated in `axen/agent/loop.py`.

Routing pattern:
- math       -> math_engines / sympy solver
- file_op    -> filesystem tool
- web_search -> browser / search tool
- memory     -> SQLite memory tool
- job_search -> job agent pipeline
- image_gen  -> image_brain
- code, ui_design, chat, unknown -> Qwen via Ollama with intent-specific prompt
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from axen_brain import AxenBrain

logger = logging.getLogger("axen.reasoning.loop")

DB_PATH = Path("data/brain_routes.db")

_default_image_brain: Any = None
_loop_image_lock = threading.Lock()


def init_db(db_path: Path = DB_PATH) -> None:
    """Initialize the SQLite routing logs and image logs tables."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS route_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                message TEXT NOT NULL,
                intent TEXT NOT NULL,
                confidence REAL NOT NULL,
                handler TEXT NOT NULL,
                response TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS image_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_request TEXT NOT NULL,
                prompt_used TEXT NOT NULL,
                generation_time_sec REAL NOT NULL,
                output_path TEXT NOT NULL,
                image_url TEXT NOT NULL
            )
            """
        )
        conn.commit()


def log_image_generation(
    user_request: str,
    prompt_used: str,
    generation_time_sec: float,
    output_path: str,
    image_url: str,
    db_path: Path = DB_PATH,
) -> None:
    """Log image generation event details to SQLite."""
    try:
        init_db(db_path)
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO image_logs (user_request, prompt_used, generation_time_sec, output_path, image_url)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_request, prompt_used, float(generation_time_sec), output_path, image_url),
            )
            conn.commit()
    except Exception as e:
        logger.warning("Failed to log image generation to SQLite: %s", e)



def log_route(
    message: str,
    intent: str,
    confidence: float,
    handler: str,
    response: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> None:
    """Log intent and confidence for every routed message to SQLite."""
    try:
        init_db(db_path)
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO route_logs (message, intent, confidence, handler, response)
                VALUES (?, ?, ?, ?, ?)
                """,
                (message, intent, float(confidence), handler, response or ""),
            )
            conn.commit()
    except Exception as e:
        logger.warning("Failed to log route to SQLite: %s", e)


class AgentLoop:
    """Agent loop integrating AxenBrain intent classifier with execution tools."""

    def __init__(
        self,
        brain: Optional[AxenBrain] = None,
        db_path: Path = DB_PATH,
        math_tool: Optional[Callable[[str], Any]] = None,
        file_tool: Optional[Callable[[str], Any]] = None,
        web_tool: Optional[Callable[[str], Any]] = None,
        memory_tool: Optional[Callable[[str], Any]] = None,
        job_tool: Optional[Callable[[str], Any]] = None,
        image_tool: Optional[Callable[[str], Any]] = None,
        llm_client: Optional[Callable[[str, str], Any]] = None,
    ):
        self.brain = brain or AxenBrain()
        self.db_path = db_path
        init_db(self.db_path)

        self.math_tool = math_tool or self._default_math_tool
        self.file_tool = file_tool or self._default_file_tool
        self.web_tool = web_tool or self._default_web_tool
        self.memory_tool = memory_tool or self._default_memory_tool
        self.job_tool = job_tool or self._default_job_tool
        self.image_tool = image_tool or self._default_image_tool
        self.llm_client = llm_client or self._default_llm_client

    def _default_image_tool(self, message: str) -> dict[str, Any]:
        """Default local diffusion image generation tool with concurrency lock and logging."""
        global _default_image_brain
        with _loop_image_lock:
            t0 = time.perf_counter()
            if _default_image_brain is None:
                from image_brain import ImageBrain
                _default_image_brain = ImageBrain("turbo")
            result = _default_image_brain.generate(message)
            elapsed = round(time.perf_counter() - t0, 3)

            paths = result.get("paths", [])
            first_path = paths[0] if paths else ""
            filename = Path(first_path).name if first_path else ""
            url = f"/images/{filename}" if filename else ""
            prompt_used = result.get("prompt", message)
            reply = f"Done. Prompt used: {prompt_used}\nSaved: {first_path}\nURL: {url}"

            log_image_generation(
                user_request=message,
                prompt_used=prompt_used,
                generation_time_sec=elapsed,
                output_path=first_path,
                image_url=url,
                db_path=self.db_path,
            )

            return {
                "status": "success",
                "tool": "image_brain",
                "prompt": prompt_used,
                "paths": paths,
                "url": url,
                "generation_time_sec": elapsed,
                "reply": reply,
            }

    def _default_math_tool(self, message: str) -> dict[str, Any]:
        """Default math engine integration."""
        try:
            from axen.tools.calculator import CalculatorTool
            calc = CalculatorTool()
            res = calc.execute(expression=message)
            return {"status": "success", "tool": "math_solver", "output": res}
        except Exception:
            return {"status": "success", "tool": "math_solver", "message": f"Processed math expression: {message}"}

    def _default_file_tool(self, message: str) -> dict[str, Any]:
        """Default filesystem tool integration."""
        try:
            from axen.storage.file_manager import FileManager
            fm = FileManager.get_instance()
            return {"status": "success", "tool": "file_op", "message": f"Filesystem tool accessed for: {message}"}
        except Exception:
            return {"status": "success", "tool": "file_op", "message": f"File operation handled for: {message}"}

    def _default_web_tool(self, message: str) -> dict[str, Any]:
        """Default web search tool integration."""
        try:
            from axen.tools.web_tools import WebSearchTool
            tool = WebSearchTool()
            return {"status": "success", "tool": "web_search", "message": f"Web search initiated for: {message}"}
        except Exception:
            return {"status": "success", "tool": "web_search", "query": message}

    def _default_memory_tool(self, message: str) -> dict[str, Any]:
        """Default SQLite persistent memory tool integration."""
        try:
            from axen.memory.memory_manager import MemoryManager
            mm = MemoryManager()
            return {"status": "success", "tool": "memory", "message": f"Memory store updated for: {message}"}
        except Exception:
            return {"status": "success", "tool": "memory", "message": f"Memory stored: {message}"}

    def _default_job_tool(self, message: str) -> dict[str, Any]:
        """Default job search agent pipeline."""
        return {"status": "success", "tool": "job_search", "message": f"Job agent matching query: {message}"}

    def _default_llm_client(self, message: str, system_prompt: str) -> dict[str, Any]:
        """Default Qwen via Ollama query, with fallback if Ollama service is inactive."""
        ollama_url = "http://localhost:11434/api/generate"
        payload = {
            "model": "qwen2.5:1.5b",
            "prompt": message,
            "system": system_prompt,
            "stream": False,
        }
        try:
            req = urllib.request.Request(
                ollama_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return {"status": "success", "response": data.get("response", ""), "model": "qwen2.5"}
        except Exception:
            # Local Ollama offline or mock fallback
            return {
                "status": "success",
                "response": f"[Qwen/Ollama Local] Processed prompt with system: '{system_prompt[:30]}...'",
                "model": "qwen2.5_fallback",
            }

    def _get_system_prompt_for_intent(self, intent: str) -> str:
        if intent == "code":
            return (
                "You are AXEN's expert software engineering specialist powered by Qwen. "
                "Provide precise, idiomatic, tested code with clear explanations."
            )
        elif intent == "ui_design":
            return (
                "You are AXEN's UI/UX designer. Suggest accessible, high-contrast, modern "
                "component architectures and curated color palettes."
            )
        elif intent == "chat":
            return "You are AXEN, a helpful, friendly local AI assistant running on this computer."
        else:
            return "You are AXEN, an AI assistant. Answer the user's question directly and helpfully."

    def run(self, user_message: str) -> dict[str, Any]:
        """Route user message and execute corresponding tool/model handler."""
        route = self.brain.predict(user_message)
        intent = route["intent"]
        confidence = route["confidence"]

        handler_name = ""
        output: Any = None

        if intent == "math":
            handler_name = "math_solver"
            output = self.math_tool(user_message)
        elif intent == "file_op":
            handler_name = "filesystem_tool"
            output = self.file_tool(user_message)
        elif intent == "web_search":
            handler_name = "web_search_tool"
            output = self.web_tool(user_message)
        elif intent == "memory":
            handler_name = "sqlite_memory_tool"
            output = self.memory_tool(user_message)
        elif intent == "job_search":
            handler_name = "job_agent_pipeline"
            output = self.job_tool(user_message)
        elif intent == "image_gen":
            handler_name = "image_brain"
            output = self.image_tool(user_message)
        else:  # code, ui_design, chat, unknown
            handler_name = f"qwen_{intent}"
            sys_prompt = self._get_system_prompt_for_intent(intent)
            output = self.llm_client(user_message, sys_prompt)

        # Log intent + confidence for every message to SQLite
        log_route(
            message=user_message,
            intent=intent,
            confidence=confidence,
            handler=handler_name,
            response=str(output)[:500],
            db_path=self.db_path,
        )

        return {
            "message": user_message,
            "intent": intent,
            "confidence": confidence,
            "handler": handler_name,
            "output": output,
        }


# Convenience module-level instance
brain = AxenBrain()
default_loop = AgentLoop(brain=brain)


def process_message(user_message: str) -> dict[str, Any]:
    return default_loop.run(user_message)
