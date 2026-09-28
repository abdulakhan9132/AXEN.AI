"""
app/ui/app.py
=============
FastAPI application serving the modern AXEN AI Web Interface.

Connects to the local AXEN AssistantController without cloud dependencies.
Provides real-time chat, tool execution status, conversation sessions,
and static frontend delivery.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.ui.backend_service import AxenBackendService

logger = logging.getLogger("axen.ui")
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Ensure static subdirectories exist
(STATIC_DIR / "css").mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "js").mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Automatically start loading the local model into GPU memory on startup
    service = AxenBackendService.get_instance()
    service.start_model_load_in_background()
    yield


app = FastAPI(
    title="AXEN Local AI Assistant",
    version="3.0.0",
    description="Offline Local AI Assistant with PyTorch/CUDA and Agentic Reasoning",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# UI Root Route
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def serve_index() -> str:
    """Serve the primary single-page AXEN application."""
    index_file = TEMPLATES_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Index template not found")
    return index_file.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Chat & Real-Time Tool Execution
# ---------------------------------------------------------------------------

@app.post("/api/chat")
async def chat_endpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standard chat endpoint.
    Compatible with both modern `message` and legacy `goal` inputs.
    """
    message = str(payload.get("message") or payload.get("goal") or "").strip()
    session_id = payload.get("session_id")
    if not message:
        return {"status": "error", "summary": "A message is required."}

    service = AxenBackendService.get_instance()
    result = service.chat(message=message, session_id=session_id)
    return result


@app.get("/api/chat/stream")
async def chat_stream(
    message: str = Query(..., description="User prompt message"),
    session_id: Optional[str] = Query(None, description="Active session ID"),
):
    """
    Server-Sent Events (SSE) streaming endpoint.
    Emits real-time progress events ('Calculating...', 'Verifying result...')
    and the final answer without exposing internal chain-of-thought.
    """
    service = AxenBackendService.get_instance()

    async def event_generator() -> AsyncGenerator[str, None]:
        status_updates: list[str] = []

        def _on_status(status_text: str):
            status_updates.append(status_text)

        # 1. Initial acknowledgment event
        yield f"event: status\ndata: {json.dumps({'status': 'Thinking...', 'phase': 'understanding'})}\n\n"

        # 2. Run chat processing in a worker call
        # The backend service sets the status callback
        result = service.chat(
            message=message,
            session_id=session_id,
            status_callback=_on_status,
        )

        # 3. If a tool was used, report tool execution and verification
        if result.get("tool_used"):
            tool_status = result.get("tool_status", "Executing tool...")
            yield f"event: status\ndata: {json.dumps({'status': tool_status, 'phase': 'tool', 'tool': result.get('tool_name')})}\n\n"

            if result.get("verification"):
                yield f"event: status\ndata: {json.dumps({'status': 'Verifying result...', 'phase': 'verifying'})}\n\n"

        # 4. Final response event
        yield f"event: result\ndata: {json.dumps(result)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Conversation Sessions
# ---------------------------------------------------------------------------

@app.post("/api/sessions/new")
def new_session(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a new conversation session and reset controller short-term memory."""
    title = (payload or {}).get("title", "New Chat")
    service = AxenBackendService.get_instance()
    new_id = service.new_session(title=title)
    return {
        "status": "ok",
        "session_id": new_id,
        "title": title,
    }


@app.get("/api/sessions")
def list_sessions(q: Optional[str] = Query(None, description="Search query")) -> Dict[str, Any]:
    """List all saved conversations, with optional search filter."""
    service = AxenBackendService.get_instance()
    sessions = service.list_sessions(search=q)
    return {
        "sessions": sessions,
        "active_session_id": service.active_session_id,
    }


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> Dict[str, Any]:
    """Get complete message history for a specific conversation."""
    service = AxenBackendService.get_instance()
    if session_id not in service.conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return service.conversations[session_id]


@app.post("/api/sessions/{session_id}/switch")
def switch_session(session_id: str) -> Dict[str, Any]:
    """Switch active session."""
    service = AxenBackendService.get_instance()
    session = service.switch_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "ok", "session": session}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> Dict[str, Any]:
    """Delete a conversation."""
    service = AxenBackendService.get_instance()
    success = service.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "ok", "active_session_id": service.active_session_id}


@app.patch("/api/sessions/{session_id}")
def rename_session(session_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Rename a conversation."""
    title = str(payload.get("title", "")).strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    service = AxenBackendService.get_instance()
    success = service.rename_session(session_id, title)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "ok", "title": title}


# ---------------------------------------------------------------------------
# Diagnostics & System Controls
# ---------------------------------------------------------------------------

@app.get("/api/status")
def status() -> Dict[str, Any]:
    """System health, model loaded state, CUDA GPU device, and tools."""
    service = AxenBackendService.get_instance()
    return service.get_status()


@app.post("/api/model/load")
def load_model() -> Dict[str, Any]:
    """Trigger background model loading if not already loaded."""
    service = AxenBackendService.get_instance()
    service.start_model_load_in_background()
    return {
        "status": "loading_initiated",
        "model_loaded": service.controller.model_ready,
    }


# Backwards compatibility endpoints
@app.get("/api/health")
def health() -> Dict[str, Any]:
    service = AxenBackendService.get_instance()
    return {
        "status": "running",
        "model_available": service.controller.model_ready,
        "tools": service.controller.tools.names(),
    }


@app.get("/api/tools")
def tools() -> Dict[str, Any]:
    service = AxenBackendService.get_instance()
    return {
        "tools": [
            {"name": t["name"], "risk": t["risk"], "description": t["description"]}
            for t in service.controller.tools.list_tools()
        ]
    }
