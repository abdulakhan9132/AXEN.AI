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

import asyncio
import queue
import uuid
import threading

@app.post("/api/chat")
async def chat_endpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standard chat endpoint.
    Compatible with both modern `message` and legacy `goal` inputs.
    """
    message = str(payload.get("message") or payload.get("goal") or "").strip()
    session_id = payload.get("session_id")
    request_id = payload.get("request_id")
    if not message:
        return {"status": "error", "summary": "A message is required."}

    service = AxenBackendService.get_instance()
    result = service.chat(message=message, session_id=session_id, request_id=request_id)
    return result


@app.api_route("/api/chat/stream", methods=["GET", "POST"])
async def chat_stream(
    request: Request,
    message: Optional[str] = Query(None, description="User prompt message"),
    session_id: Optional[str] = Query(None, description="Active session ID"),
    request_id: Optional[str] = Query(None, description="Unique request ID for cancellation"),
):
    """
    Server-Sent Events (SSE) streaming endpoint.
    Emits real-time agent execution events ('UNDERSTANDING', 'PLANNING', 'THINKING',
    'USING_TOOL', 'VERIFYING', 'GENERATING', 'COMPLETED', 'ERROR') and the final answer
    without exposing internal chain-of-thought.
    """
    if request.method == "POST":
        try:
            body = await request.json()
            message = str(body.get("message") or body.get("goal") or message or "").strip()
            session_id = body.get("session_id") or session_id
            request_id = body.get("request_id") or request_id
        except Exception:
            pass

    if not message:
        raise HTTPException(status_code=400, detail="A message is required.")

    req_id = request_id or str(uuid.uuid4())
    service = AxenBackendService.get_instance()
    cancel_event = threading.Event()
    loop = asyncio.get_running_loop()
    event_queue: asyncio.Queue = asyncio.Queue()

    def _state_callback(state_name: str, display_text: str, meta: Optional[Dict[str, Any]] = None):
        evt = {
            "type": "state",
            "state": state_name,
            "text": display_text,
            "meta": meta or {},
            "request_id": req_id,
        }
        loop.call_soon_threadsafe(event_queue.put_nowait, evt)

    def _worker():
        try:
            res = service.chat(
                message=message,
                session_id=session_id,
                request_id=req_id,
                state_callback=_state_callback,
                cancel_event=cancel_event,
            )
            loop.call_soon_threadsafe(event_queue.put_nowait, {"type": "result", "result": res, "request_id": req_id})
        except Exception as exc:
            logger.exception("Chat worker error: %s", exc)
            loop.call_soon_threadsafe(event_queue.put_nowait, {
                "type": "error",
                "error": "Something went wrong while processing your request.",
                "detail": str(exc),
                "request_id": req_id,
            })
        finally:
            loop.call_soon_threadsafe(event_queue.put_nowait, {"type": "done", "request_id": req_id})

    worker_thread = threading.Thread(target=_worker, name=f"AxenWorker-{req_id[:8]}", daemon=True)
    worker_thread.start()

    async def event_generator() -> AsyncGenerator[str, None]:
        # Emit initial task metadata
        yield f"event: init\ndata: {json.dumps({'request_id': req_id, 'session_id': session_id})}\n\n"

        while True:
            try:
                item = await event_queue.get()
                itype = item.get("type")

                if itype == "state":
                    yield f"event: state\ndata: {json.dumps(item)}\n\n"
                elif itype == "result":
                    yield f"event: result\ndata: {json.dumps(item['result'])}\n\n"
                elif itype == "error":
                    yield f"event: error\ndata: {json.dumps(item)}\n\n"
                elif itype == "done":
                    yield "event: done\ndata: {}\n\n"
                    break
            except asyncio.CancelledError:
                cancel_event.set()
                service.cancel_request(req_id)
                break
            except Exception as e:
                logger.error("SSE generator error: %s", e)
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat/cancel")
async def cancel_chat(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cancel an active agent execution request.
    Does not stop or kill the AXEN server, only safely signals cancellation to the running task.
    """
    request_id = str(payload.get("request_id") or "").strip()
    if not request_id:
        raise HTTPException(status_code=400, detail="request_id is required.")

    service = AxenBackendService.get_instance()
    cancelled = service.cancel_request(request_id)
    return {
        "status": "ok" if cancelled else "not_found",
        "request_id": request_id,
        "cancelled": cancelled,
    }


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
    """Get complete message history for a specific conversation and switch active session."""
    service = AxenBackendService.get_instance()
    session = service.switch_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return session


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
