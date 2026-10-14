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

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.ui.backend_service import AxenBackendService
from app.ai.model_manager import ModelManager

# Keep selection and loaded providers alive across requests.
model_manager = ModelManager()

logger = logging.getLogger("axen.ui")
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Ensure static subdirectories exist
(STATIC_DIR / "css").mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "js").mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize singleton instance on startup
    service = AxenBackendService.get_instance()
    # DO NOT start model load in background; ModelManager will lazy-load via ModelRouter.
    from axen.evolution.learning_worker import get_worker
    worker = get_worker()
    from axen.evolution.background_agent import get_background_agent
    background = get_background_agent(worker)
    background.busy = lambda: service._chat_lock.locked() or service._foreground_requests > 0
    background.start()
    if worker.config.get("autostart") and not worker.emergency:
        worker.command("start", "LEARNING", offline_demo=worker.config["offline_demo"])
    try:
        yield
    finally:
        background.shutdown()
        worker.shutdown()
        import axen.evolution.learning_worker as learning_runtime
        learning_runtime._worker = None

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

from pydantic import BaseModel, Field

# Evolution Engine Router
try:
    from axen.evolution.api import router as evolution_router
    app.include_router(evolution_router)
    from axen.evolution.learning_api import router as learning_router
    app.include_router(learning_router)
except Exception as _e:
    logging.getLogger("axen.ui").warning("Evolution API router could not be loaded: %s", _e)

# Mount static assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Mount generated image outputs
OUTPUTS_IMAGES_DIR = Path("outputs/images")
OUTPUTS_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(OUTPUTS_IMAGES_DIR)), name="images")


class ImageApiRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    enhance: bool = True
    preset: str = "turbo"


@app.post("/api/image")
async def generate_image_endpoint(req: ImageApiRequest) -> Dict[str, Any]:
    """Generate image using local diffusion pipeline."""
    from fastapi.concurrency import run_in_threadpool
    from image_brain import ImageBrain

    def _execute():
        brain = ImageBrain(preset=req.preset)
        return brain.generate(req.prompt, enhance=req.enhance)

    try:
        res = await run_in_threadpool(_execute)
        paths = res.get("paths", [])
        first_path = paths[0] if paths else ""
        filename = Path(first_path).name if first_path else ""
        url = f"/images/{filename}" if filename else ""
        return {
            "prompt": res.get("prompt", req.prompt),
            "url": url,
            "path": first_path,
        }
    except Exception as e:
        logger.error("Image generation API error: %s", e)
        raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")



# ---------------------------------------------------------------------------
# Unified Model Management Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/models")
def get_models() -> Dict[str, Any]:
    """Return catalog of local open-weight and configured cloud models."""
    from app.ai.model_manager import ModelManager
    manager = model_manager
    return {
        "active_model": manager.active_model_name,
        "models": manager.list_models(),
    }


class SelectModelRequest(BaseModel):
    model_id: str = Field(..., min_length=1)


@app.get("/api/brain")
@app.post("/api/sleep")
@app.post("/api/feedback")
@app.get("/api/reverse-engineering/status")
@app.post("/api/reverse-engineering")
async def node_feature_proxy(request: Request):
    """Forward Node-owned features to their existing implementation."""
    import httpx
    origin = request.headers.get("origin")
    if request.method == "POST" and origin and origin != str(request.base_url).rstrip("/"):
        raise HTTPException(status_code=403, detail="Open this feature from AEXN to continue")
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.request(
                request.method, "http://127.0.0.1:8001" + request.url.path,
                content=await request.body(),
                headers={"Content-Type": request.headers.get("content-type", "application/json")},
            )
        return JSONResponse(status_code=response.status_code, content=response.json())
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Node feature service is unavailable on port 8001")


@app.post("/api/models/select")
def select_model(req: SelectModelRequest) -> Dict[str, Any]:
    """Select the active model runtime."""
    from app.ai.model_manager import ModelManager
    manager = model_manager
    if req.model_id not in {m["id"] for m in manager.list_models()}:
        raise HTTPException(status_code=400, detail="Unknown model ID")
    success = manager.switch_model(req.model_id)
    return {
        "success": success,
        "active_model": manager.active_model_name,
        "message": f"Active model switched to {req.model_id}",
    }


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


@app.get("/vision", response_class=HTMLResponse)
@app.get("/vision.html", response_class=HTMLResponse)
def serve_vision() -> str:
    """Serve the dedicated Axen Vision Image Generation Studio."""
    vision_file = TEMPLATES_DIR / "vision.html"
    if not vision_file.exists():
        raise HTTPException(status_code=404, detail="Vision template not found")
    return vision_file.read_text(encoding="utf-8")


@app.get("/reverse-engineering", response_class=HTMLResponse)
def serve_reverse_engineering() -> str:
    """Serve the reverse engineering dashboard."""
    rev_file = TEMPLATES_DIR / "reverse-engineering.html"
    if not rev_file.exists():
        raise HTTPException(status_code=404, detail="Reverse engineering template not found")
    return rev_file.read_text(encoding="utf-8")


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
    project_id = payload.get("project_id")
    request_id = payload.get("request_id")
    attachments = payload.get("attachments") or []
    if not message:
        return {"status": "error", "summary": "A message is required."}

    think_mode = payload.get("think_mode") or payload.get("mode")
    service = AxenBackendService.get_instance()
    result = service.chat(
        message=message,
        session_id=session_id,
        project_id=project_id,
        request_id=request_id,
        attachments=attachments,
        think_mode=think_mode,
    )
    return result


@app.post("/api/v2/chat")
async def chat_v2_endpoint(request: Request):
    """Unified AEXN chat: automatic capability routing and public progress events."""
    body = await request.json()
    messages = body.get("messages", [])
    if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict):
        raise HTTPException(status_code=400, detail="messages array is required")
    message = str(messages[-1].get("content") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="A message is required")

    async def events():
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        public = {"READING": "Reading project files", "PLANNING": "Preparing a plan",
                  "GENERATING": "Preparing your response", "VERIFYING": "Checking results",
                  "UNDERSTANDING": "Understanding your request", "USING_TOOL": "Working on your request"}
        def progress(state, text, meta=None):
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "stage", "data": public.get(state, "Working on your request")})
        try:
            yield "data: " + json.dumps({"type": "stage", "data": "Understanding your request"}) + "\n\n"
            service = AxenBackendService.get_instance()
            task = asyncio.create_task(asyncio.to_thread(service.run_foreground, service.unified_response, message, model_manager,
                current_file=body.get("current_file"), attachments=body.get("attachments") or [],
                session_id=body.get("session_id"), project_id=body.get("project_id"),
                persist_messages=not body.get("session_messages_managed", False), state_callback=progress))
            while not task.done():
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.1)
                    yield "data: " + json.dumps(event) + "\n\n"
                except asyncio.TimeoutError:
                    continue
            result = await task
            while not queue.empty():
                yield "data: " + json.dumps(queue.get_nowait()) + "\n\n"
            answer = result.get("response", "")
            if not answer:
                raise RuntimeError("AEXN returned no response")
            if result.get("coding_result"):
                yield "data: " + json.dumps({"type": "workspace", "data": result["coding_result"]}) + "\n\n"
            for word in answer.split(" ")[:-1]:
                yield "data: " + json.dumps({"type": "token", "data": word + " "}) + "\n\n"
                await asyncio.sleep(0)
            yield "data: " + json.dumps({"type": "token", "data": answer.split(" ")[-1]}) + "\n\n"
            yield "data: " + json.dumps({"type": "done", "data": answer}) + "\n\n"
        except Exception as error:
            logger.exception("Unified chat failed")
            yield "data: " + json.dumps({"type": "error", "data": str(error)}) + "\n\n"
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.api_route("/api/chat/stream", methods=["GET", "POST"])
async def chat_stream(
    request: Request,
    message: Optional[str] = Query(None, description="User prompt message"),
    session_id: Optional[str] = Query(None, description="Active session ID"),
    project_id: Optional[str] = Query(None, description="Project ID"),
    request_id: Optional[str] = Query(None, description="Unique request ID for cancellation"),
):
    """
    Server-Sent Events (SSE) streaming endpoint.
    Emits real-time agent execution events ('UNDERSTANDING', 'PLANNING', 'THINKING',
    'USING_TOOL', 'VERIFYING', 'GENERATING', 'COMPLETED', 'ERROR') and the final answer
    without exposing internal chain-of-thought.
    """
    attachments = []
    think_mode = request.query_params.get("think_mode") or request.query_params.get("mode")
    if request.method == "POST":
        try:
            body = await request.json()
            message = str(body.get("message") or body.get("goal") or message or "").strip()
            session_id = body.get("session_id") or session_id
            project_id = body.get("project_id") or project_id
            request_id = body.get("request_id") or request_id
            attachments = body.get("attachments") or []
            think_mode = body.get("think_mode") or body.get("mode") or think_mode
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
                project_id=project_id,
                request_id=req_id,
                attachments=attachments,
                state_callback=_state_callback,
                cancel_event=cancel_event,
                think_mode=think_mode,
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
    project_id = (payload or {}).get("project_id")
    service = AxenBackendService.get_instance()
    new_id = service.new_session(title=title, project_id=project_id)
    return {
        "status": "ok",
        "session_id": new_id,
        "project_id": project_id,
        "title": title,
    }


@app.get("/api/sessions")
def list_sessions(
    q: Optional[str] = Query(None, description="Search query"),
    project_id: Optional[str] = Query(None, description="Project ID filter"),
) -> Dict[str, Any]:
    """List saved conversations filtered by search query or project_id."""
    service = AxenBackendService.get_instance()
    sessions = service.list_sessions(search=q, project_id=project_id)
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


@app.post("/api/sessions/{session_id}/messages")
def append_session_message(session_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Persist frontend-owned streaming turns in the existing conversation store."""
    from datetime import datetime
    service = AxenBackendService.get_instance()
    role = payload.get("role")
    content = str(payload.get("content") or "").strip()
    if role not in ("user", "assistant") or not content:
        raise HTTPException(status_code=400, detail="A user/assistant message is required")
    with service._chat_lock:
        conv = service.conversations.get(session_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        conv["messages"].append({"id": str(uuid.uuid4()), "role": role,
                                 "content": content, "timestamp": datetime.now().isoformat()})
        conv["updated_at"] = datetime.now().isoformat()
        service._save_conversations_to_disk()
    return {"status": "ok"}


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


# ---------------------------------------------------------------------------
# Phase 3 Memory Management APIs (Step 14 & 15)
# ---------------------------------------------------------------------------

@app.get("/api/memory")
def get_memories() -> Dict[str, Any]:
    """List all active persistent memories."""
    service = AxenBackendService.get_instance()
    return {"memories": service.list_memories()}


@app.post("/api/memory")
def add_memory(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Add or update a persistent memory."""
    key = str(payload.get("key", "")).strip()
    value = str(payload.get("value", "")).strip()
    mtype = str(payload.get("memory_type", "fact")).strip()
    if not key or not value:
        raise HTTPException(status_code=400, detail="Both 'key' and 'value' are required.")
    service = AxenBackendService.get_instance()
    return service.add_memory(key=key, value=value, memory_type=mtype)


@app.delete("/api/memory/{item_id}")
def delete_memory(item_id: int) -> Dict[str, Any]:
    """Deactivate or remove a memory by ID."""
    service = AxenBackendService.get_instance()
    ok = service.delete_memory(item_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Memory item not found or could not be deactivated.")
    return {"status": "ok", "message": f"Memory {item_id} deactivated."}


@app.post("/api/memory/clear")
def clear_memories() -> Dict[str, Any]:
    """Clear all active user memories."""
    service = AxenBackendService.get_instance()
    service.clear_memories()
    return {"status": "ok", "message": "All memories cleared."}


@app.get("/api/memory/export")
def export_memories() -> Dict[str, Any]:
    """Export all user memories as JSON."""
    service = AxenBackendService.get_instance()
    return service.export_memories()


@app.post("/api/memory/import")
def import_memories(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Import memories from exported JSON dictionary."""
    service = AxenBackendService.get_instance()
    imported_count = service.import_memories(payload)
    return {"status": "ok", "imported_count": imported_count}


# ---------------------------------------------------------------------------
# Knowledge & RAG Endpoints (Step 17 & 25)
# ---------------------------------------------------------------------------

@app.get("/api/knowledge/collections")
def list_knowledge_collections() -> Dict[str, Any]:
    """List all knowledge collections."""
    service = AxenBackendService.get_instance()
    collections = service.list_knowledge_collections()
    return {"status": "ok", "collections": collections}


@app.get("/api/knowledge/documents")
def list_knowledge_documents(collection_id: Optional[str] = None) -> Dict[str, Any]:
    """List indexed documents, optionally filtered by collection."""
    service = AxenBackendService.get_instance()
    docs = service.list_knowledge_documents(collection_id=collection_id)
    return {"status": "ok", "documents": docs}


@app.post("/api/knowledge/ingest")
def ingest_knowledge_document(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Ingest a document into the knowledge base."""
    service = AxenBackendService.get_instance()
    file_path = payload.get("file_path", "")
    collection_id = payload.get("collection_id", "default")
    if not file_path:
        raise HTTPException(status_code=400, detail="Missing 'file_path'")
    result = service.ingest_knowledge_file(file_path, collection_id=collection_id)
    return {"status": "ok", "result": result}


@app.delete("/api/knowledge/documents/{document_id}")
def delete_knowledge_document(document_id: str) -> Dict[str, Any]:
    """Delete a document from the knowledge base."""
    service = AxenBackendService.get_instance()
    success = service.delete_knowledge_document(document_id)
    return {"status": "ok", "deleted": success}


@app.post("/api/knowledge/clear")
def clear_knowledge_collection(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Clear all documents in a collection."""
    service = AxenBackendService.get_instance()
    collection_id = payload.get("collection_id", "default")
    count = service.clear_knowledge_collection(collection_id)
    return {"status": "ok", "cleared_count": count}


@app.post("/api/knowledge/search")
def search_knowledge(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Hybrid search over indexed knowledge chunks."""
    service = AxenBackendService.get_instance()
    query = payload.get("query", "")
    collection_id = payload.get("collection_id")
    top_k = payload.get("top_k", 4)
    results = service.search_knowledge(query, collection_id=collection_id, top_k=top_k)
    return {"status": "ok", "results": results}


# ---------------------------------------------------------------------------
# File Upload & Attachment Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/files/upload")
async def upload_files_endpoint(
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
) -> Dict[str, Any]:
    """Upload one or multiple files and extract their contents."""
    service = AxenBackendService.get_instance()
    to_process: List[UploadFile] = []
    if file:
        to_process.append(file)
    if files:
        to_process.extend(files)

    if not to_process:
        raise HTTPException(status_code=400, detail="No files provided.")

    from axen.storage.storage_models import UploadConfig
    cfg = UploadConfig()

    if len(to_process) > cfg.max_files_per_message:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files ({len(to_process)}). Maximum allowed is {cfg.max_files_per_message} files.",
        )

    file_payloads = []
    total_batch_bytes = 0
    for f in to_process:
        content = await f.read()
        total_batch_bytes += len(content)
        file_payloads.append((f, content))

    if total_batch_bytes > cfg.max_total_upload_bytes:
        max_mb = cfg.max_total_upload_bytes // (1024 * 1024)
        raise HTTPException(
            status_code=400,
            detail=f"Total upload size ({total_batch_bytes / (1024 * 1024):.1f} MB) exceeds maximum allowed limit ({max_mb} MB).",
        )

    uploaded_results = []
    has_rejected = False
    rejection_msg = ""
    for f, content in file_payloads:
        try:
            res = service.upload_file(f.filename or "unnamed", content, mime_type=f.content_type)
            uploaded_results.append(res)
        except Exception as exc:
            has_rejected = True
            rejection_msg = str(exc)
            uploaded_results.append({
                "file_id": "",
                "filename": f.filename or "unnamed",
                "status": "failed",
                "error": str(exc),
            })

    if has_rejected and len(to_process) == 1:
        raise HTTPException(status_code=400, detail=rejection_msg)

    first = uploaded_results[0] if uploaded_results else {}
    return {
        "status": "ok",
        "count": len(uploaded_results),
        "files": uploaded_results,
        "file_id": first.get("file_id"),
        "original_filename": first.get("filename") or first.get("original_filename"),
        "category": first.get("category"),
        "size": first.get("size") or first.get("size_bytes"),
    }


@app.get("/api/files")
def list_files_endpoint() -> Dict[str, Any]:
    """List all stored uploaded files."""
    service = AxenBackendService.get_instance()
    return {"status": "ok", "files": service.list_uploaded_files()}


@app.get("/api/files/{file_id}")
def get_file_endpoint(file_id: str) -> Dict[str, Any]:
    """Get metadata and extracted text for an uploaded file."""
    service = AxenBackendService.get_instance()
    fobj = service.get_uploaded_file(file_id)
    if not fobj:
        raise HTTPException(status_code=404, detail="File not found")
    return {"status": "ok", "file": fobj}


@app.delete("/api/files/{file_id}")
def delete_file_endpoint(file_id: str) -> Dict[str, Any]:
    """Delete an uploaded file from disk and metadata index."""
    service = AxenBackendService.get_instance()
    ok = service.delete_uploaded_file(file_id)
    if not ok:
        raise HTTPException(status_code=404, detail="File not found")
    return {"status": "ok", "deleted": True}


@app.post("/api/files/{file_id}/ingest-knowledge")
def ingest_file_to_knowledge_endpoint(file_id: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Ingest an uploaded file into persistent KnowledgeManager."""
    coll = (payload or {}).get("collection", (payload or {}).get("collection_id", "default"))
    service = AxenBackendService.get_instance()
    res = service.ingest_uploaded_file_to_knowledge(file_id, collection_id=coll)
    return {
        "status": "success" if res.get("success") else "error",
        "result": res,
        "chunk_count": res.get("chunk_count") or res.get("chunks_created", 1),
    }


# ---------------------------------------------------------------------------
# Projects / Workspaces System APIs
# ---------------------------------------------------------------------------

@app.get("/api/projects")
def list_projects_endpoint(
    include_archived: bool = Query(False, description="Include archived projects"),
    q: Optional[str] = Query(None, description="Search query"),
) -> Dict[str, Any]:
    """List all projects."""
    service = AxenBackendService.get_instance()
    projects = service.list_projects(include_archived=include_archived, search=q)
    return {"status": "ok", "projects": projects}


@app.post("/api/projects")
def create_project_endpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new project workspace."""
    name = str(payload.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Project name is required.")
    description = str(payload.get("description") or "").strip()
    instructions = str(payload.get("instructions") or "").strip()
    icon = str(payload.get("icon") or "folder").strip()
    settings = payload.get("settings")

    service = AxenBackendService.get_instance()
    try:
        project = service.create_project(
            name=name,
            description=description,
            instructions=instructions,
            icon=icon,
            settings=settings,
        )
        res = dict(project)
        res["status"] = "ok"
        res["project"] = project
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/projects/{project_id}")
def get_project_endpoint(project_id: str) -> Dict[str, Any]:
    """Get project details."""
    service = AxenBackendService.get_instance()
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    res = dict(project)
    res["status"] = "ok"
    res["project"] = project
    return res


@app.patch("/api/projects/{project_id}")
def update_project_endpoint(project_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Update project details or settings."""
    service = AxenBackendService.get_instance()
    updated = service.update_project(project_id, payload)
    if not updated:
        raise HTTPException(status_code=404, detail="Project not found")
    res = dict(updated)
    res["status"] = "ok"
    res["project"] = updated
    return res


@app.delete("/api/projects/{project_id}")
def delete_project_endpoint(
    project_id: str,
    delete_physical_files: bool = Query(False, description="Also delete physical files"),
) -> Dict[str, Any]:
    """Delete a project workspace."""
    service = AxenBackendService.get_instance()
    ok = service.delete_project(project_id, delete_physical_files=delete_physical_files)
    if not ok:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"status": "ok", "deleted": True}


@app.post("/api/projects/{project_id}/archive")
def archive_project_endpoint(project_id: str) -> Dict[str, Any]:
    """Archive a project workspace."""
    service = AxenBackendService.get_instance()
    ok = service.archive_project(project_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"status": "ok", "archived": True}


@app.post("/api/projects/{project_id}/restore")
def restore_project_endpoint(project_id: str) -> Dict[str, Any]:
    """Restore an archived project workspace."""
    service = AxenBackendService.get_instance()
    ok = service.restore_project(project_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"status": "ok", "restored": True}


@app.get("/api/projects/{project_id}/chats")
def list_project_chats_endpoint(project_id: str) -> Dict[str, Any]:
    """List conversations belonging to a project."""
    service = AxenBackendService.get_instance()
    proj = service.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    chats = service.list_sessions(project_id=project_id)
    return {"status": "ok", "chats": chats}


@app.post("/api/projects/{project_id}/chats")
def create_project_chat_endpoint(project_id: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a new chat conversation inside a project."""
    service = AxenBackendService.get_instance()
    proj = service.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    title = (payload or {}).get("title", f"Chat in {proj['name']}")
    new_id = service.new_session(title=title, project_id=project_id)
    return {"status": "ok", "session_id": new_id, "project_id": project_id, "title": title}


@app.get("/api/projects/{project_id}/memory")
def list_project_memory_endpoint(project_id: str) -> Dict[str, Any]:
    """List scoped memories for a project."""
    service = AxenBackendService.get_instance()
    proj = service.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    memories_list = service.list_project_memories(project_id)
    memories_dict = {m["key"]: m["value"] for m in memories_list if isinstance(m, dict) and "key" in m}
    return {"status": "ok", "memories": memories_dict, "memories_list": memories_list}


@app.post("/api/projects/{project_id}/memory")
def add_project_memory_endpoint(project_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Add a scoped memory to a project."""
    key = str(payload.get("key") or "").strip()
    value = str(payload.get("value") or "").strip()
    memory_type = str(payload.get("memory_type") or "fact")
    if not key or not value:
        raise HTTPException(status_code=400, detail="Key and value are required.")
    service = AxenBackendService.get_instance()
    proj = service.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    res = service.add_project_memory(project_id, key=key, value=value, memory_type=memory_type)
    return {"status": "ok", "result": res}


@app.delete("/api/projects/{project_id}/memory/{key}")
def delete_project_memory_endpoint(project_id: str, key: str) -> Dict[str, Any]:
    """Delete a scoped memory from a project."""
    service = AxenBackendService.get_instance()
    proj = service.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    ok = service.delete_project_memory(project_id, key)
    return {"status": "ok", "deleted": ok}


@app.get("/api/projects/{project_id}/files")
def list_project_files_endpoint(project_id: str) -> Dict[str, Any]:
    """List files linked to a project."""
    service = AxenBackendService.get_instance()
    proj = service.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    files = service.list_project_files(project_id)
    return {"status": "ok", "files": files}


@app.post("/api/projects/{project_id}/files")
def add_project_file_endpoint(project_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Associate an uploaded file with a project."""
    file_id = str(payload.get("file_id") or "").strip()
    if not file_id:
        raise HTTPException(status_code=400, detail="file_id is required.")
    service = AxenBackendService.get_instance()
    proj = service.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    ok = service.add_project_file(project_id, file_id)
    return {"status": "ok", "associated": ok}


@app.delete("/api/projects/{project_id}/files/{file_id}")
def remove_project_file_endpoint(
    project_id: str,
    file_id: str,
    delete_physical: bool = Query(False, description="Also delete physical file"),
) -> Dict[str, Any]:
    """Unlink file from project and optionally delete physical file."""
    service = AxenBackendService.get_instance()
    proj = service.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    ok = service.remove_project_file(project_id, file_id, delete_physical=delete_physical)
    return {"status": "ok", "removed": ok}


@app.get("/api/projects/{project_id}/knowledge")
def list_project_knowledge_endpoint(project_id: str) -> Dict[str, Any]:
    """List knowledge collections associated with a project."""
    service = AxenBackendService.get_instance()
    proj = service.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    colls = service.list_project_knowledge(project_id)
    return {"status": "ok", "collections": colls}


@app.post("/api/projects/{project_id}/knowledge")
def associate_project_knowledge_endpoint(project_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Associate a knowledge collection with a project."""
    collection_id = str(payload.get("collection_id") or "").strip()
    if not collection_id:
        raise HTTPException(status_code=400, detail="collection_id is required.")
    service = AxenBackendService.get_instance()
    proj = service.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    ok = service.associate_project_knowledge(project_id, collection_id)
    return {"status": "ok", "associated": ok}


# ---------------------------------------------------------------------------
# Memory 2.0 Management APIs
# ---------------------------------------------------------------------------

@app.get("/api/memory")
def get_memories_endpoint(
    scope: Optional[str] = Query(None, description="Scope filter ('global' or 'project')"),
    project_id: Optional[str] = Query(None, description="Project ID filter"),
    q: Optional[str] = Query(None, description="Search keyword"),
) -> Dict[str, Any]:
    """List or search memories across global and project scopes."""
    service = AxenBackendService.get_instance()
    memories = service.list_memories(scope=scope, project_id=project_id, search=q)
    return {"status": "ok", "memories": memories, "count": len(memories)}


@app.post("/api/memory")
def create_memory_endpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Add a new memory in the specified scope."""
    key = str(payload.get("key") or "").strip()
    value = str(payload.get("value") or "").strip()
    scope = str(payload.get("scope") or "global").strip()
    project_id = payload.get("project_id")
    memory_type = str(payload.get("memory_type") or "fact").strip()

    if not key or not value:
        raise HTTPException(status_code=400, detail="key and value are required.")

    service = AxenBackendService.get_instance()
    res = service.add_memory(key=key, value=value, scope=scope, project_id=project_id, memory_type=memory_type)
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("error", "Memory creation rejected"))
    return {"status": "ok", "memory": res.get("memory")}


@app.delete("/api/memory")
def delete_memory_endpoint(
    key: str = Query(..., description="Key of memory to delete"),
    scope: Optional[str] = Query(None, description="Scope of memory"),
    project_id: Optional[str] = Query(None, description="Project ID if project-scoped"),
) -> Dict[str, Any]:
    """Delete a memory item by key and optional scope."""
    service = AxenBackendService.get_instance()
    ok = service.delete_memory(key, scope=scope, project_id=project_id)
    return {"status": "ok", "deleted": ok}


@app.post("/api/memory/clear")
def clear_memories_endpoint(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Clear memories within a given scope or project."""
    data = payload or {}
    scope = data.get("scope")
    project_id = data.get("project_id")
    service = AxenBackendService.get_instance()
    ok = service.clear_memories(scope=scope, project_id=project_id)
    return {"status": "ok", "cleared": ok}


# ---------------------------------------------------------------------------
# Phase 5: Codex-Style Coding Workspace Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/coding/workspace")
def get_coding_workspace() -> Dict[str, Any]:
    """Get metadata for the active coding workspace."""
    from axen.coding.coding_service import CodingService
    service = CodingService.get_instance()
    return {"status": "ok", "workspace": service.get_workspace_info()}


@app.post("/api/coding/workspace")
def set_coding_workspace(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Switch active coding workspace directory with strict validation."""
    target_path = str(payload.get("path") or "").strip()
    if not target_path:
        raise HTTPException(status_code=400, detail="Workspace path is required.")

    perm_str = str(payload.get("permission") or "EXECUTE").upper()
    from axen.coding.coding_service import CodingService
    from axen.coding.permissions import PermissionLevel

    try:
        level = PermissionLevel(perm_str)
    except Exception:
        level = PermissionLevel.EXECUTE

    service = CodingService.get_instance()
    try:
        info = service.set_workspace(target_path, permission_level=level)
        return {"status": "ok", "workspace": info}
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))
    except (FileNotFoundError, NotADirectoryError) as fe:
        raise HTTPException(status_code=400, detail=str(fe))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/coding/tree")
def get_coding_file_tree(max_depth: int = Query(4, ge=1, le=8)) -> Dict[str, Any]:
    """Return recursive file tree for workspace explorer."""
    from axen.coding.coding_service import CodingService
    service = CodingService.get_instance()
    tree = service.get_file_tree(max_depth=max_depth)
    return {"status": "ok", "tree": tree}


@app.get("/api/coding/file")
def read_coding_file(path: str = Query(..., description="Relative path to file in workspace")) -> Dict[str, Any]:
    """Read a workspace file safely with boundary and size checks."""
    from axen.coding.coding_service import CodingService
    service = CodingService.get_instance()
    try:
        file_data = service.read_file(path)
        return {"status": "ok", "file": file_data}
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/coding/file/save")
def save_coding_file(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Save file with AST syntax validation and automatic backup."""
    rel_path = str(payload.get("path") or "").strip()
    content = payload.get("content", "")
    validate_syntax = bool(payload.get("validate_syntax", True))

    if not rel_path:
        raise HTTPException(status_code=400, detail="File path is required.")

    from axen.coding.coding_service import CodingService
    service = CodingService.get_instance()
    try:
        res = service.save_file(rel_path, content, validate_syntax=validate_syntax)
        if not res.get("success"):
            return {"status": "error", "result": res}
        return {"status": "ok", "result": res}
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/coding/search")
def search_coding_workspace(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Search code, symbols, or files across the active workspace."""
    query = str(payload.get("query") or "").strip()
    search_type = str(payload.get("type") or "text").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Search query is required.")

    from axen.coding.coding_service import CodingService
    service = CodingService.get_instance()
    matches = []
    if search_type == "file":
        files = service.searcher.search_files(query)
        matches = [{"file": f, "line": 0, "text": f} for f in files]
    elif search_type == "symbol":
        syms = service.searcher.search_symbols(query)
        matches = [
            {"file": s.file_path, "line": s.line_number, "text": f"{s.symbol_type}: {s.symbol_name}"}
            for s in syms
        ]
    else:
        text_matches = service.searcher.search_text(query)
        matches = [
            {"file": m.file_path, "line": m.line_number, "text": m.line_content}
            for m in text_matches
        ]

    return {"status": "ok", "query": query, "type": search_type, "matches": matches, "count": len(matches)}


@app.get("/api/coding/git/status")
def get_coding_git_status() -> Dict[str, Any]:
    """Get active Git status and changed files."""
    from axen.coding.coding_service import CodingService
    service = CodingService.get_instance()
    status = service.get_git_status()
    return {"status": "ok", "git": status}


@app.get("/api/coding/git/diff")
def get_coding_git_diff(file_path: Optional[str] = Query(None, description="Optional relative file path")) -> Dict[str, Any]:
    """Get git diff for workspace or specific file."""
    from axen.coding.coding_service import CodingService
    service = CodingService.get_instance()
    diff = service.get_diff(file_path)
    return {"status": "ok", "diff": diff, "file_path": file_path}


@app.post("/api/coding/patch/preview")
def preview_coding_patch(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and preview a patch diff without writing to disk."""
    file_path = str(payload.get("file_path") or "").strip()
    target = str(payload.get("target") or "")
    replacement = str(payload.get("replacement") or "")

    if not file_path or not target:
        raise HTTPException(status_code=400, detail="file_path and target snippet are required.")

    from axen.coding.coding_service import CodingService
    service = CodingService.get_instance()
    res = service.propose_patch(file_path, target, replacement)
    return {"status": "ok", "patch": res}


@app.post("/api/coding/patch/apply")
def apply_coding_patch(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Apply an approved patch with syntax validation and automatic backup."""
    file_path = str(payload.get("file_path") or "").strip()
    target = str(payload.get("target") or "")
    replacement = str(payload.get("replacement") or "")
    reason = str(payload.get("reason") or "User applied patch from Coding Workspace")

    if not file_path or not target:
        raise HTTPException(status_code=400, detail="file_path and target snippet are required.")

    from axen.coding.coding_service import CodingService
    service = CodingService.get_instance()
    res = service.apply_patch(file_path, target, replacement, reason=reason)
    return {"status": "ok" if res.get("success") else "error", "result": res}


@app.post("/api/coding/undo")
def undo_coding_file(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Rollback a modified file to backup or git head."""
    file_path = str(payload.get("file_path") or "").strip()
    if payload.get("approved") is not True:
        raise HTTPException(status_code=403, detail="Rollback requires explicit approval")
    if not file_path:
        raise HTTPException(status_code=400, detail="file_path is required.")

    from axen.coding.coding_service import CodingService
    service = CodingService.get_instance()
    res = service.rollback_file(file_path)
    return {"status": "ok" if res.get("success") else "error", "result": res}


@app.post("/api/coding/test")
def run_coding_tests(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run tests in the workspace and return parsed diagnostics."""
    target = (payload or {}).get("target")
    if (payload or {}).get("approved_execution") is not True:
        raise HTTPException(status_code=403, detail="Test execution requires explicit approval")
    from axen.coding.coding_service import CodingService
    service = CodingService.get_instance()
    try:
        res = service.run_tests(target=target, sandboxed=(payload or {}).get("trusted_project") is not True)
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error))
    return {"status": "ok", "test_run": res}


@app.post("/api/coding/chat")
def run_coding_chat_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a coding task using Think Engine 2.0 modes (FAST, BALANCED, DEEP).
    """
    message = str(payload.get("message") or payload.get("goal") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message/goal is required.")

    think_mode = str(payload.get("think_mode") or payload.get("mode") or "balanced").strip()
    from axen.coding.coding_service import CodingService
    service = CodingService.get_instance()

    events: List[Dict[str, Any]] = []

    def _state_cb(state: str, text: str, meta: Optional[Dict[str, Any]] = None):
        events.append({"state": state, "text": text, "meta": meta or {}})

    if "run" in message.lower() and "test" in message.lower():
        raise HTTPException(status_code=403, detail="Approve test execution using Run tests")
    try:
        res = service.propose_task(message, model_manager.get_model("router"), payload.get("current_file"), _state_cb)
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error))
    return {"status": "ok", "task_result": res, "timeline": events}


@app.post("/api/coding/proposals/{proposal_id}/apply")
def apply_coding_proposal(proposal_id: str, payload: Dict[str, Any]):
    if payload.get("approved") is not True:
        raise HTTPException(status_code=403, detail="Review and approve the proposed changes first")
    from axen.coding.coding_service import CodingService
    try:
        return {"status": "ok", "result": CodingService.get_instance().apply_proposal(proposal_id)}
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error))
    except (ValueError, OSError) as error:
        raise HTTPException(status_code=409, detail=str(error))


@app.post("/api/coding/git/checkpoint")
def create_coding_checkpoint(payload: Dict[str, Any]):
    if payload.get('approved') is not True:
        raise HTTPException(status_code=403, detail='Git checkpoint creation requires approval')
    from axen.coding.coding_service import CodingService
    try:
        return CodingService.get_instance().create_git_checkpoint()
    except (ValueError, OSError) as error:
        raise HTTPException(status_code=409, detail=str(error))


@app.get("/api/coding/git/checkpoint/{token}/preview")
def preview_coding_checkpoint(token: str):
    from axen.coding.coding_service import CodingService
    try:
        return CodingService.get_instance().preview_git_rollback(token)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error))


@app.post("/api/coding/git/checkpoint/{token}/restore")
def restore_coding_checkpoint(token: str, payload: Dict[str, Any]):
    if payload.get('approved') is not True:
        raise HTTPException(status_code=403, detail='Rollback requires approval after preview')
    from axen.coding.coding_service import CodingService
    try:
        return CodingService.get_instance().restore_git_checkpoint(token)
    except (ValueError, OSError) as error:
        raise HTTPException(status_code=409, detail=str(error))




