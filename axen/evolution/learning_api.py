"""Local operator controls and resumable server-sent worker evidence."""
from __future__ import annotations

import asyncio
import json
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .connectors import AccountConnectorManager
from .learning_worker import MODES, get_worker


def operator_control(request: Request):
    """Local deployment only: CSRF boundary, not remote-user authentication."""
    origin = request.headers.get("origin")
    if origin:
        p = urlsplit(origin)
        if p.hostname not in ("localhost", "127.0.0.1", "::1", "testserver") or p.netloc != request.url.netloc:
            raise HTTPException(403, "Operator controls require same-origin local requests")
    if request.headers.get("X-AXEN-Control") != "operator":
        raise HTTPException(403, "Explicit operator control header required")


router = APIRouter(prefix="/api/autonomy", tags=["Autonomous learning"])


class ControlRequest(BaseModel):
    action: str
    mode: str | None = None
    offline_demo: bool = False


@router.get("/status")
def status():
    from .background_agent import get_background_agent
    worker = get_worker()
    return dict(worker.snapshot(), background_agent=get_background_agent(worker).snapshot(),
                answer_agent={"role": "answer_questions", "priority": "foreground"})


@router.get("/events")
def events(after: int = 0):
    return {"events": get_worker().store.events(max(0, after))}


@router.post("/goals/{goal_id}/retry", dependencies=[Depends(operator_control)])
def retry_goal(goal_id: str):
    worker = get_worker()
    with worker.lock:
        if worker.mode != "OFF": raise HTTPException(409, "Stop the session before retrying a goal")
        try: worker.store.retry(goal_id, worker.config["sources"])
        except ValueError as exc: raise HTTPException(409, str(exc)) from exc
    return worker.snapshot()


@router.post("/control", dependencies=[Depends(operator_control)])
def control(payload: ControlRequest):
    try:
        return get_worker().command(payload.action, payload.mode, offline_demo=payload.offline_demo)
    except (ValueError, RuntimeError) as exc: raise HTTPException(409, str(exc)) from exc


@router.get("/stream")
async def stream(request: Request, after: int = 0):
    try: cursor = max(after, int(request.headers.get("Last-Event-ID", "0")))
    except ValueError: raise HTTPException(400, "Invalid event cursor")
    async def generate():
        nonlocal cursor
        while not await request.is_disconnected():
            batch = get_worker().store.events(cursor)
            for event in batch:
                cursor = event["id"]
                yield f"id: {cursor}\ndata: {json.dumps(event)}\n\n"
            if not batch: yield ": heartbeat\n\n"
            await asyncio.sleep(1)
    return StreamingResponse(generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def connectors():
    worker = get_worker()
    if not hasattr(worker, "connectors"): worker.connectors = AccountConnectorManager(worker.store)
    return worker.connectors


class ConsentRequest(BaseModel):
    service: str
    scopes: list[str] = Field(default_factory=list)
    consent: bool = False


@router.get("/connectors")
def connector_status(): return connectors().status()


@router.post("/connectors/authorize", dependencies=[Depends(operator_control)])
def authorize(payload: ConsentRequest):
    try:
        connectors().authorize(payload.service, payload.scopes, consent=payload.consent)
        return connectors().status()
    except (ValueError, PermissionError, RuntimeError) as exc: raise HTTPException(403, str(exc)) from exc


@router.post("/connectors/{service}/revoke", dependencies=[Depends(operator_control)])
def revoke(service: str):
    try: connectors().revoke(service); return connectors().status()
    except (ValueError, RuntimeError) as exc: raise HTTPException(400, str(exc)) from exc
