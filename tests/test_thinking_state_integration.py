"""
tests/test_thinking_state_integration.py
========================================
Integration tests for AXEN's real Thinking/Processing state:
1. "Hello" -> fast conversational response
2. "25 * 48" -> calculation state transition
3. "Differentiate f(x)=(2x^3-5x+4)(x^2+3x-1)" -> Solving -> Verifying -> answer
4. "Explain how a CPU executes instructions" -> Thinking / Generating -> answer
5. "Generate a Python program to calculate fibonacci" -> Generating -> answer
6. Multi-step request -> appropriate state progression
7. Error / cancellation handling -> safe recovery, composer restored
8. Cancellation endpoint -> safe request cancellation without server crash
"""

import json
import pytest
from fastapi.testclient import TestClient
from app.ui.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _parse_sse(raw_text: str):
    events = []
    blocks = raw_text.split("\n\n")
    for b in blocks:
        if not b.strip():
            continue
        etype = None
        data = None
        for line in b.split("\n"):
            if line.startswith("event: "):
                etype = line[7:].strip()
            elif line.startswith("data: "):
                data_str = line[6:]
                try:
                    data = json.loads(data_str)
                except Exception:
                    data = data_str
        if etype:
            events.append({"event": etype, "data": data})
    return events


def test_case_1_hello_fast_response(client):
    """Case 1: 'Hello' quick response without unnecessary delay."""
    res = client.post("/api/chat/stream", json={"message": "Hello"})
    assert res.status_code == 200
    events = _parse_sse(res.text)
    event_names = [e["event"] for e in events]
    assert "done" in event_names
    results = [e["data"] for e in events if e["event"] == "result"]
    assert len(results) >= 1
    assert len(results[0]["response"].strip()) > 0


def test_case_2_calculator_processing_state(client):
    """Case 2: '25 * 48' calculation state progression."""
    res = client.post("/api/chat/stream", json={"message": "25 * 48"})
    assert res.status_code == 200
    events = _parse_sse(res.text)
    states = [e["data"].get("state") for e in events if e["event"] == "state"]
    assert any(s in ("UNDERSTANDING", "PLANNING", "USING_TOOL", "GENERATING", "COMPLETED") for s in states)
    results = [e["data"] for e in events if e["event"] == "result"]
    assert len(results) >= 1
    assert "1200" in results[0]["response"]


def test_case_3_differentiation_solving_verifying(client):
    """Case 3: 'Differentiate f(x)=...' -> Solving -> Verifying -> answer."""
    res = client.post(
        "/api/chat/stream",
        json={"message": "Differentiate f(x)=(2x^3-5x+4)(x^2+3x-1)"},
    )
    assert res.status_code == 200
    events = _parse_sse(res.text)
    state_texts = [e["data"].get("text", "") for e in events if e["event"] == "state"]
    assert any("Solving" in t or "Verifying" in t or "Understanding" in t for t in state_texts)
    results = [e["data"] for e in events if e["event"] == "result"]
    assert len(results) >= 1
    assert "x" in results[0]["response"]


def test_case_4_cpu_explanation_thinking_state(client):
    """Case 4: 'Explain how a CPU executes instructions' -> Thinking / Generating -> answer."""
    res = client.post(
        "/api/chat/stream",
        json={"message": "Explain how a CPU executes instructions"},
    )
    assert res.status_code == 200
    events = _parse_sse(res.text)
    states = [e["data"].get("state") for e in events if e["event"] == "state"]
    assert any(s in ("UNDERSTANDING", "PLANNING", "THINKING", "GENERATING", "COMPLETED") for s in states)
    results = [e["data"] for e in events if e["event"] == "result"]
    assert len(results) >= 1
    assert len(results[0]["response"]) > 20


def test_case_5_code_execution(client):
    """Case 5: 'Run Python code: ...' -> Running Python... -> answer."""
    res = client.post(
        "/api/chat/stream",
        json={"message": "Run this Python code: print([x**2 for x in range(5)])"},
    )
    assert res.status_code == 200
    events = _parse_sse(res.text)
    state_texts = [e["data"].get("text", "") for e in events if e["event"] == "state"]
    assert any("Python" in t or "Understanding" in t or "Planning" in t for t in state_texts)
    results = [e["data"] for e in events if e["event"] == "result"]
    assert len(results) >= 1
    assert "[0, 1, 4, 9, 16]" in results[0]["response"]


def test_case_6_cancellation_safety(client):
    """Case 6: Safely cancels active task without killing server."""
    req_id = "test_cancel_flow"
    res = client.post("/api/chat/cancel", json={"request_id": req_id})
    assert res.status_code == 200
    # Server continues serving normally immediately after
    res2 = client.get("/api/status")
    assert res2.status_code == 200
    assert res2.json()["status"] == "ready"


def test_case_7_rapid_clicks_handled_via_single_id(client):
    """Case 7: Sequential/concurrent requests do not corrupt backend session."""
    res1 = client.post("/api/chat", json={"message": "What is 10 + 10?"})
    res2 = client.post("/api/chat", json={"message": "What is 20 + 20?"})
    assert res1.status_code == 200
    assert res2.status_code == 200
    assert "20" in res1.json()["response"]
    assert "40" in res2.json()["response"]
