"""Offline integration and security invariants; network smoke is separate."""
import json
from pathlib import Path
import socket
import threading
import time

import pytest

from axen.evolution.connectors import AccountConnectorManager
from axen.evolution.knowledge_acquisition import retrieve_learned_context
from axen.evolution.learning_store import AutonomousGoalPlanner, LearningStore
from axen.evolution.learning_worker import AutonomousLearningWorker, load_config
from axen.evolution.practice import ArithmeticSandbox, DEBUGGING_SOURCE, SkillPracticeAgent
from axen.evolution.research import ResearchDocument, ResearchUnavailable, VideoLearningAgent, WebResearchAgent
from axen.memory.memory_store import SQLiteMemoryStore


def worker_at(tmp_path, **kwargs):
    config = load_config(); config.update(session_seconds=20, offline_demo=True)
    config.update(kwargs.pop("config_overrides", {}))
    return AutonomousLearningWorker(LearningStore(tmp_path / "learning.db"), config,
        SQLiteMemoryStore(tmp_path / "memory.db"),
        resources=kwargs.pop("resources", lambda: {"available": True, "cpu_percent": 2, "memory_percent": 20}), **kwargs)


def finish(worker):
    worker.thread.join(timeout=5)
    assert not worker.thread.is_alive(), worker.snapshot()


def test_complete_offline_cycle_and_retrieval(tmp_path):
    worker = worker_at(tmp_path)
    assert worker.mode == "OFF" and worker.thread is None
    worker.command("start", "LEARNING", offline_demo=True)
    finish(worker)
    goal = worker.store.goals()[0]
    assert goal["status"] == "completed"
    assert goal["baseline"]["train"]["passed"] == 1
    assert goal["evaluation"]["train"]["passed"] == 4
    assert goal["baseline"]["held_out"]["passed"] == 0
    assert goal["evaluation"]["held_out"]["passed"] == 3
    assert goal["improvement_demonstrated"]
    entries = retrieve_learned_context("Python debugging range", worker.knowledge.memory)
    assert entries and "boundary" in entries[0]["summary"]
    assert entries[0]["sources"][0]["access"] == "explicit_offline_demo_fixture"
    assert worker.mode == "OFF" and worker.status == "session_complete"
    restored = worker_at(tmp_path)
    assert restored.mode == "OFF" and restored.thread is None
    assert restored.store.lessons()
    assert retrieve_learned_context("range", restored.knowledge.memory)


@pytest.mark.parametrize("mode", ["LEARNING", "DEVELOPMENT", "SUPERVISED EVOLUTION"])
def test_modes_do_not_modify_production(tmp_path, mode):
    worker = worker_at(tmp_path)
    worker.command("start", mode, offline_demo=True); finish(worker)
    events = worker.store.events()
    assert any(e["kind"] == "completed" for e in events)
    if mode != "LEARNING":
        assert next(e for e in events if e["kind"] == "code_evolution")["data"]["production_modified"] is False


def test_planner_authorized_measured_deduplicated(tmp_path):
    worker = worker_at(tmp_path)
    observations = [{"skill": "Python debugging", "passed": 1, "total": 4},
                    {"skill": "unauthorized", "passed": 0, "total": 3},
                    {"skill": "Python debugging", "passed": 4, "total": 4}]
    created = worker.planner.generate(observations)
    assert len(created) == 1
    assert worker.planner.generate(observations) == []
    goal = created[0]
    assert all(k in goal for k in ("id", "objective", "reason", "priority", "expected_measurable_improvement",
                                  "allowed_resources", "time_budget_seconds", "status", "evaluation_criteria"))
    assert worker.planner.generate([dict(observations[0], elapsed_seconds=999)]) == []


def test_pause_resume_emergency_and_recovery(tmp_path):
    gate = threading.Event(); entered = threading.Event()
    class Research:
        def search(self, question, catalog): return catalog
        def retrieve(self, url):
            entered.set(); assert gate.wait(timeout=3)
            return ResearchDocument(url, "2026-10-14T00:00:00Z", "Python range stop is exclusive", "hash")
    worker = worker_at(tmp_path, research_factory=lambda cancel: Research())
    worker.command("start", "LEARNING", offline_demo=False)
    assert entered.wait(timeout=3)
    worker.command("pause"); gate.set()
    time.sleep(0.1)
    assert not worker.store.lessons()
    assert worker.status == "paused"
    worker.command("resume"); finish(worker)
    worker.command("emergency-stop")
    restored = worker_at(tmp_path)
    assert restored.emergency and restored.mode == "OFF"
    with pytest.raises(ValueError, match="latched"): restored.command("start", "LEARNING")
    restored.command("reset-emergency")
    assert not restored.emergency


def test_emergency_during_research_prevents_writes(tmp_path):
    entered = threading.Event()
    class Research:
        def __init__(self, cancel): self.cancel = cancel
        def search(self, question, catalog): return catalog
        def retrieve(self, url):
            entered.set()
            while not self.cancel(): time.sleep(0.01)
            return ResearchDocument(url, "now", "Python range stop is exclusive", "hash")
    worker = worker_at(tmp_path, research_factory=Research)
    worker.command("start", "LEARNING", offline_demo=False)
    assert entered.wait(timeout=3)
    worker.command("emergency-stop"); finish(worker)
    assert not worker.store.lessons()
    assert worker.store.goals()[0]["status"] == "queued"
    assert worker.status == "emergency_stopped"


def test_resource_pause_and_session_budget(tmp_path):
    busy = {"value": True}
    worker = worker_at(tmp_path, resources=lambda: {"available": True, "cpu_percent": 99 if busy["value"] else 1, "memory_percent": 1})
    worker.command("start", "LEARNING", offline_demo=True)
    deadline = time.monotonic() + 2
    while worker.status != "resource_paused" and time.monotonic() < deadline: time.sleep(0.01)
    assert worker.status == "resource_paused" and not worker.store.goals()
    worker.command("pause"); busy["value"] = False
    time.sleep(0.05)
    assert worker.status == "paused"
    worker.command("resume"); finish(worker)
    assert worker.store.lessons()


def test_checkpoint_restoration_after_interruption(tmp_path):
    worker = worker_at(tmp_path)
    worker.planner.generate([{"skill": "Python debugging", "passed": 0, "total": 3}])
    goal = worker.store.claim()
    worker.store.checkpoint(goal, 0, "running")
    restored = worker_at(tmp_path)
    assert restored.store.goals()[0]["status"] == "queued"
    restored.command("start", "LEARNING", offline_demo=True); finish(restored)
    assert restored.store.goals()[0]["status"] == "completed"


def test_failed_practice_is_not_verified(tmp_path, monkeypatch):
    worker = worker_at(tmp_path)
    monkeypatch.setattr(worker.practice, "propose", lambda *args: DEBUGGING_SOURCE)
    worker.command("start", "LEARNING", offline_demo=True); finish(worker)
    assert worker.store.goals()[0]["status"] == "failed"
    assert not worker.store.lessons()
    assert not retrieve_learned_context("Python range", worker.knowledge.memory)
    worker.store.retry(worker.store.goals()[0]["id"], worker.config["sources"])
    assert worker.store.goals()[0]["status"] == "queued"


def test_mock_public_research_full_cycle(tmp_path):
    calls = []
    class Research:
        def search(self, question, catalog): calls.append(question); return catalog
        def retrieve(self, url):
            return ResearchDocument(url, "2026-10-14T00:00:00Z", "Python range excludes its stop boundary", "abc")
    worker = worker_at(tmp_path, research_factory=lambda cancel: Research())
    worker.command("start", "LEARNING", offline_demo=False); finish(worker)
    assert calls and worker.store.lessons()[0]["source_references"][0]["url"].startswith("https://")


def public_dns(*args, **kwargs): return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", 443))]


class Response:
    def __init__(self, text, status=200, kind="text/plain"):
        self.status_code = status; self.headers = {"Content-Type": kind}; self.body = text.encode()
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def raise_for_status(self):
        import requests
        if self.status_code >= 400: raise requests.HTTPError()
    def iter_content(self, size): yield self.body


def test_research_robots_budget_redirects_and_allowlist(monkeypatch):
    class Session:
        def get(self, url, **kwargs):
            assert kwargs["allow_redirects"] is False
            return Response("User-agent: *\nDisallow: /private\n" if url.endswith("robots.txt") else "Python range stop is exclusive")
    researcher = WebResearchAgent(["docs.python.org"], session=Session(), resolve=public_dns, max_sources=1)
    with pytest.raises(ResearchUnavailable, match="robots"): researcher.retrieve("https://docs.python.org/private")
    researcher.last_request = 0
    doc = researcher.retrieve("https://docs.python.org/allowed")
    assert doc.verification_status == "untrusted_observation" and doc.retrieved_at and doc.content_hash
    with pytest.raises(ResearchUnavailable, match="source budget"): researcher.retrieve("https://docs.python.org/allowed")
    for url in ("http://docs.python.org/", "https://evil.test/", "https://docs.python.org.evil.test/",
                "https://secret@docs.python.org/", "https://docs.python.org:444/", "https://docs.python.org/?token=secret"):
        with pytest.raises(ResearchUnavailable): researcher.validate_url(url)
    local = WebResearchAgent(["docs.python.org"], resolve=lambda *a, **k: [(2, 1, 6, "", ("127.0.0.1", 443))])
    with pytest.raises(ResearchUnavailable, match="network address"): local.validate_url("https://docs.python.org/")
    redirect = WebResearchAgent(["docs.python.org"], resolve=public_dns,
                               session=type("Session", (), {"get": lambda *a, **k: Response("", 302)})())
    with pytest.raises(ResearchUnavailable, match="Redirect"): redirect._get("https://docs.python.org/")


@pytest.mark.parametrize("source", [
    "import os\ndef solve(n):\n return 1", "def solve(n):\n return open(n)",
    "def solve(n):\n return n.__class__", "def solve(n):\n return __import__(n)",
    "def solve(n):\n return 2 ** n", "def solve(n):\n return [0] * n",
    "def solve(n):\n while True:\n  pass", "def solve(n=open('secret')):\n return n",
    "@evil\ndef solve(n):\n return n",
])
def test_arithmetic_containment_rejects_host_capabilities(source):
    with pytest.raises(ValueError): ArithmeticSandbox().run(source, [1])


def test_arithmetic_fuel_numeric_and_cancellation_limits():
    sandbox = ArithmeticSandbox()
    assert sandbox.run("def add(a,b):\n return a+b", [2, 3]) == 5
    for n in (10**13, 10001):
        with pytest.raises(ValueError): sandbox.run(DEBUGGING_SOURCE, [n])
    with pytest.raises(ValueError): sandbox.run(DEBUGGING_SOURCE, [4], cancel=lambda: True)


def test_video_requires_authorized_captions_and_application():
    agent = VideoLearningAgent()
    assert agent.search("Python")["limitation"]
    video = {"url": "https://www.youtube.com/watch?v=123", "title": "Python", "author": "Author", "duration": 120}
    assert agent.study(video)["verified"] is False
    with pytest.raises(PermissionError): agent.study(video, licensed_transcript="Lesson")
    result = agent.study(video, licensed_transcript="range excludes stop", authorization=True)
    assert not result["verified"] and result["retrieved_at"]


def test_connector_consent_readonly_revocation_and_no_secret_logs(tmp_path):
    class Vault:
        def __init__(self): self.secret = "hidden-secret"
        def get(self, service): return self.secret
        def delete(self, service): self.secret = None
    store = LearningStore(tmp_path / "connectors.db")
    manager = AccountConnectorManager(store, Vault())
    with pytest.raises(PermissionError): manager.authorize("github", ["public_read"])
    with pytest.raises(PermissionError): manager.authorize("github", ["write"], consent=True)
    with pytest.raises(PermissionError): manager.get("github", "repos/a/b")
    manager.authorize("github", ["public_read"], consent=True)
    with pytest.raises(PermissionError): manager.get("github", "user/messages")
    manager.revoke("github")
    with pytest.raises(PermissionError): manager.get("github", "repos/a/b")
    assert "hidden-secret" not in json.dumps(store.events())


def test_chat_context_contains_verified_lesson_and_sources(tmp_path):
    worker = worker_at(tmp_path)
    worker.command("start", "LEARNING", offline_demo=True); finish(worker)
    from axen.context.context_engine import ContextEngine
    class Memory:
        store = worker.knowledge.memory
        def retrieve_scored(self, **kwargs): return []
    assembled = ContextEngine(memory_manager=Memory()).build_context("Explain Python debugging range boundary", task_type="coding")
    text = json.dumps(assembled.messages_for_model)
    assert "excludes its stop" in text and "python_range_offline" in text


def test_research_contradictions_are_reported_and_verified_examples_have_provenance(tmp_path):
    worker = worker_at(tmp_path)
    concept = worker.knowledge.analyze([
        {"url": "source-a", "text": "Python range stop is exclusive"},
        {"url": "source-b", "text": "Python range includes stop"}])
    assert concept["contradictions"][0]["resolution"].startswith("Refuted")
    worker.command("start", "LEARNING", offline_demo=True); finish(worker)
    goal = worker.store.goals()[0]
    records = json.loads(Path(goal["verified_dataset"]).read_text(encoding="utf-8"))
    assert all(r["verified"] and r["metadata"]["provenance"] for split in records.values() for r in split)
    hashes = [{(r["instruction"], r["output"]) for r in records[k]} for k in ("train", "validation", "held_out")]
    assert not hashes[0] & hashes[1] and not hashes[1] & hashes[2]


def test_emergency_command_can_cancel_optional_training(tmp_path, monkeypatch):
    entered = threading.Event()
    worker = worker_at(tmp_path)
    def training(goal):
        entered.set()
        while not worker.interrupted(): time.sleep(0.01)
    monkeypatch.setattr(worker, "_train", training)
    worker.command("start", "SUPERVISED EVOLUTION", offline_demo=True)
    assert entered.wait(timeout=3)
    start = time.monotonic(); worker.command("emergency-stop")
    assert time.monotonic() - start < 1
    finish(worker)
    assert not any(event["kind"] == "completed" for event in worker.store.events())


def test_learning_api_operator_boundary_and_event_cursor(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import axen.evolution.learning_api as api
    worker = worker_at(tmp_path)
    monkeypatch.setattr(api, "get_worker", lambda: worker)
    app = FastAPI(); app.include_router(api.router)
    with TestClient(app) as client:
        assert client.get("/api/autonomy/status").json()["mode"] == "OFF"
        assert client.post("/api/autonomy/control", json={"action": "start"}).status_code == 403
        assert client.post("/api/autonomy/control", headers={"X-AXEN-Control": "operator", "Origin": "https://evil.test"},
                           json={"action": "start"}).status_code == 403
        response = client.post("/api/autonomy/control", headers={"X-AXEN-Control": "operator"},
                               json={"action": "start", "mode": "LEARNING", "offline_demo": True})
        assert response.status_code == 200
        finish(worker)
        events = client.get("/api/autonomy/events").json()["events"]
        after = events[-1]["id"]
        assert client.get(f"/api/autonomy/events?after={after}").json()["events"] == []
    worker.shutdown()
