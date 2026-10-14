"""Persistent queue, checkpoint and append-only learning event evidence."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
import uuid


def now():
    return datetime.now(timezone.utc).isoformat()


class LearningStore:
    def __init__(self, path="data/axen_learning.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS goals (
                    id TEXT PRIMARY KEY, fingerprint TEXT UNIQUE NOT NULL,
                    priority INTEGER NOT NULL, status TEXT NOT NULL,
                    step INTEGER NOT NULL DEFAULT 0, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL,
                    goal_id TEXT, kind TEXT NOT NULL, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS lessons (id TEXT PRIMARY KEY, data TEXT NOT NULL);
            """)

    @contextmanager
    def connection(self):
        with self.lock:
            db = sqlite3.connect(self.path, timeout=15)
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA journal_mode=WAL")
            try:
                with db: yield db
            finally: db.close()

    def state(self, key, default=None):
        with self.connection() as db:
            row = db.execute("SELECT data FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_state(self, key, value):
        with self.connection() as db:
            db.execute("INSERT OR REPLACE INTO state VALUES (?,?)", (key, json.dumps(value)))

    def enqueue(self, goal):
        with self.connection() as db:
            cursor = db.execute("INSERT OR IGNORE INTO goals VALUES (?,?,?,?,?,?)",
                (goal["id"], goal["fingerprint"], goal["priority"], "queued", 0, json.dumps(goal)))
            return bool(cursor.rowcount)

    def goals(self):
        with self.connection() as db:
            rows = db.execute("SELECT * FROM goals ORDER BY priority DESC, rowid LIMIT 200").fetchall()
        return [dict(json.loads(r["data"]), status=r["status"], step=r["step"]) for r in rows]

    def claim(self):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM goals WHERE status='queued' ORDER BY priority DESC,rowid LIMIT 1").fetchone()
            if not row: return None
            db.execute("UPDATE goals SET status='running' WHERE id=?", (row["id"],))
            return dict(json.loads(row["data"]), status="running", step=row["step"])

    def checkpoint(self, goal, step, status="running"):
        goal.update(step=step, status=status)
        with self.connection() as db:
            db.execute("UPDATE goals SET step=?,status=?,data=? WHERE id=?",
                       (step, status, json.dumps(goal), goal["id"]))

    def recover(self):
        with self.connection() as db:
            db.execute("UPDATE goals SET status='queued' WHERE status='running'")

    def retry(self, goal_id, allowed_resources):
        with self.connection() as db:
            row = db.execute("SELECT * FROM goals WHERE id=? AND status='failed'", (goal_id,)).fetchone()
            if not row: raise ValueError("Only a failed goal may be retried")
            goal = json.loads(row["data"])
            goal["allowed_resources"] = allowed_resources
            goal.pop("error", None)
            db.execute("UPDATE goals SET status='queued',step=0,data=? WHERE id=?", (json.dumps(goal), goal_id))
        self.event("goal_retry", {"goal_id": goal_id, "reason": "Explicit operator retry"}, goal_id)

    def event(self, kind, data, goal_id=None):
        with self.connection() as db:
            db.execute("INSERT INTO events(at,goal_id,kind,data) VALUES (?,?,?,?)",
                       (now(), goal_id, kind, json.dumps(data)))

    def events(self, after=0, limit=100):
        with self.connection() as db:
            rows = db.execute("SELECT * FROM events WHERE id>? ORDER BY id LIMIT ?",
                              (after, min(limit, 200))).fetchall()
        return [dict(r, data=json.loads(r["data"])) for r in rows]

    def lesson(self, entry):
        key = entry["goal_id"]
        with self.connection() as db:
            db.execute("INSERT OR REPLACE INTO lessons VALUES (?,?)", (key, json.dumps(entry)))

    def lessons(self):
        with self.connection() as db:
            return [json.loads(r[0]) for r in db.execute("SELECT data FROM lessons ORDER BY rowid DESC LIMIT 200")]


class AutonomousGoalPlanner:
    """Select from measured failures in user-authorized areas; no invented goals."""
    def __init__(self, store, config):
        self.store, self.config = store, config

    def generate(self, observations):
        created = []
        for observation in observations:
            skill = observation.get("skill")
            if skill not in self.config["authorized_skills"]:
                continue
            total, passed = observation.get("total", 0), observation.get("passed", 0)
            if type(total) is not int or type(passed) is not int or not 0 <= passed < total:
                continue
            import hashlib
            stable_evidence = {k: v for k, v in observation.items()
                               if k not in ("elapsed_seconds", "retrieved_at", "timestamp")}
            fingerprint = hashlib.sha256(json.dumps(stable_evidence, sort_keys=True).encode()).hexdigest()
            goal = {"id": "learn_" + uuid.uuid4().hex[:12], "fingerprint": fingerprint,
                    "objective": "Improve " + skill, "skill": skill,
                    "reason": observation.get("reason", "Measured benchmark failures"),
                    "priority": round(100 * (1 - observation["passed"] / observation["total"])),
                    "expected_measurable_improvement": "Higher pass count on fixed held-out tests",
                    "allowed_resources": list(self.config["sources"]),
                    "time_budget_seconds": self.config["session_seconds"],
                    "status": "queued", "evaluation_criteria": "Improve fixed train and held-out scores with no regression",
                    "observation": observation, "created_at": now()}
            if self.store.enqueue(goal): created.append(goal)
        return created


def record_task_observation(store, *, skill, task_failed=False, low_confidence=False,
                            execution_failed=False, request_hash=""):
    """Signals select a curriculum; benchmarks still establish actual skill gaps.

    Store no user message, credentials or model-produced permissions.
    """
    if not (task_failed or low_confidence or execution_failed): return
    observation = {"skill": skill, "total": 1, "passed": 0, "request_hash": request_hash,
        "reason": "Task telemetry: " + ", ".join(k for k, v in {
            "task_failed": task_failed, "low_confidence": low_confidence,
            "execution_failed": execution_failed}.items() if v), "evidence_type": "task_signal_not_skill_measurement"}
    with store.lock:
        observations = store.state("observations", [])
        if observation not in observations:
            store.set_state("observations", (observations + [observation])[-100:])
            store.event("task_observation", observation)
