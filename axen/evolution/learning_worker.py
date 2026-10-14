"""Persistent, bounded, explicitly enabled autonomous learning worker."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
import time

from .knowledge_acquisition import KnowledgeAcquisitionEngine, retrieve_learned_context
from .learning_store import AutonomousGoalPlanner, LearningStore, now
from .practice import DEBUGGING_SOURCE, TRAIN_CASES, HELD_OUT_CASES, SkillPracticeAgent
from .research import ResearchDocument, WebResearchAgent

ROOT = Path(__file__).resolve().parents[2]
MODES = ("OFF", "LEARNING", "DEVELOPMENT", "SUPERVISED EVOLUTION")


def load_config(path=None):
    config = json.loads(Path(path or ROOT / "configs/autonomy.json").read_text(encoding="utf-8"))
    for key, ceiling in (("session_seconds", 7200), ("max_goals", 10), ("max_sources", 10),
                         ("max_videos", 5), ("max_experiments", 10), ("max_training_jobs", 1),
                         ("max_requests", 30)):
        if type(config.get(key)) is not int or not 1 <= config[key] <= ceiling:
            raise ValueError("Invalid resource limit: " + key)
    if not isinstance(config.get("authorized_skills"), list): raise ValueError("Authorized skills required")
    background = config.get('background_agent', {})
    for key in ('enabled', 'auto_apply_tested_changes'):
        if key in background and type(background[key]) is not bool:
            raise ValueError('Invalid background policy: ' + key)
    interval = background.get('cycle_interval_seconds', 3600)
    if type(interval) is not int or not 60 <= interval <= 86400:
        raise ValueError('Background cycle interval must be 60..86400 seconds')
    for key in ("cpu_busy_percent", "memory_busy_percent"):
        if not 1 <= config.get(key, 0) <= 100: raise ValueError("Invalid resource threshold")
    return config


def resource_snapshot():
    try:
        import psutil
        return {"available": True, "cpu_percent": psutil.cpu_percent(interval=None),
                "memory_percent": psutil.virtual_memory().percent,
                "available_memory_mb": round(psutil.virtual_memory().available / 1024**2)}
    except ImportError:
        # Fail closed for background operations if real resource telemetry is absent.
        return {"available": False, "limitation": "Install psutil for resource monitoring"}


class LearningInterrupted(Exception):
    pass


class AutonomousLearningWorker:
    steps = ("DISCOVER", "RETRIEVE", "ANALYZE", "CROSS-CHECK", "ORGANIZE", "PRACTICE", "VERIFY", "STORE")

    def __init__(self, store=None, config=None, memory=None, research_factory=None,
                 resources=resource_snapshot):
        self.store = store or LearningStore(ROOT / "data/axen_learning.db")
        self.config = config or load_config()
        self.knowledge = KnowledgeAcquisitionEngine(self.store, memory)
        self.practice = SkillPracticeAgent()
        self.planner = AutonomousGoalPlanner(self.store, self.config)
        self.research_factory = research_factory
        self.resources = resources
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.thread = None; self.shutdown_event = threading.Event()
        self.mode = "OFF"; self.status = "stopped"
        self.emergency = bool(self.store.state("emergency", False))
        self.deadline = 0; self.started_at = None; self.current_goal = None
        self.completed = 0; self.experiments = 0
        self.store.recover()

    def command(self, action, mode=None, *, offline_demo=False):
        with self.condition:
            if mode is not None and mode not in MODES: raise ValueError("Unknown autonomy mode")
            if action == "emergency-stop":
                self.emergency = True; self.store.set_state("emergency", True)
                self.status = "emergency_stopped"; self.mode = "OFF"
            elif action == "stop" or mode == "OFF":
                self.status = "stopped"; self.mode = "OFF"
            elif action == "pause":
                if self.status not in ("running", "resource_paused", "idle"):
                    raise ValueError("Worker is not running")
                self.status = "paused"
            elif action in ("start", "resume"):
                if self.shutdown_event.is_set(): raise ValueError("Worker is shutting down")
                if self.emergency:
                    raise ValueError("Emergency stop latched; use explicit reset-emergency first")
                if action == "resume":
                    if self.status != "paused": raise ValueError("Worker is not paused")
                else:
                    if self.thread and self.thread.is_alive():
                        raise ValueError("Wait for the previous session to finish cancellation")
                    if self.status in ("running", "idle", "resource_paused", "paused"):
                        raise ValueError("Stop the existing session before starting another")
                    self.mode = mode or "LEARNING"
                    self.config = dict(self.config, offline_demo=offline_demo)
                    self.deadline = time.monotonic() + self.config["session_seconds"]
                    self.started_at = now(); self.completed = 0; self.experiments = 0
                self.status = "running"
                if not self.thread or not self.thread.is_alive():
                    self.thread = threading.Thread(target=self._run, name="axen-learning", daemon=False)
                    self.thread.start()
            elif action == "reset-emergency":
                if self.current_goal is not None or (self.thread and self.thread.is_alive()):
                    raise ValueError("Wait for active step cancellation")
                self.emergency = False; self.store.set_state("emergency", False)
                self.status = "stopped"; self.mode = "OFF"
            else: raise ValueError("Unknown worker command")
            self.store.event("control", {"action": action, "mode": self.mode, "status": self.status})
            self.condition.notify_all()
            return self.snapshot()

    def interrupted(self):
        return (self.shutdown_event.is_set() or self.emergency or self.mode == "OFF"
                or time.monotonic() >= self.deadline)

    def _guard(self):
        with self.condition:
            while self.status == "paused" and not self.interrupted():
                self.condition.wait(timeout=0.2)
            if self.interrupted(): raise LearningInterrupted()
            while getattr(self, 'foreground_busy', lambda: False)() and not self.interrupted():
                self.status = 'chat_paused'
                self.condition.wait(timeout=0.2)
            if self.interrupted(): raise LearningInterrupted()
            if self.status == 'chat_paused': self.status = 'running'

    def _run(self):
        try:
            while not self.interrupted():
                self._guard()
                resources = self.resources()
                if (not resources.get("available") or resources["cpu_percent"] >= self.config["cpu_busy_percent"]
                        or resources["memory_percent"] >= self.config["memory_busy_percent"]
                        or resources.get("available_memory_mb", 10000) < self.config.get("minimum_available_mb", 128)):
                    if self.status != "resource_paused":
                        self.status = "resource_paused"; self.store.event("resources", resources)
                    with self.condition: self.condition.wait(timeout=1)
                    continue
                if self.status == "resource_paused": self.status = "running"
                if self.completed >= self.config["max_goals"]:
                    self.status = "session_complete"; self.mode = "OFF"; break
                goal = self.store.claim()
                if goal is None:
                    # Detect a reproducible exercise weakness without asking the user
                    # to click Evaluate. The report explicitly identifies the fixture.
                    baseline = self.practice.evaluate(DEBUGGING_SOURCE, TRAIN_CASES, cancel=self.interrupted)
                    observations = self.store.state("observations", [])
                    observations.append(dict(baseline, skill="Python debugging",
                        reason="Fixed inclusive-sum debugging fixture fails boundary tests",
                        source=DEBUGGING_SOURCE, benchmark="inclusive-sum-v1"))
                    self.planner.generate(observations)
                    goal = self.store.claim()
                if goal is None:
                    self.status = "idle"
                    with self.condition: self.condition.wait(timeout=1)
                    continue
                self.status = "running"; self.current_goal = goal["id"]
                try:
                    self.run_goal(goal)
                    self.completed += 1
                except LearningInterrupted:
                    self.store.checkpoint(goal, goal["step"], "queued")
                    self.store.event("interrupted", {"checkpoint": goal["step"]}, goal["id"])
                    break
                except Exception as exc:
                    if self.interrupted():
                        self.store.checkpoint(goal, goal["step"], "queued")
                        self.store.event("interrupted", {"checkpoint": goal["step"]}, goal["id"])
                        break
                    goal["error"] = str(exc)
                    self.store.checkpoint(goal, goal["step"], "failed")
                    self.store.event("failed", {"error": str(exc),
                        "revised_plan": "Review source/containment availability before retrying; no automatic repeated failure"}, goal["id"])
                    self.completed += 1
                finally: self.current_goal = None
        except LearningInterrupted:
            pass
        finally:
            with self.condition:
                if self.status not in ("emergency_stopped", "session_complete"):
                    self.status = "stopped"; self.mode = "OFF"
                self.store.event("worker_exit", {"status": self.status})

    def run_goal(self, goal):
        """Each completed step is checkpointed; writes use idempotent goal IDs."""
        if goal["skill"] not in self.config["authorized_skills"]:
            raise PermissionError("Goal outside operator-authorized areas")
        if goal["skill"] != "Python debugging":
            raise ValueError("No practice curriculum configured for this authorized skill")
        for index in range(goal.get("step", 0), len(self.steps)):
            self._guard()
            kind = self.steps[index]
            self.store.event("activity", {"step": kind, "objective": goal["objective"]}, goal["id"])
            if kind == "DISCOVER":
                source = goal["observation"].get("source", DEBUGGING_SOURCE)
                goal["original_source"] = source
                goal["baseline"] = {"train": self.practice.evaluate(source, TRAIN_CASES, cancel=self.interrupted),
                    "held_out": self.practice.evaluate(source, HELD_OUT_CASES, cancel=self.interrupted)}
                self.store.event("baseline", goal["baseline"], goal["id"])
            elif kind == "RETRIEVE":
                docs = []
                if self.config["offline_demo"]:
                    path = ROOT / "knowledge/learning/python_range_offline.md"
                    body = path.read_text(encoding="utf-8")
                    docs = [ResearchDocument("local:knowledge/learning/python_range_offline.md", now(), body,
                        hashlib.sha256(body.encode()).hexdigest(), access="explicit_offline_demo_fixture").to_dict()]
                else:
                    researcher = (self.research_factory(self.interrupted) if self.research_factory else
                        WebResearchAgent(self.config["domains"], max_sources=self.config["max_sources"],
                            interval_seconds=self.config["request_interval_seconds"],
                            max_requests=self.config["max_requests"], cancel=self.interrupted))
                    catalog = [s for s in self.config["sources"] if s in goal["allowed_resources"]]
                    for source in researcher.search("Python debugging range boundary", catalog):
                        self._guard()
                        try:
                            doc = researcher.retrieve(source["url"]).to_dict(); docs.append(doc)
                            self.store.event("source", {k: doc[k] for k in ("url", "retrieved_at", "access")}, goal["id"])
                        except ValueError as exc:
                            self.store.event("research_limitation", {"url": source["url"], "reason": str(exc)}, goal["id"])
                    if not docs:
                        raise ValueError("Approved public research unavailable. Stop and explicitly select offline demo to use local material.")
                goal["documents"] = docs
                self._video(goal)
                if self.config["offline_demo"]:
                    self.store.event("source", {k: docs[0][k] for k in ("url", "retrieved_at", "access")}, goal["id"])
            elif kind == "ANALYZE":
                goal["concept"] = self.knowledge.analyze(goal["documents"])
                self.store.event("findings", goal["concept"], goal["id"])
            elif kind == "CROSS-CHECK":
                self.store.event("cross_check", {"result": goal["concept"]["cross_check"],
                    "scope": goal["concept"]["verification_scope"]}, goal["id"])
            elif kind == "ORGANIZE":
                goal["practice_plan"] = {"exercise": "Repair inclusive_sum boundary", "suite": "inclusive-sum-v1",
                    "generator": "bounded AST rule derived from studied range concept", "held_out_visible_to_generator": False}
            elif kind == "PRACTICE":
                goal["candidate_source"] = self.practice.propose(goal["original_source"], goal["concept"]["concept"])
                goal["practice"] = self.practice.evaluate(goal["candidate_source"], TRAIN_CASES, cancel=self.interrupted)
                self.store.event("practice", goal["practice"], goal["id"])
            elif kind == "VERIFY":
                held = self.practice.evaluate(goal["candidate_source"], HELD_OUT_CASES, cancel=self.interrupted)
                goal["evaluation"] = {"train": goal["practice"], "held_out": held}
                baseline = goal["baseline"]
                goal["improvement_demonstrated"] = (goal["practice"]["passed"] > baseline["train"]["passed"]
                    and held["passed"] > baseline["held_out"]["passed"]
                    and goal["practice"]["passed"] == goal["practice"]["total"]
                    and held["passed"] == held["total"])
                self.store.event("evaluation", {"baseline": baseline, "candidate": goal["evaluation"],
                    "improvement_demonstrated": goal["improvement_demonstrated"],
                    "scope": "Fixed arithmetic repair fixture, not general assistant performance"}, goal["id"])
                if not goal["improvement_demonstrated"]:
                    raise ValueError("Practice did not demonstrate improvement; lesson will not be verified")
            elif kind == "STORE":
                # Emergency-stop and memory write are serialized with trusted controls.
                with self.lock:
                    self._guard()
                    entry = self.knowledge.store_verified(goal, goal["concept"], goal["evaluation"])
                    examples = self.knowledge.verified_training_examples(goal)
                    dataset_dir = self.store.path.parent / "training" / "verified_learning"
                    dataset_dir.mkdir(parents=True, exist_ok=True)
                    dataset_path = dataset_dir / (goal["id"] + ".json")
                    dataset_path.write_text(json.dumps(examples, indent=2), encoding="utf-8")
                    goal["verified_dataset"] = str(dataset_path)
                    self.store.event("training_examples_collected", {"path": str(dataset_path),
                        "counts": {k: len(v) for k, v in examples.items()}, "weights_changed": False}, goal["id"])
                    retrieved = retrieve_learned_context("Python debugging range", self.knowledge.memory)
                    self.store.event("knowledge_stored", {"topic": entry["topic"], "retrieval": retrieved}, goal["id"])
                if self.mode in ("DEVELOPMENT", "SUPERVISED EVOLUTION"):
                    self._evolution(goal)
                if self.mode == "SUPERVISED EVOLUTION":
                    self._train(goal)
            self._guard()
            self.store.checkpoint(goal, index + 1)
        self.store.checkpoint(goal, len(self.steps), "completed")
        self.store.event("completed", {"improvement_demonstrated": goal["improvement_demonstrated"],
            "title": "AXEN Learns Python Debugging", "offline": self.config["offline_demo"]}, goal["id"])

    def _evolution(self, goal):
        # Exercise source is not a production weakness; never manufacture a
        # production patch from a synthetic fixture.
        targets = self.config.get("code_targets", [])
        if not targets:
            self.store.event("code_evolution", {"status": "not_justified", "candidate_source": goal["candidate_source"],
                "reason": "Repair verified on exercise only. No authorized reproducible production bug established.",
                "production_modified": False}, goal["id"])
            return
        from .approval import ApprovalGate
        from .controller import EvolutionController
        from .store import EvolutionStore
        from .models import Candidate, BenchmarkResult
        from .weakness_detector import Weakness
        store = EvolutionStore(ROOT / "data/axen_evolution.db")
        gate = ApprovalGate(store, repo_root=str(ROOT))
        practice = self.practice
        class Generator:
            def generate_candidates(self, experiment, weakness, original_code, **kwargs):
                return [Candidate(experiment_id=experiment.experiment_id, target_file=weakness.target_file,
                    original_code=original_code, code=practice.propose(original_code, goal["concept"]["concept"]),
                    rationale="Studied exclusive range boundary; measured inclusive-sum repair")]
        def evaluate(candidate):
            self._guard()
            train = practice.evaluate(candidate.code, TRAIN_CASES, cancel=self.interrupted)
            held = practice.evaluate(candidate.code, HELD_OUT_CASES, cancel=self.interrupted)
            return BenchmarkResult(candidate_id=candidate.candidate_id, suite_name="inclusive-sum-v1",
                passed_tests=train["passed"], total_tests=train["total"],
                held_out_passed=held["passed"], held_out_total=held["total"])
        for target in targets[:self.config["max_experiments"]]:
            self._guard()
            if target.get("contract") != "inclusive_sum_nonnegative":
                raise ValueError("Production experiment requires the operator-approved inclusive-sum contract")
            path = gate._target(target["path"])
            source = path.read_text(encoding="utf-8")
            baseline = practice.evaluate(source, TRAIN_CASES, cancel=self.interrupted)
            if baseline["passed"] == baseline["total"]: continue
            weakness = Weakness(target_module=path.stem, target_file=str(path),
                description="Inclusive-sum production target fails configured boundary contract",
                evidence=[json.dumps(baseline)], category="test_failure")
            controller = EvolutionController(store, Generator())
            results = controller.run_observation_loop(lambda: [weakness], lambda p: source,
                evaluate, max_cycles=1, max_experiments=1, num_candidates=1,
                should_stop=self.interrupted, time_budget_seconds=max(1, self.deadline - time.monotonic()))
            self.experiments += len(results)
            self.store.event("code_evolution", {"experiment_ids": [e.experiment_id for e in results],
                "status": [e.status.value for e in results], "production_modified": False}, goal["id"])

    def _video(self, goal):
        config = self.config.get("video_learning", {})
        if not config.get("enabled") or self.config["offline_demo"]: return
        from .connectors import AccountConnectorManager, YouTubeMetadataProvider
        from .research import VideoLearningAgent
        if not hasattr(self, "connectors"): self.connectors = AccountConnectorManager(self.store)
        provider = YouTubeMetadataProvider(self.connectors)
        agent = VideoLearningAgent(provider, max_videos=self.config["max_videos"])
        try:
            results = agent.search("Python range debugging tutorial")
            for video in results["videos"]:
                self._guard()
                authorized = config.get("licensed_transcripts", {}).get(video["id"])
                transcript = None
                if authorized:
                    path = (ROOT / authorized).resolve()
                    if not path.is_relative_to(ROOT / "knowledge/learning/videos"):
                        raise PermissionError("Transcript outside operator-authorized educational material")
                    if path.stat().st_size > 100_000: raise ValueError("Transcript budget exceeded")
                    transcript = path.read_text(encoding="utf-8")
                studied = agent.study(video, licensed_transcript=transcript, authorization=bool(authorized))
                self.store.event("video", {k: v for k, v in studied.items() if k != "text"}, goal["id"])
                if studied.get("text") and len(goal["documents"]) < self.config["max_sources"]:
                    goal["documents"].append(ResearchDocument(video["url"], studied["retrieved_at"],
                        studied["text"], hashlib.sha256(studied["text"].encode()).hexdigest(),
                        access="operator_authorized_local_transcript").to_dict())
        except (ValueError, PermissionError, RuntimeError) as exc:
            self.store.event("research_limitation", {"service": "youtube", "reason": str(exc)}, goal["id"])

    def _train(self, goal):
        config = self.config.get("neural_training", {})
        if not config.get("enabled"):
            self.store.event("training_limitation", {"reason": "Candidate training is not configured by the operator"}, goal["id"])
            return
        self._guard()
        if self.resources().get("available_memory_mb", 0) < 1024:
            self.store.event("training_limitation", {"reason": "At least 1 GB available RAM is required for optional candidate training"}, goal["id"])
            return
        checkpoint = (ROOT / str(config.get("checkpoint", ""))).resolve()
        dataset_setting = config.get("dataset", "")
        dataset = (Path(goal["verified_dataset"]) if dataset_setting == "verified_learning"
                   else ROOT / str(dataset_setting)).resolve()
        if not checkpoint.is_relative_to(ROOT / "checkpoints") or not dataset.is_relative_to(ROOT / "data/training"):
            raise PermissionError("Training inputs must be operator-configured checkpoints and verified data/training files")
        import torch
        from axen.model.foundation import AXENFoundationModel, FoundationModelConfig, FoundationTokenizer
        from .neural_trainer import EvolutionNeuralTrainer
        from .store import EvolutionStore
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model_config = FoundationModelConfig.from_dict(payload["config"])
        model_config.device = "cpu"
        model = AXENFoundationModel(model_config, FoundationTokenizer(vocab=payload["tokenizer_vocab"]))
        model.load_state_dict(payload["model_state"])
        records = json.loads(dataset.read_text(encoding="utf-8"))
        # Only operator-provisioned verified datasets: no arbitrary source text
        # or unverified model transcript is admitted as ground truth.
        trainer = EvolutionNeuralTrainer(store=EvolutionStore(ROOT / "data/axen_evolution.db"),
            staging_dir=str(ROOT / "checkpoints/evolution_staging"))
        result = trainer.train_candidate_weights("neural_" + goal["id"], model,
            records["train"], records["validation"], held_out_samples=records["held_out"],
            replay_samples=records.get("replay", []), max_steps=config.get("max_steps", 15),
            cancel=self.interrupted, event=lambda kind, data: self.store.event(kind, data, goal["id"]))
        from dataclasses import asdict
        self.store.event("training_result", asdict(result), goal["id"])

    def snapshot(self):
        return {"mode": self.mode, "status": self.status, "emergency_latched": self.emergency,
                "current_goal": self.current_goal, "session_started_at": self.started_at,
                "remaining_seconds": max(0, round(self.deadline - time.monotonic())) if self.mode != "OFF" else 0,
                "goals": self.store.goals(), "lessons": self.store.lessons(), "resources": self.resources(),
                "limits": {k: self.config[k] for k in ("session_seconds", "max_goals", "max_sources", "max_videos", "max_experiments", "max_training_jobs")}}

    def shutdown(self):
        self.shutdown_event.set()
        with self.condition:
            self.mode = "OFF"; self.condition.notify_all()
        if self.thread and self.thread is not threading.current_thread(): self.thread.join(timeout=15)
        if self.thread and self.thread.is_alive(): raise RuntimeError("Learning worker did not shut down within budget")


_worker = None
_singleton_lock = threading.Lock()


def get_worker():
    global _worker
    with _singleton_lock:
        if _worker is None: _worker = AutonomousLearningWorker()
        return _worker
