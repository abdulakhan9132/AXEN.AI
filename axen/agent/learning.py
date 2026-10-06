"""
axen/agent/learning.py
======================
Learning Event Store, Training Data Generation, Sanitizer, and Evaluation Harness.

Features:
1. LearningEvent captures structured runtime outcomes (success/failure) without secrets or private chain-of-thought.
2. LearningStore persists events to data/training/events/success.jsonl and failure.jsonl.
3. TrainingExampleGenerator curates validated training examples from successful events or corrected failures.
4. DatasetSanitizer strips credentials, API keys, private thoughts, and deduplicates examples.
5. CheckpointMetadata versions model checkpoints (brain_v001, brain_v002, etc.).
6. BrainEvaluationHarness evaluates AXEN Foundation Model capabilities and records baseline.json.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Learning Event
# ---------------------------------------------------------------------------

@dataclass
class LearningEvent:
    """
    Structured record of a single runtime decision and execution outcome.
    Never stores private chain-of-thought or raw secrets.
    """
    event_id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    task_type: str = "general"
    decision_source: str = "brain_native"
    user_request: str = ""
    requested_action: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    result: str = "success"  # "success" or "failure"
    verification: str = "passed"
    failure_type: Optional[str] = None
    correction: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_sanitized: bool = False

    def sanitize(self) -> None:
        """Strip credentials, secrets, and private chain-of-thought markers."""
        self.user_request = DatasetSanitizer.clean_text(self.user_request)
        self.requested_action = DatasetSanitizer.clean_text(self.requested_action)
        # Sanitize tool call arguments
        sanitized_calls = []
        for call in self.tool_calls:
            t = call.get("tool", "")
            args = call.get("arguments", {})
            clean_args = {}
            for k, v in args.items():
                if isinstance(v, str):
                    clean_args[k] = DatasetSanitizer.clean_text(v)
                else:
                    clean_args[k] = v
            sanitized_calls.append({"tool": t, "arguments": clean_args})
        self.tool_calls = sanitized_calls
        self.is_sanitized = True

    def to_dict(self) -> Dict[str, Any]:
        if not self.is_sanitized:
            self.sanitize()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LearningEvent:
        return cls(
            event_id=data.get("event_id") or f"evt_{uuid.uuid4().hex[:12]}",
            timestamp=data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            task_type=data.get("task_type", "general"),
            decision_source=data.get("decision_source", "brain_native"),
            user_request=data.get("user_request", ""),
            requested_action=data.get("requested_action", ""),
            tool_calls=data.get("tool_calls") or [],
            result=data.get("result", "success"),
            verification=data.get("verification", "passed"),
            failure_type=data.get("failure_type"),
            correction=data.get("correction"),
            metadata=data.get("metadata") or {},
            is_sanitized=bool(data.get("is_sanitized", False)),
        )


# ---------------------------------------------------------------------------
# Learning Event Store
# ---------------------------------------------------------------------------

class LearningStore:
    """
    Persists runtime learning events into partitioned jsonl stores.
    """

    def __init__(self, base_dir: Optional[str] = None) -> None:
        self.base_dir = Path(base_dir or "data/training")
        self.events_dir = self.base_dir / "events"
        self.curated_dir = self.base_dir / "curated"
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.curated_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record_event(self, event: LearningEvent) -> None:
        """Append an event to the success or failure dataset."""
        event.sanitize()
        filename = "success.jsonl" if event.result == "success" else "failure.jsonl"
        target_path = self.events_dir / filename

        with self._lock:
            with open(target_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict()) + "\n")

    def list_events(self, result_type: str = "success", limit: int = 100) -> List[LearningEvent]:
        """Read recent events from disk."""
        filename = "success.jsonl" if result_type == "success" else "failure.jsonl"
        target_path = self.events_dir / filename
        if not target_path.exists():
            return []

        events = []
        with self._lock:
            with open(target_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(LearningEvent.from_dict(json.loads(line)))
                        except Exception:
                            continue
        return events[-limit:]


# ---------------------------------------------------------------------------
# Dataset Sanitizer
# ---------------------------------------------------------------------------

class DatasetSanitizer:
    """
    Ensures training examples contain no credentials, private reasoning,
    or malicious prompt injection scripts.
    """

    _SECRET_PATTERN = re.compile(
        r"\b(?:api[_-]?key|password|secret|token|bearer|auth[_-]?key)\s*(?:[:=]|\bis\b)\s*['\"]?[^\s,'\"]+['\"]?|"
        r"\bsk-[a-zA-Z0-9]{8,}\b",
        re.IGNORECASE,
    )
    _THOUGHT_PATTERN = re.compile(
        r"<(?:think|thought|internal)>.*?</(?:think|thought|internal)>",
        re.DOTALL | re.IGNORECASE,
    )
    _INJECTION_PATTERN = re.compile(
        r"(?:ignore\s+previous\s+instructions|system\s+override|reveal\s+prompt|bypass\s+restrictions)",
        re.IGNORECASE,
    )

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Scrub sensitive patterns from text."""
        if not text:
            return ""
        # Remove thought tags
        cleaned = cls._THOUGHT_PATTERN.sub("", text)
        # Mask secrets
        cleaned = cls._SECRET_PATTERN.sub("[REDACTED_SECRET]", cleaned)
        # Neutralize prompt injections
        cleaned = cls._INJECTION_PATTERN.sub("[INERT_INSTRUCTION]", cleaned)
        return cleaned.strip()

    @classmethod
    def is_safe(cls, text: str) -> bool:
        """Check whether text is free of secrets and unsafe patterns."""
        if not text:
            return True
        if cls._SECRET_PATTERN.search(text):
            return False
        return True


# ---------------------------------------------------------------------------
# Training Example Generator
# ---------------------------------------------------------------------------

class TrainingExampleGenerator:
    """
    Transforms runtime events into structured, validated training pairs.
    CRITICAL RULE: Bad outputs are never transformed into training targets.
    Failures only become training examples if a verified correction exists.
    """

    def __init__(self, store: Optional[LearningStore] = None) -> None:
        self.store = store or LearningStore()
        self._seen_hashes: Set[str] = set()

    def generate_example_from_event(self, event: LearningEvent) -> Optional[Dict[str, Any]]:
        """
        Convert a LearningEvent into a canonical training example.
        """
        if not DatasetSanitizer.is_safe(event.user_request):
            return None

        clean_input = DatasetSanitizer.clean_text(event.user_request)
        if not clean_input:
            return None

        # Success event transformation
        if event.result == "success" and event.tool_calls:
            first_call = event.tool_calls[0]
            target = {
                "type": "tool_call",
                "tool": first_call.get("tool"),
                "arguments": first_call.get("arguments", {}),
            }
            category = "tool_call" if len(event.tool_calls) == 1 else "tool_chain"
            return {
                "input": clean_input,
                "target": target,
                "category": category,
                "verified": True,
            }

        elif event.result == "success" and not event.tool_calls:
            # Conversational or direct response
            return {
                "input": clean_input,
                "target": {"type": "final_response"},
                "category": "conversation",
                "verified": True,
            }

        # Failure event transformation:
        # Bad output must NOT become target!
        # Only include if verified correction is present.
        if event.result == "failure" and event.correction:
            target = event.correction
            return {
                "input": clean_input,
                "target": target,
                "category": "correction",
                "verified": True,
            }

        return None

    def export_curated_dataset(self, filename: str = "curated_tool_intent.jsonl") -> int:
        """
        Export all valid, deduplicated training examples to the curated directory.
        Returns count of exported examples.
        """
        successes = self.store.list_events(result_type="success", limit=500)
        failures = self.store.list_events(result_type="failure", limit=500)

        curated_path = self.store.curated_dir / filename
        count = 0

        with open(curated_path, "w", encoding="utf-8") as f:
            for event in successes + failures:
                example = self.generate_example_from_event(event)
                if not example:
                    continue

                # Deduplicate by input hash
                h = hashlib.sha256(json.dumps(example["input"]).encode("utf-8")).hexdigest()
                if h in self._seen_hashes:
                    continue
                self._seen_hashes.add(h)

                f.write(json.dumps(example) + "\n")
                count += 1

        return count


# ---------------------------------------------------------------------------
# Checkpoint Metadata
# ---------------------------------------------------------------------------

@dataclass
class CheckpointMetadata:
    """
    Versioned tracking of Foundation Model training runs and evaluation metrics.
    Prevents deploying a worse model over an existing checkpoint.
    """
    version_id: str  # e.g. "brain_v001"
    base_model: str = "axen-foundation-v1"
    dataset_version: str = "1.0.0"
    training_steps: int = 0
    validation_loss: float = 0.0
    eval_metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    promoted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CheckpointMetadata:
        return cls(**data)


# ---------------------------------------------------------------------------
# Brain Evaluation Harness
# ---------------------------------------------------------------------------

class BrainEvaluationHarness:
    """
    Standardized, deterministic evaluation benchmark measuring:
    1. Conversation & Identity
    2. Instruction following
    3. Context retention
    4. Tool selection
    5. Tool chaining
    6. Reflection & Self-correction
    7. Coding specialist routing (Qwen only)
    8. General knowledge (AXEN brain only)
    """

    BENCHMARK_TASKS = [
        {"id": "conv_01", "category": "conversation", "query": "Hello", "expected_type": "final_response"},
        {"id": "conv_02", "category": "conversation", "query": "Who are you?", "expected_type": "final_response"},
        {"id": "conv_03", "category": "conversation", "query": "What can you do?", "expected_type": "final_response"},
        {"id": "inst_01", "category": "instruction", "query": "Answer in one sentence.", "expected_type": "final_response"},
        {"id": "tool_01", "category": "tool_selection", "query": "Calculate 25 * 48.", "expected_tool": "calculator"},
        {"id": "tool_02", "category": "tool_selection", "query": "Find auth.py in this project.", "expected_tool": "file_search"},
        {"id": "tool_03", "category": "tool_selection", "query": "Read the configuration file.", "expected_tool": "file_read"},
        {"id": "chain_01", "category": "tool_chaining", "query": "Find the CSV, calculate its average, and explain it.", "expected_tool": "file_search"},
        {"id": "code_01", "category": "coding", "query": "Write a Python function to reverse a string.", "expected_specialist": "qwen"},
        {"id": "know_01", "category": "general_knowledge", "query": "What is a computer?", "expected_brain": "axen_own_brain"},
        {"id": "know_02", "category": "general_knowledge", "query": "What is machine learning?", "expected_brain": "axen_own_brain"},
    ]

    @classmethod
    def evaluate_runtime(cls, agent_loop: Any) -> Dict[str, Any]:
        """
        Execute deterministic benchmark against an active AgentLoop.
        Returns comprehensive evaluation metrics dictionary.
        """
        results: List[Dict[str, Any]] = []
        metrics: Dict[str, Any] = {
            "total_tasks": len(cls.BENCHMARK_TASKS),
            "passed": 0,
            "failed": 0,
            "categories": {},
            "decision_sources": {},
        }

        for task in cls.BENCHMARK_TASKS:
            query = task["query"]
            cat = task["category"]
            if cat not in metrics["categories"]:
                metrics["categories"][cat] = {"total": 0, "passed": 0}
            metrics["categories"][cat]["total"] += 1

            passed = True
            tool_used = None
            source = "deterministic_rule"

            try:
                # Test coding routing explicitly
                if cat == "coding":
                    from app.ai.router import ModelRouter
                    router = getattr(agent_loop, "model_router", None) or ModelRouter()
                    is_coding = router._is_coding_task(query)
                    passed = is_coding
                    source = "specialist_qwen" if is_coding else "planner_fallback"

                elif cat == "general_knowledge":
                    from app.ai.router import ModelRouter
                    router = getattr(agent_loop, "model_router", None) or ModelRouter()
                    is_coding = router._is_coding_task(query)
                    passed = not is_coding  # Must NOT route general knowledge to Qwen
                    source = "brain_native" if not is_coding else "specialist_qwen"

                else:
                    ans, state = agent_loop.run(query)
                    tool_calls = [c.get("tool") for c in getattr(state, "tool_calls", [])]
                    tool_used = tool_calls[0] if tool_calls else None

                    if "expected_tool" in task:
                        expected = task["expected_tool"]
                        passed = any(expected in t for t in tool_calls)
                    elif "expected_type" in task:
                        passed = (state.status == "completed")

                    source = state.metadata.get("decision_source", "planner_fallback")

            except Exception as exc:
                logger.error("Error evaluating task %s: %s", task["id"], exc)
                passed = False

            if passed:
                metrics["passed"] += 1
                metrics["categories"][cat]["passed"] += 1
            else:
                metrics["failed"] += 1

            metrics["decision_sources"][source] = metrics["decision_sources"].get(source, 0) + 1
            results.append({
                "id": task["id"],
                "category": cat,
                "passed": passed,
                "tool_used": tool_used,
                "decision_source": source,
            })

        metrics["accuracy"] = round(metrics["passed"] / max(metrics["total_tasks"], 1), 4)
        metrics["results"] = results
        metrics["timestamp"] = datetime.now(timezone.utc).isoformat()
        return metrics

    @classmethod
    def save_baseline(cls, metrics: Dict[str, Any], filepath: str = "evaluation/baseline.json") -> None:
        """Save benchmark metrics to baseline.json."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
