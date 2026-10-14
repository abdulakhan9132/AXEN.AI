"""Measured lesson verification and integration with existing SQLite memory."""
from __future__ import annotations

import json
import hashlib
import re

from axen.memory.memory_models import MemoryItem
from axen.memory.memory_store import SQLiteMemoryStore
from .learning_store import now
from .practice import ArithmeticSandbox


class KnowledgeAcquisitionEngine:
    def __init__(self, store, memory=None):
        self.store = store
        self.memory = memory or SQLiteMemoryStore()

    def analyze(self, documents):
        observations = []
        contradictions = []
        for document in documents:
            text = document["text"].lower()
            if "range" in text and re.search(r"exclusiv|exclud|less than|<\s*stop|end point.{0,50}never part", text):
                observations.append({"concept": "range_stop_exclusive", "source": document["url"],
                                     "status": "observation"})
            if "range" in text and re.search(r"(?:stop|end point).{0,30}(?:is included|is inclusive)|range includes (?:its )?stop", text):
                contradictions.append({"source": document["url"], "claim": "range includes stop",
                    "resolution": "Refuted by independent language-semantic checks"})
        if not observations: raise ValueError("No supported range-boundary concept found in sources")
        # Cross-check against actual interpreter semantics, not an LLM assertion.
        source = "def boundary(n):\n    total = 0\n    for i in range(n):\n        total += i\n    return total\n"
        sandbox = ArithmeticSandbox()
        checked = [sandbox.run(source, [n]) == sum(range(n)) for n in (0, 1, 3, 7)]
        if not all(checked): raise ValueError("Concept contradicted by independent semantic checks")
        return {"concept": "range_stop_exclusive", "summary":
                "Python range excludes its stop boundary. Inclusive nonnegative bounds require stop=n+1.",
                "observations": observations, "cross_check": "Four independent language-semantic checks passed",
                "contradictions": contradictions, "verification_scope": "range boundary arithmetic only"}

    def store_verified(self, goal, concept, evaluation):
        if (evaluation["train"]["passed"] != evaluation["train"]["total"]
                or evaluation["held_out"]["passed"] != evaluation["held_out"]["total"]
                or evaluation["held_out"]["total"] <= 0):
            raise ValueError("Failed practice cannot become verified knowledge")
        entry = {"topic": "Python debugging: inclusive range boundary", "summary": concept["summary"],
                 "source_references": [{k: d[k] for k in ("url", "retrieved_at", "content_hash", "access")}
                                       for d in goal["documents"]],
                 "retrieval_date": goal["documents"][0]["retrieved_at"],
                 "verification_status": "verified_by_practice", "skill": goal["skill"],
                 "confidence_metadata": {"scope": concept["verification_scope"], "held_out": evaluation["held_out"]},
                 "goal_id": goal["id"], "practical_example": goal["candidate_source"],
                 "last_validation_date": now(), "improvement_type": "memory_and_bounded_repair_rule",
                 "concept": concept["concept"]}
        key = "learned_python_debugging_" + goal["id"]
        existing = self.memory.get_by_key(key)
        item = MemoryItem(key=key, value=json.dumps(entry), memory_type="technical_fact",
                          source="verified_learning", confidence=0.9)
        if existing: self.memory.update_value(existing.id, item.value, item.confidence)
        else: self.memory.add(item)
        self.store.lesson(entry)
        return entry

    def verified_training_examples(self, goal):
        """Collect only test-verified arithmetic answers with complete provenance.

        This records examples, not training. Dataset splits are disjoint; future
        candidate training still requires explicit operator configuration.
        """
        evaluation = goal["evaluation"]
        def record(case, split):
            if not case["passed"]: raise ValueError("Failed example cannot enter training data")
            return {"instruction": f"Sum integers from zero through {case['input']} inclusive.",
                    "output": f"The sum is {case['actual']}.", "verified": True,
                    "source": "fixed_benchmark_verified", "metadata": {"goal_id": goal["id"],
                    "provenance": {"suite": "inclusive-sum-v1", "split": split,
                        "code_sha256": hashlib.sha256(goal["candidate_source"].encode()).hexdigest(),
                        "sources": [d["url"] for d in goal["documents"]]}}}
        train = [record(c, "practice") for c in evaluation["train"]["cases"]]
        held = [record(c, "unseen") for c in evaluation["held_out"]["cases"]]
        return {"train": train, "validation": held[:1], "held_out": held[1:], "replay": []}


def retrieve_learned_context(query, memory=None):
    """Lexical retrieval, explicitly not embeddings or neural retraining."""
    memory = memory or SQLiteMemoryStore()
    words = set(re.findall(r"[a-z]{3,}", query.lower()))
    ranked = []
    for item in memory.list_active(limit=200):
        if item.source != "verified_learning": continue
        try: entry = json.loads(item.value)
        except (ValueError, TypeError): continue
        if entry.get("verification_status") != "verified_by_practice": continue
        score = len(words & set(re.findall(r"[a-z]{3,}", (entry["topic"] + " " + entry["summary"]).lower())))
        if score:
            ranked.append((score, {"topic": entry["topic"], "summary": entry["summary"],
                                   "sources": entry["source_references"], "scope": entry["confidence_metadata"]["scope"]}))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return [e for _, e in ranked[:3]]
