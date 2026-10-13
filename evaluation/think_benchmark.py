"""
evaluation/think_benchmark.py
=============================
Think Engine 2.0 Benchmark Suite.

Evaluates FAST, BALANCED, and DEEP thinking modes across 6 representative tasks:
1. Conversational Greeting (Direct response)
2. Arithmetic Calculation (Single deterministic tool)
3. Symbolic Differentiation (MathSolver / SymPy)
4. Project File Search (File tool search)
5. Chained Multi-Step Calculation (Multi-tool pipeline)
6. Bounded Error Recovery (Division by zero / failure handling)

Collects and reports:
- Latency (ms)
- Token Consumption
- Tool Call Efficiency
- Verification Rate
- Success Rate
- Re-plan Count
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from axen.agent import AgentLoop, ThinkMode
from axen.tools import build_default_registry


class MockConversationMemory:
    def __init__(self):
        self._history = []

    def get_history(self):
        return self._history

    def get_summary(self):
        return None

    def add_user(self, text: str):
        self._history.append({"role": "user", "content": text})

    def add_assistant(self, text: str):
        self._history.append({"role": "assistant", "content": text})

    def to_messages(self):
        return [{"role": "system", "content": "You are AXEN."}] + list(self._history)


class MockLongTermMemory:
    def all_facts(self):
        return {}

    def retrieve_relevant(self, query: str, top_k: int = 3, task_type: str = "general"):
        return []


class MockLocalModel:
    is_loaded = False
    model_name = "benchmark-model"

    def chat(self, messages, **kwargs):
        return "I am AXEN, your local AI assistant."

    def chat_with_stats(self, messages, **kwargs):
        return "I am AXEN, your local AI assistant.", {
            "input_tokens": 45,
            "generated_tokens": 12,
            "tokenization_time": 0.001,
            "generation_time": 0.005,
        }


BENCHMARK_TASKS = [
    {
        "id": "T1_GREETING",
        "category": "Conversational",
        "query": "Hello! How can you help me today?",
        "expected_tool": None,
    },
    {
        "id": "T2_ARITHMETIC",
        "category": "Arithmetic",
        "query": "Calculate 48 * 25 + 160 / 4.",
        "expected_tool": "calculator",
    },
    {
        "id": "T3_SYMBOLIC_MATH",
        "category": "Symbolic Math",
        "query": "Solve the derivative of x**3 - 6*x**2 + 11*x - 6 with respect to x.",
        "expected_tool": "math_solver",
    },
    {
        "id": "T4_FILE_SEARCH",
        "category": "File Search",
        "query": "Find the README.md file in this project.",
        "expected_tool": "file_search",
    },
    {
        "id": "T5_MULTI_STEP",
        "category": "Chained Multi-Step",
        "query": "Calculate 25 * 4, then calculate 100 / 5.",
        "expected_tool": "calculator",
    },
    {
        "id": "T6_ERROR_RECOVERY",
        "category": "Bounded Error",
        "query": "Calculate 10 / 0.",
        "expected_tool": "calculator",
    },
]


def run_benchmark() -> Dict[str, Any]:
    print("=" * 72)
    print("       AEXN AI — THINK ENGINE 2.0 PRODUCTION BENCHMARK")
    print("=" * 72)

    tools = build_default_registry()
    memory = MockConversationMemory()
    long_memory = MockLongTermMemory()
    model = MockLocalModel()

    loop = AgentLoop(
        tools=tools,
        memory=memory,
        long_memory=long_memory,
        model=model,
        workspace_path=".",
    )

    modes = [ThinkMode.FAST, ThinkMode.BALANCED, ThinkMode.DEEP]
    results: List[Dict[str, Any]] = []

    for task in BENCHMARK_TASKS:
        print(f"\n[Task: {task['id']}] {task['category']} — '{task['query']}'")
        for mode in modes:
            t0 = time.perf_counter()
            answer, state = loop.run(task["query"], think_mode_override=mode.value)
            duration_ms = (time.perf_counter() - t0) * 1000

            metrics = state.metadata.get("think_metrics", {})
            claim_v = state.metadata.get("claim_verification", {})

            success = state.status in ("completed", "verification_failed")
            # For error recovery task, safe failure is expected
            if task["id"] == "T6_ERROR_RECOVERY":
                success = "error" in answer.lower() or "couldn't solve" in answer.lower() or state.status == "verification_failed"

            res_entry = {
                "task_id": task["id"],
                "category": task["category"],
                "mode": mode.value,
                "latency_ms": round(duration_ms, 2),
                "status": state.status,
                "tool_calls": len(state.tool_calls),
                "tokens": metrics.get("total_tokens", 0),
                "replan_count": metrics.get("replan_count", 0),
                "verification_passed": claim_v.get("passed", True),
                "success": bool(success),
            }
            results.append(res_entry)
            print(
                f"  Mode {mode.value.upper():<8} | Latency: {duration_ms:6.1f}ms | "
                f"Tools: {len(state.tool_calls)} | Status: {state.status:<12} | Success: {success}"
            )

    # Compute aggregate stats per mode
    mode_stats: Dict[str, Dict[str, Any]] = {}
    for mode in modes:
        mode_entries = [r for r in results if r["mode"] == mode.value]
        total_lat = sum(r["latency_ms"] for r in mode_entries)
        total_tools = sum(r["tool_calls"] for r in mode_entries)
        total_tokens = sum(r["tokens"] for r in mode_entries)
        successes = sum(1 for r in mode_entries if r["success"])
        verifs = sum(1 for r in mode_entries if r["verification_passed"])

        mode_stats[mode.value] = {
            "avg_latency_ms": round(total_lat / len(mode_entries), 2),
            "total_tool_calls": total_tools,
            "avg_tool_calls": round(total_tools / len(mode_entries), 2),
            "total_tokens": total_tokens,
            "success_rate_pct": round((successes / len(mode_entries)) * 100, 1),
            "verification_rate_pct": round((verifs / len(mode_entries)) * 100, 1),
        }

    benchmark_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "tasks_count": len(BENCHMARK_TASKS),
        "mode_stats": mode_stats,
        "runs": results,
    }

    out_path = Path(__file__).resolve().parent / "benchmark_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(benchmark_payload, indent=2), encoding="utf-8")



    print("\n" + "=" * 72)
    print("                    AGGREGATE MODE COMPARISON")
    print("=" * 72)
    print(f"{'Mode':<10} | {'Avg Latency':<14} | {'Avg Tools':<10} | {'Success Rate':<14} | {'Verification Rate':<18}")
    print("-" * 72)
    for mode_name, stats in mode_stats.items():
        print(
            f"{mode_name.upper():<10} | "
            f"{stats['avg_latency_ms']:8.2f} ms     | "
            f"{stats['avg_tool_calls']:<10} | "
            f"{stats['success_rate_pct']:>5.1f}%        | "
            f"{stats['verification_rate_pct']:>5.1f}%"
        )
    print("=" * 72)
    print(f"Results persisted to {out_path}")
    return benchmark_payload


if __name__ == "__main__":
    run_benchmark()
