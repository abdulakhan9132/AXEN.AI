"""
scratch/benchmark_phase1.py
===========================
Executes Phase 1 Performance Benchmark on local hardware:
- RTX 3050 Laptop GPU, 4GB VRAM
- TinyLlama 1.1B Chat on CUDA with torch.inference_mode()
- Deterministic tool bypasses and adaptive token limits
"""

import os
import sys
import time
import json
import torch

from axen.assistant.controller import AssistantController
from axen.model.hf_model import HfModel
from axen.tools import build_default_registry
from axen.memory.conversation_memory import ConversationMemory
from axen.memory.long_term_memory import LongTermMemory


def run_benchmark():
    print("=" * 60)
    print("AXEN PHASE 1 PERFORMANCE BENCHMARK")
    print("=" * 60)

    # 1. Inspect GPU & System
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    print(f"CUDA Available: {cuda_available}")
    print(f"Device: {device_name}")

    # 2. Model Loading
    t0 = time.perf_counter()
    model = HfModel()
    load_time = time.perf_counter() - t0
    mem_info = model.get_memory_info()
    print(f"Model Load Time: {load_time:.2f}s")
    print(f"Model Device: {mem_info.get('device')}, Dtype: {mem_info.get('dtype')}")
    print(f"VRAM Allocated: {mem_info.get('vram_allocated_mb', 0):.1f} MB, Reserved: {mem_info.get('vram_reserved_mb', 0):.1f} MB")
    print("-" * 60)

    # 3. Initialize Controller
    memory = ConversationMemory()
    long_memory = LongTermMemory()
    registry = build_default_registry()
    controller = AssistantController(
        model=model,
        tool_registry=registry,
        conversation_memory=memory,
        long_term_memory=long_memory,
    )

    # Warmup
    print("Warming up model...")
    controller.respond("hi")
    memory.clear()

    queries = [
        ("1. Greeting", "hi"),
        ("2. Identity", "Who are you?"),
        ("3. Calculator", "25 * 48"),
        ("4. Computer Systems", "Convert 42 to binary"),
        ("5. Simple Math", "Differentiate: f(x)=(2x^3-5x+4)(x^2+3x-1)"),
        ("6. Short Programming", "Generate Python code to print Hello World."),
        ("7. Medium Natural Lang", "Explain recursion in Python"),
    ]

    results = []

    for label, query in queries:
        memory.clear()
        first_ui_time = None
        t_req_start = time.perf_counter()

        def _state_cb(state_name, display_text, meta):
            nonlocal first_ui_time
            if first_ui_time is None:
                first_ui_time = time.perf_counter() - t_req_start

        response = controller.respond(query, state_callback=_state_cb)
        total_time = time.perf_counter() - t_req_start

        state = controller.last_agent_state
        perf = state.metadata.get("perf_timings", {})
        task_type = state.task_type

        class_time = perf.get("classification", 0.0)
        tool_time = perf.get("tools", 0.0)
        gen_time = perf.get("generation", 0.0)
        planning_time = perf.get("planning", 0.0)
        first_ui_str = f"{first_ui_time*1000:.1f}ms" if first_ui_time else "N/A"

        stats = state.metadata.get("model_stats", {})
        gen_tokens = stats.get("generated_tokens", "N/A")

        res_item = {
            "label": label,
            "query": query,
            "task_type": task_type,
            "total_s": round(total_time, 4),
            "class_s": round(class_time, 4),
            "tool_s": round(tool_time, 4),
            "gen_s": round(gen_time, 4),
            "planning_s": round(planning_time, 4),
            "first_ui": first_ui_str,
            "tokens": gen_tokens,
            "snippet": response.strip().replace("\n", " ")[:60],
        }
        results.append(res_item)
        print(f"[{label}] Total: {total_time:.3f}s | Gen: {gen_time:.3f}s | Tool: {tool_time:.3f}s | First UI: {first_ui_str} | Task: {task_type}")

    print("\n" + "=" * 80)
    print("DETAILED BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Query':<24} | {'Task':<16} | {'Class':<8} | {'Tool':<8} | {'Gen':<8} | {'Total':<8} | {'First UI':<10}"
    print(header)
    print("-" * 80)
    for r in results:
        row = f"{r['label']:<24} | {r['task_type']:<16} | {r['class_s']:<8.4f} | {r['tool_s']:<8.4f} | {r['gen_s']:<8.4f} | {r['total_s']:<8.4f} | {r['first_ui']:<10}"
        print(row)
    print("=" * 80)

    # Save to json
    out_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_info": mem_info,
            "load_time_s": load_time,
            "results": results
        }, f, indent=2)
    print(f"Saved benchmark results to {out_path}")


if __name__ == "__main__":
    run_benchmark()
