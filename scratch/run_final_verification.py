"""
Final runtime verification script for Phase 3.
Measures exact response time, task classification, tool usage, and final answer
for the 5 required messages under full model startup.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from axen.assistant.controller import AssistantController
from axen.model.hf_model import HFModelConfig


def main():
    print("=" * 60)
    print("STARTING FULL AXEN CONTROLLER RUNTIME VERIFICATION")
    print("=" * 60)

    # 1. Startup & model load
    t0 = time.perf_counter()
    config = HFModelConfig.from_yaml()
    controller = AssistantController(config=config)
    controller.load_model()
    load_time = time.perf_counter() - t0
    print(f"STARTUP: Model loaded on CUDA in {load_time:.2f}s")

    test_cases = [
        ("1. Hello, who are you?", "Hello, who are you?"),
        ("2. What is 25 * 48?", "What is 25 * 48?"),
        ("3. Differentiate x^3 + 2x.", "Differentiate x^3 + 2x."),
        ("4. Run: print(2 + 3)", "Run: print(2 + 3)"),
        ("5. Explain recursion in Python.", "Explain recursion in Python."),
    ]

    results = []

    for label, query in test_cases:
        print(f"\nEvaluating: {label}")
        start_t = time.perf_counter()
        answer = controller.respond(query)
        elapsed = time.perf_counter() - start_t

        dbg = controller.last_debug_info
        task_type = dbg.get("task_type", "unknown")
        tool_used = dbg.get("tool_used", False)
        selected_tool = dbg.get("selected_tool") or "none"

        print(f"  Response time      : {elapsed:.2f}s")
        print(f"  Task classification: {task_type}")
        print(f"  Tool used          : {tool_used} ({selected_tool})")
        print(f"  Final answer       : {answer[:120]}..." if len(answer) > 120 else f"  Final answer       : {answer}")

        results.append({
            "label": label,
            "query": query,
            "elapsed": elapsed,
            "task_type": task_type,
            "tool_used": tool_used,
            "selected_tool": selected_tool,
            "answer": answer,
        })

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"{r['label']} | Time: {r['elapsed']:.2f}s | Task: {r['task_type']} | Tool: {r['selected_tool']} ({r['tool_used']})")


if __name__ == "__main__":
    main()
