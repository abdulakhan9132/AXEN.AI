import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import torch
from axen.assistant.controller import AssistantController

ctrl = AssistantController()
t0 = time.perf_counter()
ctrl.load_model()
load_time = time.perf_counter() - t0

print("=== MODEL LOAD REPORT ===")
print(f"Load time : {load_time:.2f}s")
dev = getattr(ctrl.model, "_device", "unknown")
print(f"Device    : {dev}")
print(f"Dtype     : {ctrl.model.config.dtype}")
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    print(f"CUDA VRAM : {allocated:.1f} MB allocated, {reserved:.1f} MB reserved")

# Test inference
queries = [
    "hi",
    "Who are you?",
    "25 * 48",
    "Convert 42 to binary",
    "differentiate x**2 + 3*x",
    "Generate Python code to print Hello World.",
    "Explain recursion in Python concisely.",
]

print("\n=== QUERIES BENCHMARK ===")
for q in queries:
    t_start = time.perf_counter()
    resp = ctrl.respond(q)
    total_time = time.perf_counter() - t_start
    state = ctrl.last_agent_state
    tt = state.task_type if state else "unknown"
    timings = state.metadata.get("perf_timings", {}) if state else {}
    print(f"Q: {q}")
    print(f"   Task: {tt}")
    print(f"   Resp: {resp[:100]!r}...")
    print(f"   Total: {total_time:.4f}s | Timings: {timings}\n")
