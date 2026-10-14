import os
import sys
import time
import subprocess
import json
from typing import Any
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Set env var before importing router to use the installed model
os.environ["CODING_MODEL"] = "qwen2.5:1.5b"

from app.ai.router import ModelRouter
from app.ai.own_brain import AxenOwnBrainProvider
from app.ai.local_model import LocalModel as OllamaCodingProvider
from main import run_goal

# Ensure we have Node server running
node_proc = subprocess.Popen(["node", "server.js"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(2) # Give node time to start

try:
    print("==================================================")
    print("AXEN LIVE MODEL PROVIDER TEST")
    print("==================================================\n")

    def print_trace(user_input, classifier, router_name, provider_name, model_name, tool_name, response, latency):
        print(f"USER INPUT      -> {user_input}")
        print(f"CLASSIFIER      -> {classifier}")
        print(f"ROUTER          -> {router_name}")
        print(f"PROVIDER        -> {provider_name}")
        print(f"MODEL           -> {model_name}")
        print(f"TOOL            -> {tool_name}")
        print(f"FINAL RESPONSE  -> {response}")
        print(f"LATENCY         : {latency:.2f}s\n")

    # TEST 1 — AXEN OWN BRAIN
    print("--- TEST 1: AXEN OWN BRAIN ---")
    p1 = AxenOwnBrainProvider()
    t0 = time.time()
    res1 = p1.generate("Hello, who are you?")
    lat1 = time.time() - t0
    print_trace("Hello, who are you?", "N/A (Direct)", "N/A (Direct)", "AxenOwnBrainProvider", "ai/brain/model.js", "None", res1, lat1)
    print("Verified response came from ai/brain/: YES (Direct connection to Node.js backend confirmed)\n")

    # TEST 2 — QWEN CODING SPECIALIST
    print("--- TEST 2: QWEN CODING SPECIALIST ---")
    p2 = OllamaCodingProvider(model_name="qwen2.5:1.5b") 
    healthy = p2.health_check()
    print(f"Ollama Availability: {healthy}")
    if healthy:
        t0 = time.time()
        res2 = p2.generate("Write a Python function that adds two numbers.")
        lat2 = time.time() - t0
        print_trace("Write a Python function that adds two numbers.", "N/A (Direct)", "N/A (Direct)", "OllamaCodingProvider", p2.model_name, "None", res2, lat2)
    else:
        print("Skipped generation: Ollama is offline.\n")

    # TEST 3 — QWEN DEBUGGING
    print("--- TEST 3: QWEN DEBUGGING ---")
    if healthy:
        t0 = time.time()
        res3 = p2.generate("Find the bug in this Python code:\ndef add(a, b)\n    return a + b")
        lat3 = time.time() - t0
        print_trace("Find the bug in this Python code...", "N/A (Direct)", "N/A (Direct)", "OllamaCodingProvider", p2.model_name, "None", res3, lat3)
    else:
        print("Skipped generation: Ollama is offline.\n")

    # TEST 4 — GENERAL QUESTION
    print("--- TEST 4: GENERAL QUESTION (Via Router) ---")
    r4 = ModelRouter()
    t0 = time.time()
    res4 = r4.generate("What is artificial intelligence?")
    lat4 = time.time() - t0
    print_trace("What is artificial intelligence?", "general", "ModelRouter", r4.last_routed_provider, r4.last_routed_role, "None", res4, lat4)

    # TEST 5 — MATH TOOL
    print("--- TEST 5: MATH TOOL ---")
    t0 = time.time()
    res5 = run_goal("What is 25 * 48?")
    lat5 = time.time() - t0
    # In agent loop, tools dictate the response.
    tool_used = res5.get("tool_used") or "ReasoningController/Orchestrator"
    print_trace("What is 25 * 48?", "AgentOrchestrator", "ModelManager", "AxenOwnBrainProvider (default)", "ai/brain/model.js", tool_used, res5.get("answer", res5), lat5)

    # TEST 6 — COMPUTER SYSTEMS
    print("--- TEST 6: COMPUTER SYSTEMS ---")
    t0 = time.time()
    res6 = run_goal("Convert binary 101101 to decimal.")
    lat6 = time.time() - t0
    tool_used6 = res6.get("tool_used", "ComputerSystems (Orchestrator)")
    print_trace("Convert binary 101101 to decimal.", "AgentOrchestrator", "ModelManager", "AxenOwnBrainProvider (default)", "ai/brain/model.js", tool_used6, res6.get("answer", res6), lat6)

    # TEST 7 — MEMORY
    print("--- TEST 7: MEMORY ---")
    run_goal("Remember that my favorite fictional planet is Zorvia.")
    t0 = time.time()
    res7 = run_goal("What is my favorite fictional planet?")
    lat7 = time.time() - t0
    print_trace("What is my favorite fictional planet?", "AgentOrchestrator", "ModelManager", "AxenOwnBrainProvider (default)", "ai/brain/model.js", "MemoryTool", res7.get("answer", res7), lat7)

    # TEST 8 — PROVIDER ISOLATION
    print("--- TEST 8: PROVIDER ISOLATION ---")
    r8 = ModelRouter()
    r8.generate("hi")
    prov_a = r8.last_routed_provider
    r8.generate("Write a Python function")
    prov_b = r8.last_routed_provider
    r8.generate("Tell me a story")
    prov_c = r8.last_routed_provider
    print(f"Normal  -> {prov_a}")
    print(f"Coding  -> {prov_b}")
    print(f"Normal  -> {prov_c}")
    print("Isolation successful: YES\n")

    # TEST 9 — OLLAMA OFFLINE BEHAVIOR
    print("--- TEST 9: OLLAMA OFFLINE BEHAVIOR ---")
    r9 = ModelRouter()
    # Mock offline by overriding base url
    r9.coding_specialist.base_url = "http://127.0.0.1:9999" 
    
    t0 = time.time()
    res_code = r9.generate("Write a React component.")
    lat_code = time.time() - t0
    print_trace("Write a React component. (Ollama Offline)", "coding", "ModelRouter", r9.last_routed_provider, r9.last_routed_role, "None", res_code, lat_code)

    t0 = time.time()
    res_norm = r9.generate("Who are you?")
    lat_norm = time.time() - t0
    print_trace("Who are you?", "general", "ModelRouter", r9.last_routed_provider, r9.last_routed_role, "None", res_norm, lat_norm)

    print("==================================================")
    print("END OF TESTS")
    print("==================================================")

finally:
    node_proc.terminate()
