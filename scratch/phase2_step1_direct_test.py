import json
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from axen.model.foundation import AXENFoundationModel

prompts = [
    "Hello, who are you?",
    "What is artificial intelligence? Explain it in simple words.",
    "What is a computer?",
    "Why is the sky blue?",
    "Write a Python function that adds two numbers."
]

checkpoint_path = Path("checkpoints/foundation/brain_v004_best.pt")
print(f"Loading checkpoint: {checkpoint_path}")
model, meta = AXENFoundationModel.load_checkpoint(checkpoint_path, device="cuda" if torch.cuda.is_available() else "cpu")

results = []
print("--- Direct Foundation Model Outputs ---")
for i, p in enumerate(prompts, 1):
    raw = model.chat([{"role": "user", "content": p}], max_new_tokens=150, temperature=0.0)
    print(f"\n[{i}] Prompt: {p}")
    print(f"    Raw output: {repr(raw)}")
    results.append({
        "id": i,
        "prompt": p,
        "raw_output": raw
    })

Path("scratch/phase2_step1_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("\nSaved to scratch/phase2_step1_results.json")
