"""
bench/brain_bench.py
====================
Benchmark script for AxenBrain neural intent router:
- Per-intent accuracy
- Confusion matrix
- Inference latency per prediction on CPU (target < 5 ms)
"""

import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from axen_brain import AxenBrain, load_data

def run_benchmark():
    print("=" * 60)
    print("AXEN BRAIN INTENT ROUTER BENCHMARK")
    print("=" * 60)

    brain = AxenBrain("axen_brain.pt")
    if brain.net is None:
        print("Error: axen_brain.pt not found. Run 'python axen_brain.py' first.")
        return

    data = load_data("training/intents.jsonl")
    print(f"Total dataset examples: {len(data)}")

    labels = brain.labels
    label_to_idx = {l: i for i, l in enumerate(labels)}
    n_labels = len(labels)

    # 1. LATENCY BENCHMARK ON CPU
    print("\n[1] Measuring CPU Inference Latency...")
    sample_queries = [t for t, _ in data[50:250]]
    warmup_queries = [t for t, _ in data[:50]]
    for q in warmup_queries:
        brain.predict(q)

    latencies_ms = []
    for q in sample_queries:
        t0 = time.perf_counter()
        brain.predict(q)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    avg_lat = np.mean(latencies_ms)
    p50_lat = np.median(latencies_ms)
    p95_lat = np.percentile(latencies_ms, 95)
    p99_lat = np.percentile(latencies_ms, 99)

    print(f"  Sample count:       {len(sample_queries)}")
    print(f"  Average latency:    {avg_lat:.3f} ms (Target: < 5.0 ms) -> {'PASS' if avg_lat < 5.0 else 'WARN'}")
    print(f"  Median (p50):       {p50_lat:.3f} ms")
    print(f"  95th percentile:    {p95_lat:.3f} ms")
    print(f"  99th percentile:    {p99_lat:.3f} ms")
    print(f"  Throughput:         {1000.0 / avg_lat:.1f} predictions/sec")

    # 2. PER-INTENT ACCURACY & CONFUSION MATRIX
    print("\n[2] Evaluating Intent Classification Accuracy & Confusion Matrix...")
    confusion = np.zeros((n_labels, n_labels + 1), dtype=int)  # extra col for 'unknown'
    per_intent_total = {l: 0 for l in labels}
    per_intent_correct = {l: 0 for l in labels}

    for text, true_label in data:
        if true_label not in label_to_idx:
            continue
        row = label_to_idx[true_label]
        pred = brain.predict(text)
        pred_label = pred["intent"]

        per_intent_total[true_label] += 1
        if pred_label == true_label:
            per_intent_correct[true_label] += 1

        if pred_label in label_to_idx:
            col = label_to_idx[pred_label]
        else:
            col = n_labels  # unknown
        confusion[row, col] += 1

    total_eval = sum(per_intent_total.values())
    total_correct = sum(per_intent_correct.values())
    overall_acc = total_correct / total_eval if total_eval > 0 else 0.0

    print("\nPer-Intent Accuracy Summary:")
    print("-" * 50)
    print(f"{'Intent':<15} | {'Samples':<8} | {'Correct':<8} | {'Accuracy':<10}")
    print("-" * 50)
    for l in labels:
        tot = per_intent_total[l]
        corr = per_intent_correct[l]
        acc = (corr / tot) * 100.0 if tot > 0 else 0.0
        print(f"{l:<15} | {tot:<8} | {corr:<8} | {acc:.2f}%")
    print("-" * 50)
    print(f"{'OVERALL':<15} | {total_eval:<8} | {total_correct:<8} | {overall_acc * 100.0:.2f}%")

    # 3. CONFUSION MATRIX TABLE
    print("\nConfusion Matrix (Rows: True, Columns: Predicted):")
    col_headers = [l[:6] for l in labels] + ["unk"]
    hdr = f"{'True \\ Pred':<12} | " + " | ".join(f"{h:<6}" for h in col_headers)
    print("-" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for i, l in enumerate(labels):
        row_str = " | ".join(f"{confusion[i, j]:<6}" for j in range(n_labels + 1))
        print(f"{l:<12} | {row_str}")
    print("-" * len(hdr))

    return {
        "overall_accuracy": overall_acc,
        "avg_latency_ms": avg_lat,
        "labels": labels,
    }

if __name__ == "__main__":
    run_benchmark()
