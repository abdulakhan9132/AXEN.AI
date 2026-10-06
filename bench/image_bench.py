"""
bench/image_bench.py
====================
Standalone manual benchmark script for AXEN ImageBrain (SD-Turbo preset):
- Measures pipeline load time
- Measures image generation latency (seconds per image)
- Tracks peak CUDA VRAM usage on 4 GB laptop GPUs
- Verifies output image generation and clean memory unloader

Usage:
    python bench/image_bench.py
"""

import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from image_brain import ImageBrain, free_ollama_vram


def run_benchmark():
    print("=" * 65)
    print("AXEN LOCAL IMAGE GENERATION BENCHMARK (SD-TURBO / 4GB VRAM)")
    print("=" * 65)

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    total_vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if cuda_available else 0.0
    )

    print(f"Device:           {device_name}")
    print(f"CUDA Available:   {cuda_available}")
    if cuda_available:
        print(f"Total VRAM:       {total_vram_gb:.2f} GB")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    free_ollama_vram()

    out_dir = Path("outputs/images/bench")
    brain = ImageBrain(preset="turbo", out_dir=out_dir)

    # 1. Pipeline Load Time
    print("\n[1] Measuring Model Load Time...")
    t0_load = time.perf_counter()
    brain._load()
    t_load = time.perf_counter() - t0_load
    print(f"  Model load time: {t_load:.2f} s")

    # 2. Image Generation Latency
    test_prompt = "a futuristic dashboard UI mockup, dark theme, glowing blue accents, 8k render"
    print(f"\n[2] Generating Benchmark Image...")
    print(f"  Prompt: '{test_prompt}'")

    if cuda_available:
        torch.cuda.reset_peak_memory_stats()

    t0_gen = time.perf_counter()
    result = brain.generate(test_prompt, enhance=False, seed=42, n=1)
    t_gen = time.perf_counter() - t0_gen

    print(f"  Generation time: {t_gen:.2f} s")
    print(f"  Saved Image:     {result['paths'][0]}")

    # 3. Peak VRAM Usage
    if cuda_available:
        peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024**2)
        peak_res_mb = torch.cuda.max_memory_reserved() / (1024**2)
        print("\n[3] Peak GPU Memory Metrics:")
        print(f"  Peak Allocated:  {peak_alloc_mb:.1f} MB ({peak_alloc_mb/1024:.2f} GB)")
        print(f"  Peak Reserved:   {peak_res_mb:.1f} MB ({peak_res_mb/1024:.2f} GB)")
        print(f"  Safe for 4GB:    {'YES' if peak_res_mb < 3800 else 'NO'}")

    # 4. Cleanup & Unload Verification
    print("\n[4] Unloading Pipeline...")
    brain.unload()
    if cuda_available:
        rem_alloc_mb = torch.cuda.memory_allocated() / (1024**2)
        print(f"  Post-unload VRAM Allocated: {rem_alloc_mb:.1f} MB")

    print("\n" + "=" * 65)
    print("BENCHMARK COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    run_benchmark()
