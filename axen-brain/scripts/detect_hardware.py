"""Hardware Detection and Model Sizing Script for AXEN Brain.

Detects CPU, RAM, GPU, and VRAM. Recommends model presets and batch sizes,
provides realistic training time estimates, and outlines expected model quality.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import psutil
import torch


@dataclass
class HardwareSpecs:
    cpu_name: str
    physical_cores: int
    logical_cores: int
    total_ram_gb: float
    available_ram_gb: float
    gpu_available: bool
    gpu_name: Optional[str] = None
    vram_gb: Optional[float] = None
    cuda_version: Optional[str] = None
    bf16_supported: bool = False
    fp16_supported: bool = False


@dataclass
class PresetEstimate:
    preset_name: str
    param_count_m: float
    context_length: int
    recommended_micro_batch: int
    recommended_grad_accum: int
    effective_batch_size: int
    estimated_vram_gb: float
    can_fit_in_vram: bool
    estimated_tokens_per_sec: float
    training_time_fast_poc: str
    training_time_chinchilla: str
    quality_expectation: str


def detect_system_hardware() -> HardwareSpecs:
    """Inspect the local machine and return detailed hardware specifications."""
    physical_cores = psutil.cpu_count(logical=False) or 1
    logical_cores = psutil.cpu_count(logical=True) or 1
    vm = psutil.virtual_memory()
    total_ram_gb = round(vm.total / (1024**3), 2)
    available_ram_gb = round(vm.available / (1024**3), 2)

    cpu_name = platform.processor() or platform.machine() or "Generic CPU"

    gpu_available = torch.cuda.is_available()
    gpu_name = None
    vram_gb = None
    cuda_version = None
    bf16_supported = False
    fp16_supported = False

    if gpu_available:
        try:
            device_props = torch.cuda.get_device_properties(0)
            gpu_name = device_props.name
            vram_gb = round(device_props.total_memory / (1024**3), 2)
            cuda_version = torch.version.cuda or "Unknown"
            # BF16 is generally supported on Ampere (sm_80)+ architectures
            bf16_supported = torch.cuda.is_bf16_supported() if hasattr(torch.cuda, "is_bf16_supported") else False
            fp16_supported = True
        except Exception:
            gpu_available = False

    return HardwareSpecs(
        cpu_name=cpu_name,
        physical_cores=physical_cores,
        logical_cores=logical_cores,
        total_ram_gb=total_ram_gb,
        available_ram_gb=available_ram_gb,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        cuda_version=cuda_version,
        bf16_supported=bf16_supported,
        fp16_supported=fp16_supported,
    )


def compute_estimates_and_recommendations(hw: HardwareSpecs) -> Dict[str, Any]:
    """Calculate sizing, memory consumption, speeds, and recommendations."""
    vram = hw.vram_gb or 0.0
    has_gpu = hw.gpu_available

    # Approximate training throughput based on hardware
    if has_gpu:
        if vram >= 8.0:
            tiny_tps = 45000.0
            small_tps = 18000.0
            medium_tps = 8000.0
        else:  # e.g., 4GB VRAM Laptop GPU like RTX 3050
            tiny_tps = 30000.0
            small_tps = 11000.0
            medium_tps = 4500.0
    else:
        # CPU-only baseline
        tiny_tps = 1000.0
        small_tps = 250.0
        medium_tps = 90.0

    def format_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{int(seconds)} seconds"
        elif seconds < 3600:
            return f"{int(seconds / 60)} minutes"
        elif seconds < 86400:
            hours = seconds / 3600
            return f"{hours:.1f} hours"
        else:
            days = seconds / 86400
            return f"{days:.1f} days"

    # Presets definition
    # 1. Tiny preset: ~10M params
    tiny_params = 10.0
    tiny_vram = 1.2 if has_gpu else 0.0
    tiny_micro_batch = 16 if has_gpu else 4
    tiny_grad_accum = 2 if has_gpu else 4
    tiny_poc_tokens = 20_000_000  # 20M tokens for fast prototype
    tiny_chinchilla_tokens = 200_000_000  # 20 tokens per param = 200M

    # 2. Small preset: ~50M params
    small_params = 49.5
    small_vram = 2.8 if has_gpu else 0.0
    small_micro_batch = 4 if has_gpu else 2
    small_grad_accum = 4 if has_gpu else 8
    small_poc_tokens = 50_000_000  # 50M tokens
    small_chinchilla_tokens = 1_000_000_000  # 1B tokens

    # 3. Medium preset: ~120M params
    medium_params = 124.2
    # Medium with AdamW (4 bytes weights + 4 bytes grads + 8 bytes optimizer state = 16 bytes/param ~ 2GB pure state, plus activations)
    medium_vram = 4.8 if has_gpu else 0.0
    medium_micro_batch = 2 if has_gpu else 1
    medium_grad_accum = 8 if has_gpu else 16
    medium_poc_tokens = 100_000_000  # 100M tokens
    medium_chinchilla_tokens = 2_480_000_000  # 2.48B tokens

    presets = [
        PresetEstimate(
            preset_name="tiny (~10M)",
            param_count_m=tiny_params,
            context_length=512,
            recommended_micro_batch=tiny_micro_batch,
            recommended_grad_accum=tiny_grad_accum,
            effective_batch_size=tiny_micro_batch * tiny_grad_accum,
            estimated_vram_gb=tiny_vram,
            can_fit_in_vram=(tiny_vram <= vram) if has_gpu else True,
            estimated_tokens_per_sec=tiny_tps,
            training_time_fast_poc=format_duration(tiny_poc_tokens / tiny_tps),
            training_time_chinchilla=format_duration(tiny_chinchilla_tokens / tiny_tps),
            quality_expectation=(
                "Basic syntax, grammar, short-story coherence (TinyStories level), "
                "memorizes short facts, simple vocabulary. Cannot do complex multi-step reasoning."
            ),
        ),
        PresetEstimate(
            preset_name="small (~50M)",
            param_count_m=small_params,
            context_length=1024,
            recommended_micro_batch=small_micro_batch,
            recommended_grad_accum=small_grad_accum,
            effective_batch_size=small_micro_batch * small_grad_accum,
            estimated_vram_gb=small_vram,
            can_fit_in_vram=(small_vram <= vram) if has_gpu else False,
            estimated_tokens_per_sec=small_tps,
            training_time_fast_poc=format_duration(small_poc_tokens / small_tps),
            training_time_chinchilla=format_duration(small_chinchilla_tokens / small_tps),
            quality_expectation=(
                "Good multi-paragraph coherence, simple dialogues, follows basic instructions, "
                "basic topic consistency. Still prone to hallucinations on factual queries."
            ),
        ),
        PresetEstimate(
            preset_name="medium (~120M)",
            param_count_m=medium_params,
            context_length=1024,
            recommended_micro_batch=medium_micro_batch,
            recommended_grad_accum=medium_grad_accum,
            effective_batch_size=medium_micro_batch * medium_grad_accum,
            estimated_vram_gb=medium_vram,
            can_fit_in_vram=(medium_vram <= vram) if has_gpu else False,
            estimated_tokens_per_sec=medium_tps,
            training_time_fast_poc=format_duration(medium_poc_tokens / medium_tps),
            training_time_chinchilla=format_duration(medium_chinchilla_tokens / medium_tps),
            quality_expectation=(
                "Solid conversational ability, simple arithmetic, basic logic and reasoning with <think> tags, "
                "fluent English. Close to GPT-2 small quality."
            ),
        ),
    ]

    # Decide recommendations
    if not has_gpu:
        recommended_preset = "tiny"
        recommendation_reason = (
            "No CUDA GPU detected. The 'tiny' preset runs on CPU with ~1,000 tokens/sec. "
            "A quick 20M token prototype takes ~5.5 hours."
        )
    elif vram < 5.0:
        recommended_preset = "tiny"
        recommendation_reason = (
            f"Detected {hw.gpu_name} with {vram:.1f} GB VRAM. "
            "The 'tiny' preset (~10M params) fits comfortably in memory (uses ~1.2 GB VRAM) "
            "and allows fast experimentation (20M token prototype in ~11 minutes!). "
            "The 'small' preset (~50M params) can also run using gradient accumulation "
            "(micro-batch 4, ~2.8 GB VRAM), while 'medium' will likely trigger Out-Of-Memory (OOM) "
            "unless gradient checkpointing or 8-bit Adam is enabled."
        )
    elif vram < 10.0:
        recommended_preset = "small"
        recommendation_reason = (
            f"Detected {hw.gpu_name} with {vram:.1f} GB VRAM. "
            "The 'small' preset (~50M params) is optimal for high quality without risk of OOM."
        )
    else:
        recommended_preset = "medium"
        recommendation_reason = (
            f"Detected {hw.gpu_name} with {vram:.1f} GB VRAM. "
            "The 'medium' preset (~120M params) can be trained efficiently."
        )

    return {
        "hardware": asdict(hw),
        "recommended_preset": recommended_preset,
        "recommendation_reason": recommendation_reason,
        "presets": [asdict(p) for p in presets],
    }


def print_report(data: Dict[str, Any]) -> None:
    """Print an aesthetic, readable terminal report."""
    hw = data["hardware"]
    print("=" * 70)
    print("  AXEN BRAIN — HARDWARE DETECTION & MODEL SIZING REPORT")
    print("=" * 70)
    print(f"  CPU Cores       : {hw['physical_cores']} physical, {hw['logical_cores']} logical")
    print(f"  System RAM      : {hw['total_ram_gb']:.1f} GB total ({hw['available_ram_gb']:.1f} GB available)")
    if hw["gpu_available"]:
        print(f"  GPU Device      : {hw['gpu_name']}")
        print(f"  GPU VRAM        : {hw['vram_gb']:.1f} GB")
        print(f"  CUDA Version    : {hw['cuda_version']}")
        print(f"  Precision Support: FP16: {hw['fp16_supported']} | BF16: {hw['bf16_supported']}")
    else:
        print("  GPU Device      : None (CUDA not available — running on CPU)")
    print("-" * 70)

    print(f"\n>>> PRIMARY RECOMMENDATION: [{data['recommended_preset'].upper()} PRESET]")
    print(f"    {data['recommendation_reason']}\n")

    print("=" * 70)
    print("  CONFIG PRESETS BREAKDOWN & HONEST EXPECTATIONS")
    print("=" * 70)
    for p in data["presets"]:
        fit_status = "Fits in VRAM [OK]" if p["can_fit_in_vram"] else "May OOM on GPU [WARNING]"
        print(f"\n[*] Preset: {p['preset_name'].upper()}")
        print(f"    Parameters         : {p['param_count_m']:.1f} Million")
        print(f"    Context Window     : {p['context_length']} tokens")
        print(f"    Micro-Batch Size   : {p['recommended_micro_batch']} (accum={p['recommended_grad_accum']}, eff_batch={p['effective_batch_size']})")
        print(f"    Estimated VRAM     : ~{p['estimated_vram_gb']:.1f} GB ({fit_status})")
        print(f"    Throughput Rate    : ~{int(p['estimated_tokens_per_sec']):,} tokens/sec")
        print(f"    Prototype Run      : {p['training_time_fast_poc']} (quick proof of concept)")
        print(f"    Full 20x Token Run : {p['training_time_chinchilla']} (Chinchilla compute-optimal)")
        print(f"    Expected Quality   : {p['quality_expectation']}")

    print("\n" + "=" * 70)
    print("  NEXT STEPS")
    print("=" * 70)
    print("  1. Inspect configs in 'configs/' (tiny.yaml, small.yaml, medium.yaml).")
    print("  2. To start training, start with 'tiny' preset to verify the full pipeline,")
    print("     then scale up to 'small' once data and tokenizer are validated.")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="AXEN Brain Hardware Detection & Presets Recommender")
    parser.add_argument("--json", action="store_true", help="Output specifications as formatted JSON")
    args = parser.parse_args()

    hw = detect_system_hardware()
    data = compute_estimates_and_recommendations(hw)

    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print_report(data)


if __name__ == "__main__":
    main()
