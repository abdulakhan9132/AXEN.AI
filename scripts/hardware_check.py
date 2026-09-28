"""AXEN Hardware Detection & Capability Estimator.

Inspects CPU, RAM, GPU, and VRAM to recommend optimal model sizes for both:
1. Local open-weights model inference (via Ollama / GGUF).
2. From-scratch PyTorch transformer training.
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
class HardwareProfile:
    cpu_model: str
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


def detect_hardware() -> HardwareProfile:
    """Collect hardware metrics using psutil and torch."""
    physical = psutil.cpu_count(logical=False) or 1
    logical = psutil.cpu_count(logical=True) or 1
    ram = psutil.virtual_memory()

    cpu_name = platform.processor() or platform.machine() or "Generic CPU"

    gpu_available = torch.cuda.is_available()
    gpu_name = None
    vram_gb = None
    cuda_version = None
    bf16 = False
    fp16 = False

    if gpu_available:
        try:
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            vram_gb = round(props.total_memory / (1024**3), 2)
            cuda_version = torch.version.cuda
            bf16 = torch.cuda.is_bf16_supported() if hasattr(torch.cuda, "is_bf16_supported") else False
            fp16 = True
        except Exception:
            gpu_available = False

    return HardwareProfile(
        cpu_model=cpu_name,
        physical_cores=physical,
        logical_cores=logical,
        total_ram_gb=round(ram.total / (1024**3), 2),
        available_ram_gb=round(ram.available / (1024**3), 2),
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        cuda_version=cuda_version,
        bf16_supported=bf16,
        fp16_supported=fp16,
    )


def evaluate_capabilities(hw: HardwareProfile) -> Dict[str, Any]:
    """Generate recommendations based on real hardware limits."""
    vram = hw.vram_gb or 0.0

    # 1. Inference recommendations (Ollama / Local open-weights)
    if hw.gpu_available and vram >= 10.0:
        recommended_inference = "Qwen2.5-7B (4-bit Q4_K_M) or Llama-3.1-8B"
        inference_notes = "Plenty of VRAM for 7B/8B models with fast token generation (>30 tok/s)."
    elif hw.gpu_available and vram >= 3.5:
        recommended_inference = "Qwen2.5-3B (or Qwen2.5-1.5B for maximum speed)"
        inference_notes = (
            f"With {vram:.1f} GB VRAM, Qwen2.5-3B fits completely in GPU memory (~2.2 GB VRAM). "
            "Qwen2.5-7B requires ~4.5 GB, which will partially offload to system RAM and run slower."
        )
    else:
        recommended_inference = "Qwen2.5-1.5B or Qwen2.5-0.5B (CPU / low memory)"
        inference_notes = "Lightweight models run comfortably on CPU or integrated graphics."

    # 2. Training-from-scratch recommendations (PyTorch)
    if not hw.gpu_available:
        recommended_train_preset = "tiny"
        train_notes = "CPU training is slow (~800 tok/s). Use tiny preset (~10M params) for learning."
    elif vram <= 4.5:
        recommended_train_preset = "tiny (~10M) for fast iterations, small (~50M) with gradient accumulation"
        train_notes = (
            f"GPU has {vram:.1f} GB VRAM. The ~10M model trains at ~30k tok/s and finishes 20M tokens in ~11 min. "
            "The ~50M model needs micro-batch=4, grad_accum=4 to fit in ~2.8 GB VRAM. "
            "The ~120M model will OOM unless gradient checkpointing is enabled."
        )
    else:
        recommended_train_preset = "small (~50M) or medium (~120M)"
        train_notes = f"{vram:.1f} GB VRAM can accommodate standard batch sizes."

    # RAM warning if available memory is below 1.5 GB
    ram_warning = None
    if hw.available_ram_gb < 1.5:
        ram_warning = (
            f"Available system RAM is low ({hw.available_ram_gb:.1f} GB free out of {hw.total_ram_gb:.1f} GB). "
            "Please close unused browser tabs and background applications before running model training or inference."
        )

    return {
        "hardware": asdict(hw),
        "inference_recommendation": {
            "model": recommended_inference,
            "notes": inference_notes,
        },
        "training_recommendation": {
            "preset": recommended_train_preset,
            "notes": train_notes,
        },
        "ram_warning": ram_warning,
    }


def format_text_report(analysis: Dict[str, Any]) -> str:
    hw = analysis["hardware"]
    lines = [
        "=" * 68,
        "  AXEN SYSTEM & HARDWARE REPORT",
        "=" * 68,
        f"  CPU Cores           : {hw['physical_cores']} physical, {hw['logical_cores']} logical",
        f"  System RAM          : {hw['total_ram_gb']:.1f} GB total ({hw['available_ram_gb']:.1f} GB free)",
    ]

    if hw["gpu_available"]:
        lines.extend([
            f"  GPU Device          : {hw['gpu_name']}",
            f"  GPU VRAM            : {hw['vram_gb']:.1f} GB",
            f"  CUDA Version        : {hw['cuda_version']}",
            f"  FP16 Support        : {hw['fp16_supported']}",
            f"  BF16 Support        : {hw['bf16_supported']}",
        ])
    else:
        lines.append("  GPU Device          : None (Running on CPU)")

    lines.extend([
        "-" * 68,
        "  LOCAL INFERENCE RECOMMENDATION (Ollama / GGUF):",
        f"    Model : {analysis['inference_recommendation']['model']}",
        f"    Notes : {analysis['inference_recommendation']['notes']}",
        "-" * 68,
        "  FROM-SCRATCH TRAINING RECOMMENDATION (PyTorch):",
        f"    Preset: {analysis['training_recommendation']['preset']}",
        f"    Notes : {analysis['training_recommendation']['notes']}",
    ])

    if analysis["ram_warning"]:
        lines.extend([
            "-" * 68,
            f"  [! WARNING]: {analysis['ram_warning']}",
        ])

    lines.append("=" * 68)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="AXEN Hardware Detection")
    parser.add_argument("--json", action="store_true", help="Print output as JSON")
    args = parser.parse_args()

    hw = detect_hardware()
    analysis = evaluate_capabilities(hw)

    if args.json:
        print(json.dumps(analysis, indent=2))
    else:
        print(format_text_report(analysis))


if __name__ == "__main__":
    main()
