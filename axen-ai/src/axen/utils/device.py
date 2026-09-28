from __future__ import annotations

from typing import Dict

import torch


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_info() -> Dict[str, object]:
    device = detect_device()
    info: Dict[str, object] = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["vram_mb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2))
    else:
        info["gpu_name"] = "CPU"
        info["vram_mb"] = 0
    return info
