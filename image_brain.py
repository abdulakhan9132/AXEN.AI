"""
image_brain.py
==============
Local text-to-image generation brain for AXEN AI using pretrained diffusion models
(SD-Turbo and SD 1.5) optimized for 4 GB VRAM GPUs.

Features:
- 4 GB VRAM friendly: attention slicing + model CPU offload + Ollama VRAM purging.
- Lazy GPU loading: importing image_brain does not initialize PyTorch or CUDA.
- Graceful prompt enhancement with local Ollama LLM, falling back cleanly if offline.
- Automatic OOM recovery: retries once at 384x384 resolution on CUDA OOM.
- Thread-safe generation lock.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("axen.image_brain")

PRESETS: Dict[str, Dict[str, Any]] = {
    "turbo": {"id": "stabilityai/sd-turbo", "steps": 4, "guidance": 0.0, "size": 512},
    "sd15":  {"id": "stable-diffusion-v1-5/stable-diffusion-v1-5", "steps": 25, "guidance": 7.5, "size": 512},
}

OLLAMA_DEFAULT_URL = "http://localhost:11434"
NEGATIVE = "blurry, low quality, distorted, deformed, extra fingers, watermark, text, cropped"

# Global concurrency lock to guarantee only one image generation runs on GPU at a time
_IMAGE_LOCK = threading.Lock()


def get_configured_ollama_url() -> str:
    """Read Ollama URL from environment or project config."""
    return os.getenv("OLLAMA_BASE_URL", OLLAMA_DEFAULT_URL).rstrip("/")


def get_default_llm_model() -> str:
    """Read Ollama LLM model from env or config file."""
    env_model = os.getenv("LOCAL_MODEL_NAME")
    if env_model:
        return env_model
    try:
        config_path = Path("configs/axen_local.yaml")
        if config_path.exists():
            import yaml
            data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            model = data.get("llm", {}).get("model")
            if model:
                return str(model)
    except Exception:
        pass
    return "qwen2.5:3b"


def free_ollama_vram(models: Optional[Sequence[str]] = None) -> None:
    """Unload Ollama models so the GPU has maximum free VRAM for diffusion."""
    targets = list(models) if models else ["qwen2.5-coder:3b", "qwen2.5:3b", "qwen2.5:1.5b"]
    url = get_configured_ollama_url()
    for m in targets:
        try:
            requests.post(
                f"{url}/api/generate",
                json={"model": m, "keep_alive": 0},
                timeout=5,
            )
        except Exception:
            pass


class ImageBrain:
    """AXEN Local Text-to-Image Generation Engine."""

    def __init__(
        self,
        preset: str = "turbo",
        out_dir: str | Path = "outputs/images",
        llm: Optional[str] = None,
    ) -> None:
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(PRESETS.keys())}")
        self.preset_name = preset
        self.cfg = dict(PRESETS[preset])
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.llm = llm or get_default_llm_model()
        self.pipe: Any = None

    def _load(self) -> None:
        """Lazy load the diffusion pipeline onto GPU/CPU on first generation request."""
        if self.pipe is not None:
            return

        free_ollama_vram([self.llm, "qwen2.5-coder:3b", "qwen2.5:1.5b"])

        # Lazy imports so importing image_brain never touches GPU
        import torch
        from diffusers import AutoPipelineForText2Image

        cuda = torch.cuda.is_available()
        dtype = torch.float16 if cuda else torch.float32

        try:
            try:
                pipe = AutoPipelineForText2Image.from_pretrained(
                    self.cfg["id"],
                    torch_dtype=dtype,
                    variant="fp16" if cuda else None,
                )
            except Exception:
                pipe = AutoPipelineForText2Image.from_pretrained(
                    self.cfg["id"],
                    torch_dtype=dtype,
                )
        except Exception as e:
            msg = f"Failed to load image model '{self.cfg['id']}': {e}"
            logger.error(msg)
            raise RuntimeError(f"Image model download/load error: {msg}") from e

        try:
            pipe.enable_attention_slicing()
        except Exception as e:
            logger.debug("Attention slicing not available: %s", e)

        if cuda:
            try:
                pipe.enable_model_cpu_offload()
            except Exception as e:
                logger.warning("Could not enable model CPU offload: %s", e)
                pipe = pipe.to("cuda")

        self.pipe = pipe

    def unload(self) -> None:
        """Free pipeline from memory and empty CUDA cache."""
        import gc
        self.pipe = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def enhance_prompt(self, idea: str) -> str:
        """Use the local LLM to expand a concise idea into a detailed visual prompt."""
        url = get_configured_ollama_url()
        try:
            r = requests.post(
                f"{url}/api/chat",
                json={
                    "model": self.llm,
                    "stream": False,
                    "keep_alive": 0,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Rewrite the user's idea as ONE detailed image-generation prompt (max 60 words): "
                                "subject, setting, lighting, style, composition. Output only the prompt."
                            ),
                        },
                        {"role": "user", "content": idea},
                    ],
                },
                timeout=30,
            )
            if r.status_code == 200:
                out = r.json().get("message", {}).get("content", "").strip().strip('"')
                if out:
                    return out
        except Exception as e:
            logger.debug("Prompt enhancement skipped (Ollama offline/busy): %s", e)
        return idea

    def _generate_pass(
        self,
        prompt: str,
        size: int,
        seed: Optional[int],
        n: int,
        idea: str,
    ) -> List[str]:
        """Performs raw diffusion inference for n images at the given size."""
        import torch

        paths: List[str] = []
        cuda = torch.cuda.is_available()

        for i in range(n):
            gen = torch.Generator("cuda" if cuda else "cpu")
            s = seed if seed is not None else int(torch.randint(0, 2**31 - 1, (1,)))
            gen.manual_seed(s + i)

            kwargs: Dict[str, Any] = {
                "prompt": prompt,
                "num_inference_steps": self.cfg["steps"],
                "guidance_scale": self.cfg["guidance"],
                "width": size,
                "height": size,
                "generator": gen,
            }
            if self.cfg["guidance"] > 1.0:
                kwargs["negative_prompt"] = NEGATIVE

            res = self.pipe(**kwargs)
            img = res.images[0]

            clean_name = re.sub(r"[^a-z0-9]+", "_", idea.lower()).strip("_")[:40] or "image"
            name = f"{clean_name}_{int(time.time())}_{i}.png"
            out_file = self.out / name
            img.save(out_file, format="PNG")
            paths.append(str(out_file))

        return paths

    def generate(
        self,
        idea: str,
        enhance: bool = True,
        seed: Optional[int] = None,
        n: int = 1,
    ) -> Dict[str, Any]:
        """
        Generate image(s) from text description.
        Returns:
            {"prompt": str, "paths": List[str]}
        """
        with _IMAGE_LOCK:
            self._load()
            prompt = self.enhance_prompt(idea) if enhance else idea
            target_size = self.cfg["size"]

            try:
                paths = self._generate_pass(prompt, size=target_size, seed=seed, n=n, idea=idea)
                return {"prompt": prompt, "paths": paths}
            except Exception as e:
                err_str = str(e).lower()
                is_oom = "out of memory" in err_str or "cuda error: out of memory" in err_str

                if is_oom and target_size > 384:
                    logger.warning("CUDA OOM at %dx%d. Unloading and retrying once at 384x384...", target_size, target_size)
                    self.unload()
                    self._load()
                    try:
                        paths = self._generate_pass(prompt, size=384, seed=seed, n=n, idea=idea)
                        return {"prompt": prompt, "paths": paths}
                    except Exception as retry_err:
                        self.unload()
                        raise RuntimeError(f"CUDA Out of Memory during image generation retry: {retry_err}") from retry_err
                elif is_oom:
                    self.unload()
                    raise RuntimeError("CUDA Out of Memory during image generation. Free VRAM and try again.") from e
                else:
                    raise


if __name__ == "__main__":
    brain = ImageBrain("turbo")
    res = brain.generate("a futuristic dashboard UI mockup, dark theme, glowing blue accents")
    print("Generated Prompt:", res["prompt"])
    print("Paths:", res["paths"])
