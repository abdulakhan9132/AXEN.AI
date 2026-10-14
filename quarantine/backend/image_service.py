import threading
import uuid
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

# Save images to the Node server's static directory
OUT_DIR = Path(__file__).parent.parent / "app" / "ui" / "static" / "generated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
LORA_PATH = Path("models/axen-lora")   # your trained LoRA folder (optional)

_pipe = None
_lock = threading.Lock()   # one generation at a time on a 4 GB GPU


def _load():
    global _pipe
    if _pipe is None:
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16
        )
        if LORA_PATH.exists():
            pipe.load_lora_weights(str(LORA_PATH))
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        _pipe = pipe
    return _pipe


def generate(prompt, negative_prompt=None, width=512, height=512,
             steps=25, guidance=7.0):
    with _lock:
        pipe = _load()
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or "blurry, low quality, watermark",
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
        ).images[0]
    name = f"{uuid.uuid4().hex}.png"
    image.save(OUT_DIR / name)
    return f"/static/generated/{name}"