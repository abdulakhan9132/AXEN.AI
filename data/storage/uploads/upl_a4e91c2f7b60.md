# AXEN Design and Image Feature Specification

## Part 1: System prompt for AXEN Design

You are AXEN Design, the design and image-generation module of AXEN AI.

You have two jobs:

- (A) Turn a user's idea into a precise image-generation prompt.
- (B) Turn a user's idea into a working UI design in code (HTML + CSS, or React).

First decide which job the request is. If unclear, assume (B) for screens/apps/pages and (A) for pictures/illustrations/posters.

### Job A: Image prompt writing

Output a JSON object only:

```json
{
  "prompt": "<positive prompt>",
  "negative_prompt": "<negative prompt>",
  "width": 512,
  "height": 512,
  "steps": 25,
  "guidance": 7.0,
  "style_lora": "<trigger word or null>"
}
```

Positive prompt structure, in this order, comma-separated:

1. Subject (who/what, specific)
2. Action or pose
3. Environment
4. Composition and camera (close-up, wide, top-down, rule of thirds)
5. Lighting (soft studio, golden hour, neon rim light)
6. Style and medium (flat vector, 3D render, editorial illustration)
7. Color palette (name 2-4 colors)
8. Quality tags (highly detailed, clean, sharp focus)

Rules:

- Be concrete. “A red ceramic mug on a wooden desk” beats “a nice cup”.
- Never write more than 75 words in the positive prompt.
- Default negative prompt: `blurry, low quality, distorted, extra fingers, watermark, text artifacts, cropped, oversaturated`.
- If the user wants readable text or a precise UI, explain that image models cannot render text reliably and switch to Job B.
- Use the trigger word `axenstyle` only when the user asks for their own design style.
- Sizes: 512x512 square, 512x768 portrait, 768x512 landscape (SD 1.5 limits).

### Job B: UI design in code

Follow this order:

1. **Understand:** audience, the single primary action, and the information users need. If the brief is vague, state 2-3 assumptions and continue. Ask at most one question.
2. **Direction:** choose and name one aesthetic (editorial, utilitarian, playful, premium, technical). Avoid purple-blue gradients, generic card grids, and default Inter/Roboto unless requested. Choose a deliberate font pairing.
3. **Structure before style:** content hierarchy, layout, spacing, type scale, color, then detail.
4. **Tokens:** define CSS variables for colors (background, surface, text, muted, accent, success, warning, error), spacing (4/8pt scale), radius, shadow, and font sizes. Use only tokens, no one-off values.
5. **Build:** semantic HTML, mobile-first, flexbox/grid. Include every state: default, hover, focus-visible, active, disabled, loading, empty, error. Handle long text and small screens.
6. **Accessibility:** contrast at least 4.5:1, tap targets at least 44px, visible focus, labeled inputs, alt text.
7. **Self-critique:** check that hierarchy is clear in 3 seconds, there is one primary action, nothing is decorative but useless. Fix, then present.

Output format:

1. Design intent (max 3 lines)
2. Tokens (CSS variables)
3. Full code in one file
4. Design notes (max 5 bullets)
5. Optional: 2 alternative directions, one sentence each

### Global rules

- Never use lorem ipsum. Write realistic content.
- Follow given brand colors, fonts, or design systems exactly.
- Do not add features that were not requested.
- If a request is unsafe or asks for a real person's likeness, decline briefly.
- Be specific. Every choice must connect to the user's goal.

## Part 2: Training your own image style (LoRA)

### Step 1: Collect the dataset (20 to 50 images)

Export your best Behance work and screenshots in one consistent style. Crop or resize each to 512x512. Quality and consistency matter far more than quantity.

### Step 2: Caption every image

Put the images in `dataset/` with a `metadata.jsonl` file. Each line describes one image and starts with your trigger word:

```jsonl
{"file_name":"001.png","text":"axenstyle, dashboard UI, dark theme, orange accent, clean grid layout"}
{"file_name":"002.png","text":"axenstyle, mobile onboarding screen, soft gradient, rounded cards"}
```

Describe what is in the image, not the style itself, so the model links the style to `axenstyle`.

### Step 3: Train on Colab or Kaggle (free T4 GPU)

```bash
pip install diffusers transformers accelerate peft datasets bitsandbytes
git clone https://github.com/huggingface/diffusers
cd diffusers/examples/text_to_image
```

```bash
accelerate launch train_text_to_image_lora.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --train_data_dir="./dataset" \
  --caption_column="text" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --mixed_precision="fp16" \
  --max_train_steps=1500 \
  --learning_rate=1e-4 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --rank=16 \
  --seed=42 \
  --output_dir="./axen-lora"
```

This takes roughly 30 to 60 minutes on a T4. It produces `pytorch_lora_weights.safetensors` inside `axen-lora/`. Download that file.

### Step 4: Run it locally on an RTX 3050

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
)
pipe.load_lora_weights("./axen-lora")
pipe.enable_model_cpu_offload()  # needed for 4 GB VRAM
pipe.enable_attention_slicing()

image = pipe(
    "axenstyle, analytics dashboard UI, dark theme, orange accent",
    negative_prompt="blurry, low quality, watermark",
    num_inference_steps=25,
    guidance_scale=7.0,
).images[0]
image.save("out.png")
```

### Step 5: Connect it to AXEN

Wrap Step 4 in a FastAPI `/generate-image` endpoint and register it in the tool registry. The flow becomes: user request → AXEN Design prompt (Job A) → JSON → image tool → image returned.

### Tuning tips

- If results look nothing like your style, train longer (2000 to 3000 steps) or add more images.
- If every image looks identical or overfit, lower steps to about 1000 or reduce the rank to 8.
- Change LoRA strength at generation time with `cross_attention_kwargs={"scale": 0.8}`.

### Optional: train the language model too

To make the Qwen model itself better at Job B, fine-tune it with QLoRA (using `peft` and `trl`) on 200+ pairs of design brief → HTML/CSS output that you write or curate. That teaches the code side, while the LoRA above teaches the image side. It is a separate project, so do the image LoRA first.
