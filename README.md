# AXEN - Personal AI Assistant

Axen's chat runs on its built-in local assistant. It does not connect to external AI providers or require their API keys. The small built-in model is experimental and cannot reliably plan and implement complex websites or generate original images. The local project workspace and browser preview are available as foundations for a future coding workflow.

Generated projects are stored in `data/generated-sites/`. The built-in preview serves project files locally. Axen can store source files in multiple programming languages, but this does not mean it can execute every language or install project dependencies.
---

## Architecture Overview

1. **Local Open-Weights Models**: Fast inference powered by Ollama / llama.cpp (e.g. `Qwen2.5-3B`, `Qwen2.5-1.5B`).
2. **From-Scratch Transformer (PyTorch)**: An educational decoder-only transformer with modern LLM architecture:
   - RoPE (Rotary Position Embeddings)
   - RMSNorm pre-normalization
   - SwiGLU feed-forward networks
   - Tied input/output embeddings
3. **Reasoning Agent Loop**: Step-by-step thinking with `<think> ... </think>` tags and verified tool invocation (Pydantic validation, max 8 steps, bounded retries).
4. **Polyglot Extensions**: Distinct non-Python components located in `polyglot/<language>/`, each performing **one real job** (communication strictly via HTTP, stdin/stdout JSON, gRPC, or files).
5. **Specialized Brain Modules**: Intent classification and code generation (`./code_brain.py` at root is canonical; `axen-brain/code_brain.py` is kept for reference).

---

## Hardware Profile (Auto-Detected)

On this system, running `python scripts/hardware_check.py` detected:

- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (4.0 GB VRAM)
- **CUDA**: 12.4 (FP16 & BF16 supported)
- **CPU**: 6 physical cores, 12 logical threads
- **RAM**: ~8.0 GB total

### Recommendations for This Hardware:
- **Local Inference**: Use **`Qwen2.5-3B`** (or `Qwen2.5-1.5B`). Fits comfortably in 4GB VRAM without memory paging.
- **From-Scratch Training**:
  - **`tiny` (~10M params)**: Optimal for fast iteration (~30,000 tokens/sec, prototype in ~11 minutes).
  - **`small` (~50M params)**: Feasible using gradient accumulation (`micro_batch=4`, `accum=4`, uses ~2.8 GB VRAM).
  - **`medium` (~120M params)**: Requires gradient checkpointing to avoid Out-Of-Memory (OOM) on 4GB VRAM.

> [!WARNING]
> Because system RAM is 8 GB, close memory-heavy apps and browser tabs during model training to avoid system swapping.

---

## Configuration Presets

Presets are defined in `configs/training_presets.yaml`:

| Preset | Parameters | Hidden Dim | Layers | Heads | Context | VRAM (Train) | Prototype Time (GPU) | Quality Expectation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tiny** | **10.06M** | 320 | 6 | 5 | 512 | ~1.2 GB | ~11 min (20M tok) | Basic syntax, TinyStories level, simple patterns |
| **Small** | **49.51M** | 512 | 13 | 8 | 1024 | ~2.8 GB | ~1.3 hrs (50M tok) | Multi-paragraph coherence, simple dialogues |
| **Medium** | **124.20M** | 768 | 14 | 12 | 1024 | ~4.8 GB | ~6.2 hrs (100M tok) | Conversational English, basic arithmetic & logic |

---

## Project Structure

```
├── Makefile                      # Cross-platform make targets
├── README.md                     # Project documentation
├── requirements.txt              # Pinned Python dependencies
├── configs/
│   ├── model.yaml                # Local inference & agent config
│   └── training_presets.yaml     # Transformer architecture presets
├── scripts/
│   └── hardware_check.py         # Hardware detection and recommendations
├── polyglot/
│   └── README.md                 # Polyglot rules and architecture
└── tests/
    └── test_milestone1.py        # Automated test suite for Milestone 1
```

---

## Quickstart Commands (Windows)

### 1. Run Hardware Detection
```powershell
python scripts/hardware_check.py
```
*(Or output JSON)*:
```powershell
python scripts/hardware_check.py --json
```

### 2. Run Test Suite
```powershell
python -m pytest tests/test_milestone1.py -v
```

### 3. Using Make (if installed)
```bash
make check-hardware
make test
```

---

## Milestones Tracker

- [x] **Milestone 1**: Hardware check, environment, project layout, README, Make targets.
- [ ] **Milestone 2**: Run a local open model with Ollama, CLI chat, and the 30-prompt evaluation script.
- [ ] **Milestone 3**: FastAPI streaming backend with SQLite history.
- [ ] **Milestone 4**: Frontend chat UI (HTML, CSS, JavaScript, TypeScript).
- [ ] **Milestone 5**: Tools, agent loop, and approvals.
- [ ] **Milestone 6**: RAG over local documents.
- [ ] **Milestone 7**: LoRA fine-tuning and before/after evaluation.
- [ ] **Milestone 8**: Scripts and containers (Bash, PowerShell, Dockerfile, Make).
- [ ] **Milestone 9–14**: Polyglot modules (low-level, backend, mobile, functional, formats, rare).
- [ ] **Milestone 15**: Final `languages.md` report and one-command start script.

---

## LICENSE NOTE (Phase 11 Image Generation)

AXEN AI includes local text-to-image generation support via pretrained diffusion models:
- **SD-Turbo (`stabilityai/sd-turbo`)**: Released under the Stability AI Non-Commercial Research Community License. It is intended for research and non-commercial personal use unless an appropriate Stability AI commercial membership/license is held.
- **Stable Diffusion 1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5`)**: Released under the CreativeML Open RAIL-M license, which is more permissive and permits commercial usage subject to RAIL-M use-based restrictions.

