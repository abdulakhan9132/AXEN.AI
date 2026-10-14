# AXEN - Personal AI Assistant

## Complete web application

Run `npm start` or, on Windows, `.venv\Scripts\python.exe run_ui.py`. Both start the existing Python application on http://127.0.0.1:8000 and the Node feature companion on port 8001. `npm run start:node` retains the standalone Node service.

AEXN AI uses one chat and one composer for conversations, images and project work. The internal router uses configured local conversational, coding and reasoning providers, and the existing AEXN GPT checkpoint for literary tasks. Configure `CONVERSATIONAL_MODEL`, `CODING_MODEL` and optionally `REASONING_MODEL` for models installed in your Ollama runtime. A configured API key does not grant cloud consent; this interface defaults to local-only requests. An unavailable provider produces an explicit error instead of silently sending your content to a cloud API. The small Shakespeare checkpoint is experimental and is not a general-purpose coding assistant.

Code replies are displayed directly in chat. The previous Codex workspace UI is removed. The new Code Agent currently has milestone 1 implemented: confined file tools, per-action approval tickets, diff previews and SQLite audit logging. The model loop, API and Code mode UI are subsequent milestones. Commands use the existing offline Docker sandbox and fail closed without an existing local `AEXN_SANDBOX_IMAGE`; no new dependencies are installed automatically.

The existing Think Engine chooses reasoning automatically and publishes brief progress messages. Chat uses SSE progress and response frames; model inference currently completes before the response text is emitted, so this is not native token-by-token model streaming. Image requests reuse the existing local ImageBrain pipeline. Attach documents using the existing upload tools.

Generated projects are stored in `data/generated-sites/`. The built-in preview serves project files locally. Axen can store source files in multiple programming languages, but this does not mean it can execute every language or install project dependencies.

## Reverse engineering

Run `npm start`, open `http://127.0.0.1:8000/reverse-engineering`, and inspect `.` (Axen itself) or a source subfolder such as `ai` or `axen/coding`. For another codebase, copy its source into a subfolder of this project. Chat also accepts `/reverse ai`, `/inspect axen/coding`, or `reverse engineer this project`.

The report lists source files, symbols, imports, route patterns, execution/network boundaries, and resolved local JavaScript/TypeScript/Python import connections with line numbers. Download it as Markdown from the inspector. No AI API key or running language model is required. Code is read, never executed by this tool. Pattern matching is approximate; it does not prove runtime behavior or decompile binaries. Scans skip hidden/secret files, dependency directories, weights and user data, and cap coverage at 600 files / 6 MB. Review any partial-coverage message.

## Evaluated small-brain training

`npm run train:reviewed -- 300` trains a separate candidate on an original 36-example curriculum plus replay of the existing starter corpus. Ten held-out examples evaluate next-character prediction. `npm run train:reviewed -- 300 --promote` activates only a candidate that improves this narrow validation score, keeps starter replay loss within tolerance, and passes three basic answer-relevance checks. The keyword checks are a minimal filter, not a comprehensive benchmark. It backs up the previous weights and refuses activation if the active checkpoint changed during training. Run this while Sleep and Learn is idle, then restart Axen to load an activated candidate.

The latest report, original/candidate samples and weight backups live in `ai/data/training/`. The curriculum lives in `ai/data/curriculum/`. This is an experiment on the existing 62,208-parameter model; a loss reduction is not evidence of frontier-level coding or reasoning. Reverse engineering uses source analysis independently of those weights. This workflow does not train, copy or access proprietary model weights, and does not train the installed Qwen model.

Run `npm run test:reverse` to check source extraction, directory boundaries, source exclusion and the HTTP/chat workflows.
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

