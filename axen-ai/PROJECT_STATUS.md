# AXEN AI Project Status

## Environment
- Workspace: `C:\Users\HN089WS\OneDrive\Desktop\job agent`
- Project root: `axen-ai/`
- Python: 3.13.15
- Virtual environment: `.venv` at the parent workspace root

## Hardware
- GPU: Not detected in the current environment
- CUDA: unavailable
- Device target: CPU-first, with CUDA support prepared when available

## Software
- Python 3.13.15
- PyTorch, NumPy, PyYAML, tqdm, and pytest are installed in the workspace venv
- The project is set up for a small educational decoder-only Transformer that runs on CPU and auto-detects CUDA.

## Architecture
- Package layout: `src/axen/...`
- Core modules:
  - `config/` for YAML-driven configuration
  - `data/` for cleaning, splitting, and dataset creation
  - `tokenizer/` for a simple character-level tokenizer
  - `model/` for embeddings, attention, feed-forward, and transformer blocks
  - `training/` for optimizer and training loop
  - `inference/` for generation
  - `evaluation/` for validation and metrics
  - `utils/` for device and reproducibility helpers
  - `cli/` for commands

## Dependencies
- PyTorch
- NumPy
- PyYAML
- tqdm
- pytest

## Current milestone
- MILESTONE 4: Desktop UI and local chat integration

## Next milestone
- Expand data and training experiments only after preserving the tiny overfit regression test

## Desktop UI
- Status: COMPLETED
- Model integration: READY when `checkpoints/overfit.pt` exists; otherwise MODEL NOT TRAINED
- Storage: local SQLite at `data/axen_chats.sqlite3`
- Launch: `python main.py desktop` or `python scripts/launch_desktop.py`
- Tests: 18 passed
- Inference fix: generation now enforces token ranges and uses a rolling context window
- Root cause fixed: generation exceeded the 32-position embedding table after appending tokens

## Local data pipeline
- Status: PREPARATION IMPLEMENTED; full training intentionally not started
- Supported inputs: TXT, Markdown, JSON, JSONL, CSV
- Commands: `data-preview`, `prepare-data`, `data-stats`
- Tokenizer: existing AXEN tokenizer, fitted only on training records
- Manifest: `data/processed/manifest.json` with metadata and source hashes only
- Environment: Python 3.13.15, PyTorch 2.14.0+cpu, CUDA unavailable
- Tests: 20 passed

## Known limitations
- This environment currently has no GPU, so CUDA-specific testing will be conditional.
- The first tokenizer is intentionally simple (character-level) and will be replaced by a more advanced tokenizer in later phases.
- This is a small educational model, not a production-scale language model.
