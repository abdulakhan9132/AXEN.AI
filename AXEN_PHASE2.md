# AXEN Phase 2 — Local Claude-like AI Assistant

## What was built

AXEN Phase 2 adds a complete **local AI assistant layer** on top of the existing AXEN project.  
It runs fully **offline**, with **no cloud APIs, no API keys, no Ollama** (for the new layer).  
All inference uses **PyTorch + HuggingFace Transformers** directly on your RTX 3050.

---

## Architecture

```
axen/
├── __init__.py                  ← Public API surface
├── model/
│   ├── hf_model.py              ← HFLocalModel: direct HF Transformers inference
│   └── __init__.py
├── memory/
│   ├── conversation_memory.py   ← Short-term: current session history (token-trimmed)
│   ├── long_term_memory.py      ← Long-term: persistent JSON fact store
│   └── __init__.py
├── tools/
│   ├── tool_registry.py         ← Typed ToolDefinition + ToolRegistry + ToolResult
│   ├── calculator.py            ← AST-safe arithmetic (no eval, no shell)
│   ├── python_tool.py           ← Sandboxed subprocess Python executor
│   └── __init__.py
└── assistant/
    ├── controller.py            ← AssistantController: ties everything together
    └── __init__.py

axen_chat.py                     ← Interactive CLI entry point
configs/axen_local.yaml          ← Model + Ollama + reasoning configuration
tests/test_axen_phase2.py        ← 66 tests (all passing)
```

---

## Default Model

**TinyLlama/TinyLlama-1.1B-Chat-v1.0**
- 1.1 billion parameters
- Chat fine-tuned
- ~2.2 GB in float16
- Fits comfortably in your RTX 3050 (4 GB VRAM)
- Downloaded once from HuggingFace Hub, then cached locally
- No internet required after first download

To use a different model:
```bash
python axen_chat.py --model "Qwen/Qwen2-1.5B-Instruct"
```

---

## How to run

### 1. Chat with the language model
```bash
python axen_chat.py
```
First run downloads the model (~2 GB). Subsequent runs load from cache in seconds.

### 2. Tool-only mode (no model download)
```bash
python axen_chat.py --no-model
```
Calculator and Python executor still work. Only general conversation is unavailable.

### 3. Check status
```bash
python axen_chat.py --status
```

---

## Chat Commands

| Command | Description |
|---|---|
| `/quit` or `/exit` | Exit |
| `/reset` | Clear conversation history |
| `/status` | Show model + memory status |
| `/remember <fact>` | Save a fact permanently |
| `/recall` | List all saved facts |
| `/tools` | List available tools |
| `/help` | Show command list |

---

## Tools

### Calculator (`calculator.evaluate`)
- **Risk**: Low
- Safe AST-based arithmetic — no `eval()`, no shell
- Operators: `+`, `-`, `*`, `/`, `**`, `%`, `//`
- Example: `calculate 12345 * 678` → `8370110`

### Python Executor (`python.execute`)
- **Risk**: Medium
- Runs code in a subprocess with 5-second timeout
- AST-validated before execution (blocked imports: `subprocess`, `socket`, etc.)
- Returns actual `stdout`, `stderr`, and `returncode`
- Example:
  ```
  run this:
  ```python
  for i in range(5):
      print(i ** 2)
  ```
  ```

---

## Memory

### Short-term (ConversationMemory)
- Holds the current session's message history
- Automatically trims oldest messages when the context window is full
- Reset with `/reset`

### Long-term (LongTermMemory)
- Persistent JSON file at `data/axen_long_term.json`
- Save: `remember that my favorite language is Python`
- Recall: `do you remember anything about me?`
- Forget via `/recall` + code: `long_memory.forget("key")`

---

## Tests

```bash
# Phase 2 tests only
python -m pytest tests/test_axen_phase2.py -v

# All tests
python -m pytest tests/ -v
```

**66 Phase 2 tests — all passing:**
- HFModelConfig (4 tests)
- HFLocalModel structural (4 tests)
- ConversationMemory (7 tests)
- LongTermMemory (9 tests)
- ToolRegistry (5 tests)
- Calculator tool (14 tests)
- Python tool (9 tests)
- AssistantController no-model mode (11 tests)
- DefaultRegistry integration (3 tests)

---

## What is preserved

The following existing AXEN systems are **completely unchanged**:
- `app/` — agent orchestrator, Ollama model manager, memory store
- `reasoning/` — math solver, coding intelligence (37 languages)
- `axen.js` — XOR neural network educational experiment
- All existing tests in `tests/coding/`, `tests/math/`, `tests/test_agent_core.py`

The Ollama backend continues to serve math formalization and coding reasoning.  
The new HF local model handles general conversation independently.

---

## Configuration

`configs/axen_local.yaml` contains all settings:
- `hf_model.*` — HuggingFace model settings (name, device, dtype, tokens, temperature)
- `llm.*` — Ollama settings (preserved)
- `reasoning.*` — Math/coding reasoning settings (preserved)
