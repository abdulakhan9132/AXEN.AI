# Project Status

## Environment
- Workspace: `C:\Users\HN089WS\OneDrive\Desktop\job agent`
- Python: 3.13.15
- Virtual environment: `.venv`
- OS: Windows 11 / Windows_NT

## Existing Files
- `backend/` contains a minimal FastAPI health app.
- `frontend/`, `mobile/`, `automation/`, `data/` are present as project folders but no core agent implementation exists yet.
- No existing `ai/`, `agent/`, `tools/`, `tests/`, or project-level `main.py` were found before Phase 1 work.

## Dependencies
- Runtime dependencies already available in the workspace venv: `fastapi`, `uvicorn`, `pydantic`, `requests`.
- No local AI runtime is installed.
- `ollama` CLI is not installed in the environment.

## AI Runtime Status
- Local AI runtime: unavailable
- Fallback architecture: local model adapter with Ollama support when installed
- Model independence: implemented through an abstract `AIModel` interface and model manager

## Architecture
- Phase 1 uses a modular local agent framework with the following major areas:
  - `app/ai/` for model abstraction, prompts, and context
  - `app/agent/` for planning, execution, verification, observation, memory, state, permissions, recovery
  - `app/tools/` for registry and tool implementations
  - `app/ui/` for the local web interface
  - `tests/` for mock-model-driven validation

## Implementation Phases
- Phase 1: local agent core, planning loop, tool execution, verification, memory, CLI, basic UI
- Phase 2: browser automation
- Phase 3: web search
- Phase 4: job discovery
- Phase 5: CV matching
- Phase 6: application preparation
- Phase 7: human approval
- Phase 8: application tracking
- Phase 9: phone / scrcpy integration

## Known Issues
- No local foundation model is present, so the system must gracefully degrade to a local-model-unavailable fallback.
- The project was a blank slate for the core AI layer; it required a fresh Phase 1 scaffold rather than editing a prior implementation.
