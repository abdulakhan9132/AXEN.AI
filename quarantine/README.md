# AEXN AI — Quarantined Files

This directory contains components separated during **Phase 0 — Safe Repository Cleanup**.
None of these files are required by the active AEXN AI runtime, tests, or Think Engine.

## Quarantined Items

1. **`backend/`**:
   - `backend/main.py`: Earlier standalone prototype ("Private Job Agent") for an unrelated role (UI/UX Designer in Bengaluru).
   - `backend/image_routes.py` & `backend/image_service.py`: Deprecated Stable Diffusion prototype superseded by `image_brain.py` and `app/ui/app.py`.
2. **`calculator.py`**:
   - 3-line dummy file generated during an automated demo task run from `main.py`.
3. **`test_agent/`**:
   - Ad-hoc test folder created during an automated demo task run from `main.py`.

## Restoration Instructions
To restore any quarantined item to its original location:
```powershell
# Restore backend
Move-Item quarantine/backend backend

# Restore calculator.py
Move-Item quarantine/calculator.py calculator.py

# Restore test_agent
Move-Item quarantine/test_agent test_agent
```
Or to restore the entire repository to the exact checkpoint:
```powershell
git checkout checkpoint-think-engine-2.0
```
