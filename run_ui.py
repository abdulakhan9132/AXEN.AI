"""
run_ui.py
=========
Launcher script for the modern AXEN AI Web Interface.

Usage:
    python run_ui.py
    or
    python main.py serve
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn
from app.ui.app import app

def main():
    print("""
======================================================
           AXEN® — Local AI Assistant v3.0
======================================================
  Interface : http://127.0.0.1:8000
  Backend   : PyTorch CUDA + AssistantController
  Status    : Fully Offline · No Cloud APIs
======================================================
    """)
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()
