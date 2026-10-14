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
import os
import subprocess
import requests
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn

def main():
    # All launch paths use the same configuration, memory and queue files even
    # when the executable was invoked from outside the repository.
    os.chdir(ROOT)
    companion = None
    try:
        ready = requests.get("http://127.0.0.1:8001/api/brain", timeout=2).ok
    except requests.RequestException:
        ready = False
    if not ready:
        companion = subprocess.Popen(
            ["node", "server.js"], cwd=ROOT,
            env={**os.environ, "PORT": "8001", "HOST": "127.0.0.1"},
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    print("AEXN AI: http://127.0.0.1:8000 | local-first automatic routing")
    print("Two-agent mode: chat answers + automatic background evaluation and tested improvement cycles")
    try:
        uvicorn.run("app.ui.app:app", host="127.0.0.1", port=8000)
    finally:
        if companion is not None and companion.poll() is None:
            companion.terminate()
            try:
                companion.wait(timeout=10)
            except subprocess.TimeoutExpired:
                companion.kill()
                companion.wait(timeout=5)

if __name__ == "__main__":
    main()
