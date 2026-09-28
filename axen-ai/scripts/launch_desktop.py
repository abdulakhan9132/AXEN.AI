from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from axen.ui.desktop import launch_desktop


if __name__ == "__main__":
    launch_desktop()
