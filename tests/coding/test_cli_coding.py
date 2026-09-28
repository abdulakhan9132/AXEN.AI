"""Tests for AXEN Coding CLI commands."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parent.parent.parent


def test_cli_code_languages():
    proc = subprocess.run(
        [sys.executable, "main.py", "code-languages"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert '"total": 37' in proc.stdout
    assert "Python" in proc.stdout
    assert "Rust" in proc.stdout
    assert "CUDA" in proc.stdout


def test_cli_code_detect():
    proc = subprocess.run(
        [sys.executable, "main.py", "code-detect", "file.rs"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert '"detected": "Rust"' in proc.stdout


def test_cli_code_review():
    proc = subprocess.run(
        [sys.executable, "main.py", "code-review", "def foo(): pass"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert '"metrics"' in proc.stdout
