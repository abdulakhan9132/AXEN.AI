"""
axen/security/__init__.py
========================
AXEN Cybersecurity Research and Ethical Hacking Subsystem.

Defensive security research, authorized vulnerability analysis,
secure code auditing, prompt injection defense, and security benchmarking.
"""

from __future__ import annotations

from .models import (
    AuthorizedTarget,
    SecurityReport,
    VulnerabilityCategory,
    VulnerabilityFinding,
    VulnerabilitySeverity,
    VulnerabilityStatus,
)
from .target_gate import SecurityTargetGate, TargetAuthorizationError
from .code_analyzer import SecureCodeAnalyzer
from .prompt_defense import PromptInjectionDefense
from .engine import EthicalHackingEngine

__all__ = [
    "AuthorizedTarget",
    "SecurityReport",
    "VulnerabilityCategory",
    "VulnerabilityFinding",
    "VulnerabilitySeverity",
    "VulnerabilityStatus",
    "SecurityTargetGate",
    "TargetAuthorizationError",
    "SecureCodeAnalyzer",
    "PromptInjectionDefense",
    "EthicalHackingEngine",
]
