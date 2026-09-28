"""AXEN Code Security & Safety Checker.

Treats all generated and untrusted code as potentially dangerous.
Inspects code for destructive operations, credential theft, system tampering,
and malicious network access before compilation or execution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List


class SecurityRisk(str, Enum):
    SAFE = "safe"
    WARNING = "warning"
    DANGEROUS = "dangerous"


@dataclass
class SecurityFinding:
    severity: SecurityRisk
    category: str
    description: str
    pattern_matched: str


@dataclass
class SecurityReport:
    risk: SecurityRisk
    is_safe_to_execute: bool
    findings: list[SecurityFinding]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk": self.risk.value,
            "is_safe_to_execute": self.is_safe_to_execute,
            "findings_count": len(self.findings),
            "findings": [
                {
                    "severity": f.severity.value,
                    "category": f.category,
                    "description": f.description,
                    "pattern": f.pattern_matched,
                }
                for f in self.findings
            ],
        }


# Dangerous patterns across multiple programming languages and shell commands
_DANGEROUS_RULES = [
    # 1. Arbitrary / Recursive file deletion
    (re.compile(r"\brm\s+-(?:r[fF]|f[rR])\s+(?:/|[a-zA-Z]:|\*|\~|\$HOME)", re.I), SecurityRisk.DANGEROUS, "file_destruction", "Recursive root/home directory deletion command"),
    (re.compile(r"\bdel\s+(?:/[sS]\s+/[qQ]|/[qQ]\s+/[sS])\s+[cC]:", re.I), SecurityRisk.DANGEROUS, "file_destruction", "Windows recursive drive deletion command"),
    (re.compile(r"\bshutil\.rmtree\s*\(\s*['\"](?:/|[a-zA-Z]:[\\/]|~)", re.I), SecurityRisk.DANGEROUS, "file_destruction", "Python root directory recursive deletion"),
    (re.compile(r"\bos\.remove\s*\(\s*['\"](?:/|[a-zA-Z]:[\\/])", re.I), SecurityRisk.DANGEROUS, "file_destruction", "Deletion of root drive or operating system files"),
    (re.compile(r"\bformat\s+[a-zA-Z]:", re.I), SecurityRisk.DANGEROUS, "file_destruction", "Disk format command"),
    (re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", re.I), SecurityRisk.DANGEROUS, "denial_of_service", "Bash fork bomb"),

    # 2. System and security tampering
    (re.compile(r"(?:\bshutdown\s+[-/][a-zA-Z0-9]|/sbin/shutdown|/usr/sbin/shutdown|\breboot\s+-[a-zA-Z]|\binit\s+0\b|\bsystemctl\s+(?:poweroff|reboot))", re.I), SecurityRisk.DANGEROUS, "system_tampering", "Machine shutdown or reboot instruction"),
    (re.compile(r"\bSet-MpPreference\s+.*?-DisableRealtimeMonitoring\b", re.I), SecurityRisk.DANGEROUS, "security_tampering", "Attempt to disable Windows Defender antivirus"),
    (re.compile(r"\bsc\s+delete\b", re.I), SecurityRisk.DANGEROUS, "system_tampering", "Windows service deletion"),
    (re.compile(r"\bnet\s+user\s+\w+\s+/add\b", re.I), SecurityRisk.DANGEROUS, "privilege_escalation", "Attempt to create system user accounts"),

    # 3. Credential theft & private keys
    (re.compile(r"\.ssh[\\/](?:id_rsa|id_ed25519|authorized_keys)", re.I), SecurityRisk.DANGEROUS, "credential_access", "Attempted access to SSH private keys"),
    (re.compile(r"['\"]?/(?:etc/shadow|etc/passwd)['\"]?", re.I), SecurityRisk.DANGEROUS, "credential_access", "Attempted access to Linux password hashes"),
    (re.compile(r"\bmimikatz\b", re.I), SecurityRisk.DANGEROUS, "credential_theft", "Known credential dumping tool reference"),

    # 4. Malicious network & reverse shells
    (re.compile(r"/dev/tcp/\d+\.\d+\.\d+\.\d+/\d+", re.I), SecurityRisk.DANGEROUS, "reverse_shell", "Linux bash socket reverse shell"),
    (re.compile(r"\bnc\s+-(?:e|c)\s+(?:/bin/sh|/bin/bash|cmd\.exe)", re.I), SecurityRisk.DANGEROUS, "reverse_shell", "Netcat remote shell execution"),
    (re.compile(r"\bcurl\s+.*?\|\s*(?:bash|sh|powershell)", re.I), SecurityRisk.DANGEROUS, "remote_execution", "Pipe untrusted remote web script directly into shell"),
    (re.compile(r"\bwget\s+.*?\|\s*(?:bash|sh|powershell)", re.I), SecurityRisk.DANGEROUS, "remote_execution", "Pipe untrusted remote web script directly into shell"),

    # 5. Suspicious / Warning indicators
    (re.compile(r"\beval\s*\(", re.I), SecurityRisk.WARNING, "code_injection", "Dynamic eval() usage is prone to code injection"),
    (re.compile(r"\bsubprocess\.call\s*\(.*?shell\s*=\s*True", re.I), SecurityRisk.WARNING, "shell_injection", "Python shell=True with user inputs can allow shell injection"),
    (re.compile(r"\bexec\s*\(", re.I), SecurityRisk.WARNING, "code_injection", "Dynamic exec() usage executes arbitrary strings"),
]


class SecurityChecker:
    """Evaluates code safety against security rules."""

    def inspect(self, code_text: str) -> SecurityReport:
        if not code_text or not code_text.strip():
            return SecurityReport(risk=SecurityRisk.SAFE, is_safe_to_execute=True, findings=[])

        findings: list[SecurityFinding] = []
        max_risk = SecurityRisk.SAFE

        for pattern, severity, category, description in _DANGEROUS_RULES:
            match = pattern.search(code_text)
            if match:
                findings.append(
                    SecurityFinding(
                        severity=severity,
                        category=category,
                        description=description,
                        pattern_matched=match.group(0),
                    )
                )
                if severity == SecurityRisk.DANGEROUS:
                    max_risk = SecurityRisk.DANGEROUS
                elif severity == SecurityRisk.WARNING and max_risk != SecurityRisk.DANGEROUS:
                    max_risk = SecurityRisk.WARNING

        safe_to_execute = (max_risk != SecurityRisk.DANGEROUS)
        return SecurityReport(risk=max_risk, is_safe_to_execute=safe_to_execute, findings=findings)


_DEFAULT_CHECKER = SecurityChecker()


def check_security(code: str) -> SecurityReport:
    return _DEFAULT_CHECKER.inspect(code)
