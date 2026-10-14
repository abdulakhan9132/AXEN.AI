"""
axen/security/prompt_defense.py
==============================
Defensive hardening and prompt injection detection for AXEN AI Agent.

Protects against:
- Direct system instruction overrides ("Ignore previous instructions...")
- Roleplay & jailbreak payloads ("You are now DAN / Developer Mode...")
- Hidden payload injections (zero-width characters, invisible unicode, HTML comment smuggling)
- Indirect prompt injection embedded within untrusted external text or web search results
"""

from __future__ import annotations

import base64
import logging
import re
from typing import Dict, List, Optional, Tuple

from .models import (
    VulnerabilityCategory,
    VulnerabilityFinding,
    VulnerabilitySeverity,
)

logger = logging.getLogger("axen.security.prompt_defense")


class PromptInjectionDefense:
    """Detects, classifies, and neutralizes prompt injection attacks."""

    # Core injection patterns
    _DIRECT_OVERRIDE_PATTERNS = [
        re.compile(r"(?i)\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|rules|prompts|guidelines)\b"),
        re.compile(r"(?i)\bdisregard\s+(?:all\s+)?(?:previous|prior|above)\b"),
        re.compile(r"(?i)\breset\s+all\s+(?:instructions|system\s+prompts)\b"),
        re.compile(r"(?i)\byou\s+are\s+now\s+(?:DAN|unfiltered|jailbroken|evil|unrestricted)\b"),
        re.compile(r"(?i)\bdeveloper\s+mode\s+(?:enabled|activated|on)\b"),
        re.compile(r"(?i)\breveal\s+(?:your|the)\s+(?:system\s+prompt|hidden\s+instructions|secret\s+key)\b"),
        re.compile(r"(?i)\bprint\s+(?:your|the)\s+system\s+instructions\b"),
    ]

    # Delimiter and evasion patterns
    _DELIMITER_SMUGGLING = re.compile(r"(?i)(?:<!--.*?-->|--- BEGIN UNTRUSTED|\[SYSTEM_OVERRIDE\]|<\|im_start\|>|<\|im_end\|>)")
    _ZERO_WIDTH_CHARS = re.compile(r"[\u200B\u200C\u200D\uFEFF\u202A-\u202E]")

    def analyze_text(self, text: str, source_context: str = "user_input") -> Tuple[bool, List[VulnerabilityFinding]]:
        """
        Scan text for prompt injection signatures and evasion techniques.

        Returns:
            (is_malicious, findings)
        """
        findings: List[VulnerabilityFinding] = []
        if not text:
            return False, findings

        # 1. Zero-width unicode smuggling check
        zw_matches = self._ZERO_WIDTH_CHARS.findall(text)
        if len(zw_matches) >= 3:
            findings.append(
                VulnerabilityFinding(
                    title="Obfuscated Zero-Width Character Injection",
                    category=VulnerabilityCategory.PROMPT_INJECTION,
                    severity=VulnerabilitySeverity.HIGH,
                    target_path=source_context,
                    root_cause="Zero-width unicode characters embedded to evade keyword and semantic filters.",
                    security_impact="Bypasses basic safety filters while being parsed by tokenizer.",
                    detection_method="unicode_sanitizer",
                    remediation_guidance="Strip non-printable zero-width unicode codepoints prior to model ingestion.",
                )
            )

        # Clean zero-width chars for lexical inspection
        sanitized = self._ZERO_WIDTH_CHARS.sub("", text)

        # 2. Direct override inspection
        for pat in self._DIRECT_OVERRIDE_PATTERNS:
            match = pat.search(sanitized)
            if match:
                findings.append(
                    VulnerabilityFinding(
                        title=f"Direct Prompt Injection Attempt ('{match.group(0)}')",
                        category=VulnerabilityCategory.PROMPT_INJECTION,
                        severity=VulnerabilitySeverity.CRITICAL,
                        target_path=source_context,
                        code_snippet=match.group(0),
                        root_cause="Input contains explicit directive to hijack agent instruction hierarchy.",
                        security_impact="Agent policy bypass, prompt leakage, or unauthorized action execution.",
                        detection_method="regex_signature_engine",
                        remediation_guidance="Quarantine input as inert document text; deny execution directives.",
                    )
                )

        # 3. Delimiter smuggling
        delims = self._DELIMITER_SMUGGLING.findall(sanitized)
        if delims:
            findings.append(
                VulnerabilityFinding(
                    title="Prompt Delimiter Smuggling Detected",
                    category=VulnerabilityCategory.PROMPT_INJECTION,
                    severity=VulnerabilitySeverity.MEDIUM,
                    target_path=source_context,
                    code_snippet=", ".join(delims[:3]),
                    root_cause="Input attempts to mimic system delimiters or markdown comments.",
                    security_impact="May cause LLM to misidentify user content as trusted system commands.",
                    detection_method="delimiter_pattern_matcher",
                    remediation_guidance="Escape brackets and XML tags before prompt assembly.",
                )
            )

        is_malicious = any(f.severity in (VulnerabilitySeverity.HIGH, VulnerabilitySeverity.CRITICAL) for f in findings)
        return is_malicious, findings

    def sanitize_untrusted_input(self, raw_text: str, label: str = "document") -> str:
        """
        Quarantine and neutralize untrusted input, stripping evasions and wrapping in safe boundary.
        """
        if not raw_text:
            return ""

        # Remove invisible zero-width evasion characters
        cleaned = self._ZERO_WIDTH_CHARS.sub("", raw_text)

        # Defuse delimiter collision
        defused = (
            cleaned
            .replace("<|im_start|>", "[escaped_token]")
            .replace("<|im_end|>", "[escaped_token]")
            .replace("```", "'''")
        )

        return (
            f"--- BEGIN INERT DATA [{label}] ---\n"
            f"{defused}\n"
            f"--- END INERT DATA [{label}] ---"
        )
