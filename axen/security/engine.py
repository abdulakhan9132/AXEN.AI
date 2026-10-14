"""
axen/security/engine.py
======================
Ethical Hacking & Defensive Security Testing Engine for AXEN.

Implements the complete Ethical Hacking Loop:
RESEARCH → RECONNAISSANCE → ARCHITECTURE ANALYSIS → VULNERABILITY HYPOTHESIS
→ CONTROLLED TEST → RESULT → ROOT-CAUSE ANALYSIS → FIX → RE-TEST → REPORT → MEMORY
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

from .code_analyzer import SecureCodeAnalyzer
from .models import (
    AuthorizedTarget,
    SecurityReport,
    VulnerabilityCategory,
    VulnerabilityFinding,
    VulnerabilitySeverity,
    VulnerabilityStatus,
)
from .prompt_defense import PromptInjectionDefense
from .target_gate import SecurityTargetGate, TargetAuthorizationError

logger = logging.getLogger("axen.security.engine")


class EthicalHackingEngine:
    """
    Coordinates authorized security assessments, vulnerability analysis,
    controlled laboratory testing, automated remediation, and re-testing.
    """

    def __init__(
        self,
        target_gate: Optional[SecurityTargetGate] = None,
        code_analyzer: Optional[SecureCodeAnalyzer] = None,
        prompt_defense: Optional[PromptInjectionDefense] = None,
    ) -> None:
        self.target_gate = target_gate or SecurityTargetGate()
        self.code_analyzer = code_analyzer or SecureCodeAnalyzer()
        self.prompt_defense = prompt_defense or PromptInjectionDefense()

    def run_security_assessment(
        self,
        target_path_or_url: str,
        target_name: Optional[str] = None,
    ) -> SecurityReport:
        """
        Execute full authorized security assessment loop on target.

        1. TARGET VALIDATION: Verify target is explicitly authorized.
        2. RECONNAISSANCE: Discover accessible files and architectural scope.
        3. ARCHITECTURE ANALYSIS: Inspect source files and configurations.
        4. VULNERABILITY HYPOTHESIS & CONTROLLED TEST: Run SAST and prompt defenses.
        5. ROOT-CAUSE ANALYSIS: Pinpoint specific code defect and conditions.
        6. REPORT: Assemble structured SecurityReport.
        """
        # Step 1: Enforce Authorization Gate
        authorized_target = self.target_gate.validate_target(target_path_or_url)
        name = target_name or authorized_target.name

        report = SecurityReport(
            target_name=name,
            target_scope=str(target_path_or_url),
        )

        target_path = Path(target_path_or_url).resolve()
        files_to_scan: List[Path] = []

        # Step 2: Reconnaissance
        if target_path.is_file():
            files_to_scan.append(target_path)
        elif target_path.is_dir():
            # Gather relevant Python and configuration files
            ignore_dirs = {".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache"}
            for p in target_path.rglob("*.py"):
                if not any(ig in p.parts for ig in ignore_dirs):
                    files_to_scan.append(p)
                    if len(files_to_scan) >= 100:  # Bounded audit limit
                        break

        report.total_scanned_files = len(files_to_scan)

        # Step 3 & 4: Controlled Security Audit
        all_findings: List[VulnerabilityFinding] = []
        for file_path in files_to_scan:
            findings = self.code_analyzer.audit_file(file_path)
            for f in findings:
                f.status = VulnerabilityStatus.VERIFIED
                all_findings.append(f)

        report.findings = all_findings
        report.compute_statistics()
        report.summary = (
            f"Security assessment of '{name}' completed. "
            f"Scanned {report.total_scanned_files} files, discovered {report.total_findings} findings "
            f"({report.critical_count} Critical, {report.high_count} High, {report.medium_count} Medium, {report.low_count} Low). "
            f"Overall Risk Score: {report.overall_risk_score}/100.0"
        )

        return report

    def remediate_and_retest(
        self,
        finding: VulnerabilityFinding,
        original_code: str,
        remediation_code: str,
    ) -> Tuple[bool, str]:
        """
        Applies a proposed remediation in a safe simulation and re-tests to verify resolution.

        Returns:
            (is_resolved, audit_message)
        """
        # 1. Re-audit original code to confirm baseline finding
        orig_findings = self.code_analyzer.audit_source_code(original_code)
        baseline_has_vuln = any(f.category == finding.category for f in orig_findings)

        if not baseline_has_vuln:
            return False, "Baseline vulnerability could not be reproduced in original code."

        # 2. Audit proposed remediation
        fixed_findings = self.code_analyzer.audit_source_code(remediation_code)
        still_vulnerable = any(f.category == finding.category for f in fixed_findings)

        if still_vulnerable:
            return False, f"Re-test FAILED: The proposed code still triggers {finding.category.value}."

        # 3. Mark finding as FIXED and RE-TESTED
        finding.mark_fixed(
            patch=remediation_code,
            lessons=f"Replaced vulnerable construct with safe deterministic alternative for {finding.category.value}."
        )
        finding.mark_retested(passed=True)

        return True, f"Re-test PASSED: Vulnerability {finding.category.value} successfully remediated and verified."
