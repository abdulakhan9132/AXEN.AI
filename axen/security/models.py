"""
axen/security/models.py
======================
Core data models for AXEN Cybersecurity Research & Ethical Hacking Agent.

Covers:
- Vulnerability categories (OWASP Top 10 + AI/Agent Vulnerabilities)
- Finding records, severities, and remediation tracking
- Authorized target boundaries
- Comprehensive security reports
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class VulnerabilitySeverity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFORMATIONAL = "INFORMATIONAL"


class VulnerabilityCategory(str, Enum):
    COMMAND_INJECTION = "COMMAND_INJECTION"
    SQL_INJECTION = "SQL_INJECTION"
    PATH_TRAVERSAL = "PATH_TRAVERSAL"
    PROMPT_INJECTION = "PROMPT_INJECTION"
    UNSAFE_DESERIALIZATION = "UNSAFE_DESERIALIZATION"
    HARDCODED_SECRETS = "HARDCODED_SECRETS"
    INSECURE_CONFIGURATION = "INSECURE_CONFIGURATION"
    UNSAFE_EVAL = "UNSAFE_EVAL"
    WEAK_CRYPTOGRAPHY = "WEAK_CRYPTOGRAPHY"
    SSRF = "SSRF"
    CROSS_SITE_SCRIPTING = "CROSS_SITE_SCRIPTING"
    BROKEN_AUTHENTICATION = "BROKEN_AUTHENTICATION"
    UNCONTROLLED_RESOURCE_CONSUMPTION = "UNCONTROLLED_RESOURCE_CONSUMPTION"


class VulnerabilityStatus(str, Enum):
    DISCOVERED = "DISCOVERED"
    VERIFIED = "VERIFIED"
    FIX_PROPOSED = "FIX_PROPOSED"
    FIXED = "FIXED"
    RE_TESTED = "RE_TESTED"
    MITIGATED = "MITIGATED"
    FALSE_POSITIVE = "FALSE_POSITIVE"


@dataclass
class AuthorizedTarget:
    """
    Defines explicit, authorized boundaries for security testing.
    Testing is strictly prohibited against any target not verified as authorized.
    """
    target_id: str = field(default_factory=lambda: f"tgt_{uuid.uuid4().hex[:8]}")
    name: str = ""
    target_type: str = "local_workspace"  # local_workspace, local_api, test_container, mock_service
    endpoint: str = ""                    # File path, localhost URL, or socket
    authorized_by: str = "system_operator"
    is_active: bool = True
    allowed_actions: List[str] = field(default_factory=lambda: ["static_analysis", "ast_audit", "safe_simulation"])
    created_at: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())

    def is_scope_allowed(self, path_or_url: str) -> bool:
        """Verify that a path or URL falls strictly within this authorized scope."""
        if not self.is_active:
            return False
        clean_target = self.endpoint.replace("\\", "/").rstrip("/")
        clean_query = str(path_or_url).replace("\\", "/").rstrip("/")
        return clean_query.startswith(clean_target)


@dataclass
class VulnerabilityFinding:
    """A verified security vulnerability finding with full audit trail."""
    finding_id: str = field(default_factory=lambda: f"vuln_{uuid.uuid4().hex[:10]}")
    title: str = ""
    category: VulnerabilityCategory = VulnerabilityCategory.INSECURE_CONFIGURATION
    severity: VulnerabilitySeverity = VulnerabilitySeverity.MEDIUM
    status: VulnerabilityStatus = VulnerabilityStatus.DISCOVERED
    target_path: str = ""
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    code_snippet: str = ""
    root_cause: str = ""
    conditions_required: str = ""
    security_impact: str = ""
    detection_method: str = "static_ast_analysis"
    proof_of_concept: str = ""
    remediation_guidance: str = ""
    remediation_patch: str = ""
    discovered_at: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())
    resolved_at: Optional[str] = None
    lessons_learned: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def mark_fixed(self, patch: str = "", lessons: str = "") -> None:
        """Update status to FIXED with patch and lessons learned."""
        self.status = VulnerabilityStatus.FIXED
        self.remediation_patch = patch
        if lessons:
            self.lessons_learned = lessons
        self.resolved_at = dt.datetime.now(dt.timezone.utc).isoformat()

    def mark_retested(self, passed: bool) -> None:
        """Record outcome of verification re-test."""
        if passed:
            self.status = VulnerabilityStatus.RE_TESTED
        else:
            self.status = VulnerabilityStatus.DISCOVERED

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value if isinstance(self.category, VulnerabilityCategory) else str(self.category)
        d["severity"] = self.severity.value if isinstance(self.severity, VulnerabilitySeverity) else str(self.severity)
        d["status"] = self.status.value if isinstance(self.status, VulnerabilityStatus) else str(self.status)
        return d


@dataclass
class SecurityReport:
    """Comprehensive security audit report generated from an authorized assessment."""
    report_id: str = field(default_factory=lambda: f"sec_{uuid.uuid4().hex[:8]}")
    target_name: str = ""
    target_scope: str = ""
    findings: List[VulnerabilityFinding] = field(default_factory=list)
    total_scanned_files: int = 0
    total_findings: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    overall_risk_score: float = 0.0  # 0.0 (clean) to 100.0 (maximum critical exposure)
    summary: str = ""
    created_at: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())

    def compute_statistics(self) -> None:
        """Calculate counts and aggregate risk score."""
        self.total_findings = len(self.findings)
        self.critical_count = sum(1 for f in self.findings if f.severity == VulnerabilitySeverity.CRITICAL)
        self.high_count = sum(1 for f in self.findings if f.severity == VulnerabilitySeverity.HIGH)
        self.medium_count = sum(1 for f in self.findings if f.severity == VulnerabilitySeverity.MEDIUM)
        self.low_count = sum(1 for f in self.findings if f.severity == VulnerabilitySeverity.LOW)

        # Weighted risk calculation
        raw_score = (
            self.critical_count * 25.0
            + self.high_count * 15.0
            + self.medium_count * 5.0
            + self.low_count * 1.0
        )
        self.overall_risk_score = round(min(100.0, raw_score), 2)
