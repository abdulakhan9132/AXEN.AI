"""AXEN Code Analyzer.

Performs static review and software engineering analysis on source code:
structure, readability, maintainability, complexity, error handling,
security, and performance considerations.
Avoids arbitrary subjective scores unless explicitly configured.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .registry import LanguageSpec, get_language_registry
from .security import SecurityRisk, check_security


@dataclass
class CodeMetrics:
    total_lines: int
    code_lines: int
    comment_lines: int
    blank_lines: int
    functions_count: int
    classes_count: int
    estimated_max_indent_depth: int


@dataclass
class AnalysisReport:
    language: str
    metrics: CodeMetrics
    readability_observations: list[str]
    maintainability_observations: list[str]
    performance_considerations: list[str]
    error_handling_observations: list[str]
    security_observations: list[str]
    improvement_recommendations: list[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "metrics": {
                "total_lines": self.metrics.total_lines,
                "code_lines": self.metrics.code_lines,
                "comment_lines": self.metrics.comment_lines,
                "blank_lines": self.metrics.blank_lines,
                "functions_count": self.metrics.functions_count,
                "classes_count": self.metrics.classes_count,
                "estimated_max_indent_depth": self.metrics.estimated_max_indent_depth,
            },
            "readability": self.readability_observations,
            "maintainability": self.maintainability_observations,
            "performance": self.performance_considerations,
            "error_handling": self.error_handling_observations,
            "security": self.security_observations,
            "recommendations": self.improvement_recommendations,
        }


class CodeAnalyzer:
    """Analyzes source code across languages."""

    def __init__(self) -> None:
        self.registry = get_language_registry()

    def analyze(self, code: str, language_name_or_slug: str) -> AnalysisReport:
        spec = self.registry.get(language_name_or_slug)
        lang_name = spec.name if spec else language_name_or_slug

        lines = code.splitlines()
        total_lines = len(lines)
        blank_lines = sum(1 for line in lines if not line.strip())
        comment_lines = sum(1 for line in lines if line.strip().startswith(("#", "//", "--", "/*", "*", ";", "%")))
        code_lines = total_lines - blank_lines - comment_lines

        # Structural counts
        func_patterns = [r"\bdef\s+\w+", r"\bfunc\s+\w+", r"\bfn\s+\w+", r"\bfunction\s+\w+", r"\bpublic\s+(?:static\s+)?[\w<>]+\s+\w+\s*\("]
        class_patterns = [r"\bclass\s+\w+", r"\bstruct\s+\w+", r"\binterface\s+\w+"]

        funcs = sum(len(re.findall(pat, code)) for pat in func_patterns)
        classes = sum(len(re.findall(pat, code)) for pat in class_patterns)

        # Indentation depth
        max_depth = 0
        for line in lines:
            if line.strip():
                leading_spaces = len(line) - len(line.lstrip(" "))
                depth = leading_spaces // 4
                if depth > max_depth:
                    max_depth = depth

        metrics = CodeMetrics(
            total_lines=total_lines,
            code_lines=code_lines,
            comment_lines=comment_lines,
            blank_lines=blank_lines,
            functions_count=funcs,
            classes_count=classes,
            estimated_max_indent_depth=max_depth,
        )

        readability: list[str] = []
        maintainability: list[str] = []
        performance: list[str] = []
        error_handling: list[str] = []
        security: list[str] = []
        recommendations: list[str] = []

        # Readability analysis
        if comment_lines == 0 and total_lines > 20:
            readability.append("No comments or documentation strings detected in code over 20 lines.")
            recommendations.append("Add docstrings / comments explaining high-level purpose and function contracts.")
        else:
            readability.append(f"Balanced comment-to-code ratio ({comment_lines} comment lines / {code_lines} code lines).")

        if max_depth >= 4:
            readability.append(f"High nesting level detected (max indent depth: {max_depth}).")
            recommendations.append("Consider early returns, guard clauses, or extracting subroutines to flatten nesting.")

        # Error handling analysis
        try_matches = len(re.findall(r"\b(?:try|catch|except|rescue|Result|Option)\b", code))
        if try_matches > 0:
            error_handling.append(f"Contains {try_matches} error-handling or defensive constructs.")
        else:
            error_handling.append("No explicit error handling (try/catch/except/Result) found.")
            recommendations.append("Ensure external inputs and I/O operations are wrapped with error handling.")

        # Performance considerations
        if re.search(r"for\s+.*?in\s+.*?:.*?\bfor\s+.*?in\s+.*?:", code, re.S):
            performance.append("Nested loops detected; verify time complexity does not scale quadratically on large inputs.")

        # Security check integration
        sec_report = check_security(code)
        if sec_report.risk == SecurityRisk.SAFE:
            security.append("No common high-risk patterns or credential leakage detected.")
        else:
            for f in sec_report.findings:
                security.append(f"[{f.severity.value.upper()}] {f.description}")
                recommendations.append(f"Security mitigation: Review pattern '{f.pattern_matched}' for {f.category}.")

        return AnalysisReport(
            language=lang_name,
            metrics=metrics,
            readability_observations=readability,
            maintainability_observations=maintainability,
            performance_considerations=performance,
            error_handling_observations=error_handling,
            security_observations=security,
            improvement_recommendations=recommendations,
        )


_DEFAULT_ANALYZER = CodeAnalyzer()


def analyze_code(code: str, language: str) -> AnalysisReport:
    return _DEFAULT_ANALYZER.analyze(code, language)
