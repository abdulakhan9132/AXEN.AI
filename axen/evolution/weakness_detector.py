"""
axen/evolution/weakness_detector.py
====================================
Identifies reproducible weaknesses by scanning test results and
performance metrics to find target modules for improvement.

Detection strategies:
1. Parse pytest/test output for consistent failures
2. Identify modules with below-threshold pass rates
3. Detect performance regressions against baselines
4. Flag resource-heavy operations exceeding budgets
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("axen.evolution.weakness_detector")


@dataclass
class Weakness:
    """A detected, reproducible weakness in the system."""
    weakness_id: str = ""
    target_module: str = ""
    target_file: str = ""
    description: str = ""
    evidence: List[str] = field(default_factory=list)
    severity: str = "medium"  # "low", "medium", "high", "critical"
    failure_count: int = 0
    pass_rate: float = 0.0
    category: str = "test_failure"  # "test_failure", "performance", "resource", "quality"
    suggested_improvement: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class WeaknessDetector:
    """
    Scans test results and performance data to identify reproducible weaknesses.

    Weaknesses must be reproducible (appear in multiple runs) to avoid
    targeting flaky tests or transient issues.
    """

    def __init__(
        self,
        failure_threshold: float = 0.9,
        min_failure_count: int = 2,
        perf_regression_pct: float = 0.2,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.min_failure_count = min_failure_count
        self.perf_regression_pct = perf_regression_pct

    def detect_from_test_output(self, test_output: str) -> List[Weakness]:
        """
        Parse raw pytest output to identify failing test modules.

        Extracts FAILED test lines and groups them by source module.
        """
        weaknesses: List[Weakness] = []
        failed_pattern = re.compile(r"FAILED\s+([\w/.]+)::(\w+)")
        error_pattern = re.compile(r"ERROR\s+([\w/.]+)")

        # Group failures by module
        module_failures: Dict[str, List[str]] = {}
        for match in failed_pattern.finditer(test_output):
            module = match.group(1)
            test_name = match.group(2)
            module_failures.setdefault(module, []).append(test_name)

        for match in error_pattern.finditer(test_output):
            module = match.group(1)
            module_failures.setdefault(module, []).append("(collection_error)")

        for module, failures in module_failures.items():
            if len(failures) >= self.min_failure_count:
                weaknesses.append(Weakness(
                    weakness_id=f"weak_{module.replace('/', '_').replace('.', '_')}",
                    target_module=module,
                    target_file=module if module.endswith(".py") else f"{module}.py",
                    description=f"{len(failures)} test failures in {module}",
                    evidence=[f"FAILED {module}::{t}" for t in failures],
                    severity="high" if len(failures) >= 5 else "medium",
                    failure_count=len(failures),
                    category="test_failure",
                ))

        return weaknesses

    def detect_from_benchmark_results(
        self,
        results: List[Dict[str, Any]],
        baselines: Optional[Dict[str, float]] = None,
    ) -> List[Weakness]:
        """
        Detect weaknesses from benchmark result data.

        Identifies:
        - Modules with pass rates below threshold
        - Performance regressions vs baselines
        - Resource usage exceeding budgets
        """
        weaknesses: List[Weakness] = []
        baselines = baselines or {}

        for result in results:
            module = result.get("module", result.get("target_module", "unknown"))
            pass_rate = result.get("pass_rate", 1.0)
            avg_time = result.get("avg_execution_time_ms", 0.0)
            peak_mem = result.get("peak_memory_mb", 0.0)

            # Check pass rate threshold
            if pass_rate < self.failure_threshold:
                total = result.get("total_tests", 0)
                failed = result.get("failed_tests", 0)
                weaknesses.append(Weakness(
                    weakness_id=f"weak_passrate_{module.replace('/', '_')}",
                    target_module=module,
                    target_file=result.get("target_file", ""),
                    description=f"Pass rate {pass_rate:.1%} below threshold {self.failure_threshold:.1%}",
                    evidence=[f"Passed {total - failed}/{total} tests"],
                    severity="high" if pass_rate < 0.5 else "medium",
                    pass_rate=pass_rate,
                    failure_count=failed,
                    category="test_failure",
                ))

            # Check performance regression
            baseline_time = baselines.get(f"{module}_time", 0.0)
            if baseline_time > 0 and avg_time > baseline_time * (1 + self.perf_regression_pct):
                regression_pct = ((avg_time - baseline_time) / baseline_time) * 100
                weaknesses.append(Weakness(
                    weakness_id=f"weak_perf_{module.replace('/', '_')}",
                    target_module=module,
                    target_file=result.get("target_file", ""),
                    description=f"Performance regression: {regression_pct:.1f}% slower than baseline",
                    evidence=[
                        f"Current avg: {avg_time:.1f}ms",
                        f"Baseline avg: {baseline_time:.1f}ms",
                    ],
                    severity="medium",
                    category="performance",
                ))

            # Check resource budget
            mem_budget = result.get("memory_budget_mb", 512.0)
            if peak_mem > mem_budget:
                weaknesses.append(Weakness(
                    weakness_id=f"weak_resource_{module.replace('/', '_')}",
                    target_module=module,
                    target_file=result.get("target_file", ""),
                    description=f"Peak memory {peak_mem:.1f}MB exceeds budget {mem_budget:.1f}MB",
                    evidence=[f"Peak memory: {peak_mem:.1f}MB, Budget: {mem_budget:.1f}MB"],
                    severity="high" if peak_mem > mem_budget * 1.5 else "medium",
                    category="resource",
                ))

        return weaknesses

    def prioritize(self, weaknesses: List[Weakness]) -> List[Weakness]:
        """
        Sort weaknesses by priority for improvement targeting.

        Priority order: critical > high > medium > low,
        then by failure count descending.
        """
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        return sorted(
            weaknesses,
            key=lambda w: (severity_order.get(w.severity, 4), -w.failure_count),
        )
