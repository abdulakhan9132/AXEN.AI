"""
axen/evolution/benchmark_suite.py
=================================
Benchmark suite management with train / held-out test partitioning for anti-overfitting.

AlphaEvolve principle: To prevent candidates from merely memorizing or gaming
the specific test cases that exposed the weakness, test suites are partitioned
into in-distribution (train) and held-out validation sets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import logging
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("axen.evolution.benchmark_suite")


@dataclass
class TestCase:
    """Individual deterministic test case."""

    name: str
    input_data: Any
    expected_output: Any
    assertion_fn: Optional[Callable[[Any, Any], bool]] = None
    timeout_seconds: float = 5.0
    tags: List[str] = field(default_factory=list)

    def verify(self, actual_output: Any) -> bool:
        """Run verification assertion."""
        if self.assertion_fn is not None:
            return bool(self.assertion_fn(actual_output, self.expected_output))
        return actual_output == self.expected_output


class BenchmarkSuite:
    """
    Manages deterministic test cases and partitions them into in-distribution and held-out sets.
    """

    def __init__(
        self,
        name: str = "default_suite",
        split_ratio: float = 0.7,
        seed: int = 42,
    ) -> None:
        self.name = name
        self.split_ratio = max(0.1, min(0.9, split_ratio))
        self.seed = seed
        self._test_cases: List[TestCase] = []

    def add_test(
        self,
        name: str,
        input_data: Any,
        expected_output: Any,
        assertion_fn: Optional[Callable[[Any, Any], bool]] = None,
        timeout_seconds: float = 5.0,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Add a test case to the suite."""
        self._test_cases.append(
            TestCase(
                name=name,
                input_data=input_data,
                expected_output=expected_output,
                assertion_fn=assertion_fn,
                timeout_seconds=timeout_seconds,
                tags=tags or [],
            )
        )

    def partition(self) -> Tuple[List[TestCase], List[TestCase]]:
        """
        Deterministically split tests into (train_cases, held_out_cases).

        Uses stable hash-based partitioning so tests are consistently split across runs.
        """
        if not self._test_cases:
            return [], []

        # Sort for stable initial ordering
        sorted_cases = sorted(self._test_cases, key=lambda t: t.name)
        rng = random.Random(self.seed)
        shuffled = list(sorted_cases)
        rng.shuffle(shuffled)

        split_idx = max(1, int(len(shuffled) * self.split_ratio))
        train = shuffled[:split_idx]
        held_out = shuffled[split_idx:]
        if not held_out and len(shuffled) > 1:
            held_out = [train.pop()]
        return train, held_out

    @property
    def total_tests(self) -> int:
        return len(self._test_cases)
