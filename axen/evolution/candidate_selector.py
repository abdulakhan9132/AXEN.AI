"""
axen/evolution/candidate_selector.py
======================================
Selects the most promising candidate from a set of evaluated alternatives.

Uses Pareto-optimal ranking on (correctness, performance, resource) metrics,
with configurable weights. Enforces anti-overfitting checks before selection.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .models import BenchmarkResult, Candidate, CandidateStatus

logger = logging.getLogger("axen.evolution.candidate_selector")


class CandidateSelector:
    """
    Ranks and selects the best candidate based on multi-objective criteria.

    Selection criteria (default weights):
    - Correctness: 60% (test pass rate on both in-dist and held-out)
    - Performance: 25% (execution speed improvement)
    - Resource efficiency: 15% (memory usage reduction)

    A candidate is only selectable if:
    1. It passes all in-distribution tests (or improves pass rate)
    2. It does not exhibit overfitting (held-out gap check)
    3. Its composite score exceeds the baseline
    """

    def __init__(
        self,
        correctness_weight: float = 0.6,
        performance_weight: float = 0.25,
        resource_weight: float = 0.15,
        min_improvement_threshold: float = 0.01,
        max_overfitting_gap: float = 0.5,
    ) -> None:
        self.correctness_weight = correctness_weight
        self.performance_weight = performance_weight
        self.resource_weight = resource_weight
        self.min_improvement_threshold = min_improvement_threshold
        self.max_overfitting_gap = max_overfitting_gap

    def select_best(
        self,
        candidates: List[Candidate],
        benchmark_map: Dict[str, BenchmarkResult],
        baseline_score: float = 0.0,
    ) -> Optional[Candidate]:
        """
        Select the single best candidate from evaluated candidates.

        Args:
            candidates: List of candidates with benchmark results.
            benchmark_map: Mapping from candidate_id to BenchmarkResult.
            baseline_score: The current baseline composite score to beat.

        Returns:
            The best candidate, or None if no candidate qualifies.
        """
        scored = self.rank_candidates(candidates, benchmark_map)
        if not scored:
            logger.info("No candidates qualified for selection")
            return None

        best_candidate, best_score = scored[0]

        # Must exceed baseline by minimum threshold
        if best_score <= baseline_score + self.min_improvement_threshold:
            logger.info(
                "Best candidate %s score %.4f does not exceed baseline %.4f + threshold %.4f",
                best_candidate.candidate_id, best_score,
                baseline_score, self.min_improvement_threshold,
            )
            return None

        # Anti-overfitting check
        benchmark = benchmark_map.get(best_candidate.candidate_id)
        if benchmark and benchmark.detect_overfitting(threshold=self.max_overfitting_gap):
            logger.warning(
                "Best candidate %s flagged for overfitting (held-out gap too large)",
                best_candidate.candidate_id,
            )
            return None

        best_candidate.transition_to(CandidateStatus.SELECTED)
        logger.info(
            "Selected candidate %s with composite score %.4f (baseline: %.4f)",
            best_candidate.candidate_id, best_score, baseline_score,
        )
        return best_candidate

    def rank_candidates(
        self,
        candidates: List[Candidate],
        benchmark_map: Optional[Dict[str, BenchmarkResult]] = None,
    ) -> List[Tuple[Candidate, float]]:
        """
        Rank candidates by composite score, filtering out failures.

        Returns:
            List of (candidate, composite_score) tuples, sorted descending.
        """
        scored: List[Tuple[Candidate, float]] = []

        if benchmark_map is None:
            benchmark_map = {}
            for c in candidates:
                b = getattr(c, "benchmark_result", None)
                if b is not None:
                    benchmark_map[c.candidate_id] = b

        for candidate in candidates:
            if candidate.status == CandidateStatus.FAILED:
                continue

            benchmark = benchmark_map.get(candidate.candidate_id)
            if (not benchmark or benchmark.total_tests <= 0
                    or not benchmark.is_improvement
                    or benchmark.detect_overfitting(threshold=self.max_overfitting_gap)):
                continue

            # Compute individual scores (normalized 0-1)
            correctness_score = benchmark.pass_rate
            performance_score = self._normalize_performance(benchmark)
            resource_score = self._normalize_resource(benchmark)

            candidate.correctness_score = correctness_score
            candidate.performance_score = performance_score
            candidate.resource_score = resource_score
            candidate.held_out_score = benchmark.held_out_pass_rate

            composite = candidate.compute_composite_score(
                correctness_weight=self.correctness_weight,
                performance_weight=self.performance_weight,
                resource_weight=self.resource_weight,
            )

            scored.append((candidate, composite))

        # Sort by composite score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _normalize_performance(self, benchmark: BenchmarkResult) -> float:
        """
        Normalize execution time to a 0-1 score.

        Lower execution time = higher score.
        Uses an inverse mapping: score = 1 / (1 + avg_time_ms / 1000)
        """
        if benchmark.avg_execution_time_ms <= 0:
            return 1.0
        return round(1.0 / (1.0 + benchmark.avg_execution_time_ms / 1000.0), 4)

    def _normalize_resource(self, benchmark: BenchmarkResult) -> float:
        """
        Normalize memory usage to a 0-1 score.

        Lower memory = higher score.
        Uses an inverse mapping: score = 1 / (1 + peak_memory_mb / 512)
        """
        if benchmark.peak_memory_mb <= 0:
            return 1.0
        return round(1.0 / (1.0 + benchmark.peak_memory_mb / 512.0), 4)

    def get_pareto_front(
        self,
        candidates: List[Candidate],
        benchmark_map: Dict[str, BenchmarkResult],
    ) -> List[Candidate]:
        """
        Compute the Pareto-optimal front over (correctness, performance, resource).

        A candidate is Pareto-optimal if no other candidate is better on
        all three objectives simultaneously.
        """
        scored = self.rank_candidates(candidates, benchmark_map)
        if not scored:
            return []

        pareto: List[Candidate] = []
        for candidate, _ in scored:
            dominated = False
            for other, _ in scored:
                if other.candidate_id == candidate.candidate_id:
                    continue
                if (
                    other.correctness_score >= candidate.correctness_score
                    and other.performance_score >= candidate.performance_score
                    and other.resource_score >= candidate.resource_score
                    and (
                        other.correctness_score > candidate.correctness_score
                        or other.performance_score > candidate.performance_score
                        or other.resource_score > candidate.resource_score
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                pareto.append(candidate)

        return pareto
