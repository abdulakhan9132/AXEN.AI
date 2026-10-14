"""
axen/evolution/code_search.py
=============================
Evolutionary Code Search (AlphaEvolve architecture).

Iteratively searches for improved code implementations by:
1. Generating candidate mutations/variants targeting a weakness
2. Deterministically evaluating each candidate across train and held-out test sets
3. Filtering candidates that fail regression or overfit
4. Ranking via Pareto optimization on correctness, latency, and memory
5. Archiving all variants and promoting top candidates for human review
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from .benchmark_suite import BenchmarkSuite
from .candidate_generator import CandidateGenerator
from .candidate_selector import CandidateSelector
from .evaluator import CandidateEvaluator
from .models import BenchmarkResult, Candidate, CandidateStatus, Experiment
from .store import EvolutionStore
from .weakness_detector import Weakness

logger = logging.getLogger("axen.evolution.code_search")


class EvolutionaryCodeSearch:
    """
    Search engine for generating, benchmarking, and archiving code variants.
    """

    def __init__(
        self,
        generator: CandidateGenerator,
        evaluator: Optional[CandidateEvaluator] = None,
        selector: Optional[CandidateSelector] = None,
        store: Optional[EvolutionStore] = None,
    ) -> None:
        self.generator = generator
        self.evaluator = evaluator or CandidateEvaluator()
        self.selector = selector or CandidateSelector()
        self.store = store

    def search_improvements(
        self,
        experiment: Experiment,
        weakness: Weakness,
        original_code: str,
        benchmark_suite: BenchmarkSuite,
        num_candidates: int = 3,
        function_name: Optional[str] = None,
    ) -> List[Candidate]:
        """
        Run an evolutionary search cycle for the specified weakness and test suite.

        Args:
            experiment: Active experiment tracking this search.
            weakness: The diagnosed weakness.
            original_code: Baseline implementation code.
            benchmark_suite: Partitioned test suite.
            num_candidates: How many variants to generate.
            function_name: Optional target function name to evaluate.

        Returns:
            List of evaluated and ranked Candidate objects.
        """
        logger.info(
            "Starting evolutionary code search for experiment %s (target: %s)",
            experiment.experiment_id,
            weakness.target_module,
        )

        # 1. Establish baseline benchmark
        baseline_candidate = Candidate(
            experiment_id=experiment.experiment_id,
            code=original_code,
            rationale="Baseline original implementation",
        )
        baseline_bench = self.evaluator.evaluate_candidate(
            candidate=baseline_candidate,
            suite=benchmark_suite,
            baseline_pass_rate=0.0,
            function_name=function_name,
        )
        baseline_pass_rate = baseline_bench.pass_rate
        logger.info(
            "Baseline pass rate: %.2f (held-out: %.2f)",
            baseline_pass_rate,
            baseline_bench.held_out_pass_rate,
        )

        # 2. Generate Candidate Variants
        candidates = self.generator.generate_candidates(
            experiment_id=experiment.experiment_id,
            weakness=weakness,
            original_code=original_code,
            count=num_candidates,
        )

        if not candidates:
            logger.warning("No candidates generated for experiment %s", experiment.experiment_id)
            return []

        # 3. Evaluate Each Candidate
        evaluated_candidates: List[Candidate] = []
        for candidate in candidates:
            try:
                bench = self.evaluator.evaluate_candidate(
                    candidate=candidate,
                    suite=benchmark_suite,
                    baseline_pass_rate=baseline_pass_rate,
                    function_name=function_name,
                )
                candidate.benchmark_result = bench
                bench.baseline_held_out_pass_rate = baseline_bench.held_out_pass_rate
                bench.is_improvement = (bench.improvement_delta > 0 and not bench.is_overfitting
                    and bench.held_out_pass_rate >= baseline_bench.held_out_pass_rate)

                # Update candidate status based on evaluation
                if bench.is_overfitting:
                    candidate.status = CandidateStatus.REJECTED
                    candidate.error = "Rejected: Candidate overfit in-distribution tests."
                elif bench.pass_rate < baseline_pass_rate:
                    candidate.status = CandidateStatus.REJECTED
                    candidate.error = f"Rejected: Pass rate {bench.pass_rate} < baseline {baseline_pass_rate}"
                elif bench.is_improvement:
                    candidate.status = CandidateStatus.EVALUATED
                else:
                    candidate.status = CandidateStatus.EVALUATED

                if self.store is not None:
                    self.store.save_candidate(candidate)
                    self.store.save_benchmark_result(bench)

                evaluated_candidates.append(candidate)
            except Exception as exc:
                logger.exception("Error evaluating candidate %s: %s", candidate.candidate_id, exc)
                candidate.status = CandidateStatus.FAILED
                candidate.error = str(exc)
                if self.store is not None:
                    self.store.save_candidate(candidate)
                evaluated_candidates.append(candidate)

        # 4. Rank candidates using CandidateSelector
        ranked_pairs = self.selector.rank_candidates(evaluated_candidates)
        return [c for c, _ in ranked_pairs]
