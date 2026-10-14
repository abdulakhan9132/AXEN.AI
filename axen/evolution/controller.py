"""
axen/evolution/controller.py
=============================
Central orchestrator for the AXEN Self-Evolution Engine.

Implements the Darwin–Gödel Machine cycle:
  DETECT WEAKNESS → GENERATE CANDIDATES → EVALUATE → SELECT → APPROVE → MERGE

Coordinates all evolution components:
- WeaknessDetector: identifies reproducible failures
- CandidateGenerator: produces alternative implementations
- CandidateEvaluator: runs deterministic benchmarks (Phase 4)
- CandidateSelector: ranks and picks the best candidate
- ApprovalGate: human-in-the-loop approval (Phase 6)
- EvolutionHistory: records version lineage

All changes require human approval before merge. No autonomous
self-modification of production code is permitted.
"""

from __future__ import annotations

import logging
import hashlib
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .candidate_generator import CandidateGenerator
from .candidate_selector import CandidateSelector
from .history import EvolutionHistory
from .models import (
    ApprovalRequest,
    BenchmarkResult,
    Candidate,
    CandidateStatus,
    Experiment,
    ExperimentStatus,
)
from .store import EvolutionStore
from .weakness_detector import Weakness, WeaknessDetector

logger = logging.getLogger("axen.evolution.controller")


class EvolutionController:
    """
    Central controller for the AXEN Self-Evolution Engine.

    Orchestrates the full improvement cycle from weakness detection
    through candidate generation, evaluation, selection, and
    human approval.
    """

    def __init__(
        self,
        store: EvolutionStore,
        generator: CandidateGenerator,
        selector: Optional[CandidateSelector] = None,
        detector: Optional[WeaknessDetector] = None,
        evaluator: Any = None,  # CandidateEvaluator from Phase 4
        sandbox: Any = None,  # SandboxExecutor from Phase 6
    ) -> None:
        self.store = store
        self.generator = generator
        self.selector = selector or CandidateSelector()
        self.detector = detector or WeaknessDetector()
        self.history = EvolutionHistory(store)
        self.evaluator = evaluator
        self.sandbox = sandbox

    # -----------------------------------------------------------------------
    # Full Evolution Cycle
    # -----------------------------------------------------------------------

    def run_experiment(
        self,
        weakness: Weakness,
        original_code: str,
        evaluate_fn: Optional[Callable[[Candidate], BenchmarkResult]] = None,
        num_candidates: int = 3,
    ) -> Experiment:
        """
        Run a complete evolution experiment for a single weakness.

        Lifecycle:
        1. Create experiment record
        2. Generate candidate improvements
        3. Evaluate each candidate (if evaluator provided)
        4. Select best candidate
        5. Create approval request (human must approve)

        Args:
            weakness: The identified weakness to address.
            original_code: Current source code of the target file.
            evaluate_fn: Optional function to evaluate a candidate.
            num_candidates: Number of candidates to generate.

        Returns:
            The experiment record (status will be AWAITING_APPROVAL if
            a candidate was selected, FAILED otherwise).
        """
        # Step 1: Create experiment
        experiment = Experiment(
            title=f"Improve {weakness.target_module}",
            description=weakness.description,
            target_module=weakness.target_module,
            target_file=weakness.target_file,
            weakness_description=weakness.description,
            weakness_evidence=weakness.evidence,
            config={"num_candidates": num_candidates},
        )
        experiment.transition_to(ExperimentStatus.DETECTING)
        self.store.save_experiment(experiment)
        logger.info("Created experiment %s: %s", experiment.experiment_id, experiment.title)

        try:
            # Step 2: Generate candidates
            experiment.transition_to(ExperimentStatus.GENERATING)
            self.store.save_experiment(experiment)

            candidates = self.generator.generate_candidates(
                experiment=experiment,
                weakness=weakness,
                original_code=original_code,
                num_candidates=num_candidates,
            )

            if not candidates:
                experiment.transition_to(ExperimentStatus.FAILED)
                experiment.error = "No valid candidates generated"
                self.store.save_experiment(experiment)
                logger.warning("Experiment %s: no candidates generated", experiment.experiment_id)
                return experiment

            # Save candidates
            for cand in candidates:
                self.store.save_candidate(cand)
                experiment.candidate_ids.append(cand.candidate_id)
            self.store.save_experiment(experiment)

            # Step 3: Evaluate candidates
            experiment.transition_to(ExperimentStatus.EVALUATING)
            self.store.save_experiment(experiment)

            benchmark_map: Dict[str, BenchmarkResult] = {}
            for cand in candidates:
                result = self._evaluate_candidate(cand, evaluate_fn)
                if result:
                    benchmark_map[cand.candidate_id] = result
                    self.store.save_benchmark_result(result)
                    cand.benchmark_ids.append(result.result_id)
                    cand.transition_to(
                        CandidateStatus.PASSED if result.is_improvement else CandidateStatus.FAILED
                    )
                else:
                    cand.transition_to(CandidateStatus.FAILED)
                self.store.save_candidate(cand)

            # Step 4: Select best candidate
            experiment.transition_to(ExperimentStatus.SELECTING)
            self.store.save_experiment(experiment)

            best = self.selector.select_best(
                candidates=candidates,
                benchmark_map=benchmark_map,
            )

            if not best:
                experiment.transition_to(ExperimentStatus.FAILED)
                experiment.error = "No candidate met selection criteria"
                self.store.save_experiment(experiment)
                logger.info("Experiment %s: no candidate selected", experiment.experiment_id)
                return experiment

            experiment.selected_candidate_id = best.candidate_id
            self.store.save_candidate(best)

            # Step 5: Create approval request
            benchmark = benchmark_map.get(best.candidate_id)
            approval = ApprovalRequest(
                experiment_id=experiment.experiment_id,
                candidate_id=best.candidate_id,
                request_type="code_merge",
                title=f"Approve: {experiment.title}",
                description=f"Candidate {best.candidate_id} improves {weakness.target_module}",
                diff_summary=best.diff_summary,
                benchmark_summary=benchmark.to_dict() if benchmark else {},
            )
            self.store.save_approval(approval)

            experiment.transition_to(ExperimentStatus.AWAITING_APPROVAL)
            self.store.save_experiment(experiment)
            logger.info(
                "Experiment %s awaiting approval for candidate %s",
                experiment.experiment_id, best.candidate_id,
            )
            return experiment

        except Exception as exc:
            experiment.transition_to(ExperimentStatus.FAILED)
            experiment.error = str(exc)
            self.store.save_experiment(experiment)
            logger.error("Experiment %s failed: %s", experiment.experiment_id, exc)
            return experiment

    # -----------------------------------------------------------------------
    # Weakness Detection
    # -----------------------------------------------------------------------

    def detect_weaknesses(
        self,
        test_output: Optional[str] = None,
        benchmark_data: Optional[List[Dict[str, Any]]] = None,
        baselines: Optional[Dict[str, float]] = None,
    ) -> List[Weakness]:
        """
        Detect reproducible weaknesses from test results and benchmarks.

        Args:
            test_output: Raw pytest output text.
            benchmark_data: List of benchmark result dictionaries.
            baselines: Baseline metrics for regression detection.

        Returns:
            Prioritized list of detected weaknesses.
        """
        all_weaknesses: List[Weakness] = []

        if test_output:
            all_weaknesses.extend(
                self.detector.detect_from_test_output(test_output)
            )

        if benchmark_data:
            all_weaknesses.extend(
                self.detector.detect_from_benchmark_results(
                    benchmark_data, baselines=baselines
                )
            )

        prioritized = self.detector.prioritize(all_weaknesses)
        logger.info("Detected %d weaknesses (%d prioritized)", len(all_weaknesses), len(prioritized))
        return prioritized

    # -----------------------------------------------------------------------
    # Batch Evolution
    # -----------------------------------------------------------------------

    def run_evolution_cycle(
        self,
        weaknesses: List[Weakness],
        code_reader: Callable[[str], str],
        evaluate_fn: Optional[Callable[[Candidate], BenchmarkResult]] = None,
        max_experiments: int = 5,
        num_candidates_per: int = 3,
    ) -> List[Experiment]:
        """
        Run a batch of evolution experiments for multiple weaknesses.

        Args:
            weaknesses: Prioritized list of weaknesses to address.
            code_reader: Function that reads source code given a file path.
            evaluate_fn: Optional evaluation function for candidates.
            max_experiments: Maximum number of experiments to run.
            num_candidates_per: Candidates to generate per experiment.

        Returns:
            List of experiment records.
        """
        experiments: List[Experiment] = []

        for weakness in weaknesses[:max_experiments]:
            try:
                original_code = code_reader(weakness.target_file)
            except Exception as exc:
                logger.warning(
                    "Cannot read source for %s: %s", weakness.target_file, exc
                )
                continue

            experiment = self.run_experiment(
                weakness=weakness,
                original_code=original_code,
                evaluate_fn=evaluate_fn,
                num_candidates=num_candidates_per,
            )
            experiments.append(experiment)

        logger.info(
            "Evolution cycle complete: %d experiments (%d awaiting approval)",
            len(experiments),
            sum(1 for e in experiments if e.status == ExperimentStatus.AWAITING_APPROVAL),
        )
        return experiments

    # -----------------------------------------------------------------------
    # Approval Processing
    # -----------------------------------------------------------------------

    def run_observation_loop(
        self,
        observe: Callable[[], List[Weakness]],
        code_reader: Callable[[str], str],
        evaluate_fn: Callable[[Candidate], BenchmarkResult],
        *,
        max_cycles: int = 3,
        max_experiments: int = 5,
        num_candidates: int = 3,
        time_budget_seconds: float = 300.0,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> List[Experiment]:
        """Observe and experiment within explicit budgets; never merge or auto-start.

        The evaluator must execute code on a fixed trusted suite. This loop
        measures original_code first and compares candidates with it. Failed and
        pending attempts are not repeated for identical evidence and source
        during this run. Time and cancellation are checked between experiments;
        callbacks must enforce their own execution timeouts.
        """
        if any(type(n) is not int or n < 1 for n in
               (max_cycles, max_experiments, num_candidates)):
            raise ValueError("Loop counts must be positive integers")
        if not math.isfinite(time_budget_seconds) or time_budget_seconds <= 0:
            raise ValueError("Time budget must be positive and finite")
        if not callable(evaluate_fn):
            raise ValueError("A measured evaluator is required")

        deadline = time.monotonic() + time_budget_seconds
        attempted: set[str] = set()
        experiments: List[Experiment] = []

        def stopped() -> bool:
            return time.monotonic() >= deadline or bool(should_stop and should_stop())

        for _ in range(max_cycles):
            if stopped() or len(experiments) >= max_experiments:
                break
            weaknesses = observe()
            if not weaknesses:
                break
            for weakness in weaknesses:
                if stopped() or len(experiments) >= max_experiments:
                    return experiments
                original_code = code_reader(weakness.target_file)
                fingerprint = hashlib.sha256(repr((
                    weakness.target_file, weakness.description,
                    sorted(weakness.evidence), original_code,
                )).encode("utf-8")).hexdigest()
                if fingerprint in attempted:
                    continue
                attempted.add(fingerprint)

                def measured_evaluator(candidate: Candidate) -> BenchmarkResult:
                    baseline_candidate = replace(
                        candidate, code=candidate.original_code, proposed_code=candidate.original_code,
                        candidate_id=f"baseline_{candidate.candidate_id}",
                    )
                    baseline = evaluate_fn(baseline_candidate)
                    baseline.candidate_id = baseline_candidate.candidate_id
                    baseline.experiment_id = candidate.experiment_id
                    result = evaluate_fn(candidate)
                    if (baseline.total_tests <= 0
                            or not 0 <= baseline.passed_tests <= baseline.total_tests
                            or not 0 <= baseline.held_out_passed <= baseline.held_out_total
                            or baseline.total_tests != result.total_tests
                            or baseline.held_out_total != result.held_out_total
                            or baseline.suite_name != result.suite_name):
                        raise ValueError("Baseline and candidate require matching, nonempty suites")
                    baseline.compute_pass_rate()
                    result.baseline_pass_rate = baseline.pass_rate
                    result.baseline_held_out_pass_rate = baseline.held_out_pass_rate
                    self.store.save_benchmark_result(baseline)
                    result.metadata["baseline_result_id"] = baseline.result_id
                    return result

                experiments.append(self.run_experiment(
                    weakness, original_code, measured_evaluator, num_candidates,
                ))
        return experiments

    def approve_experiment(
        self,
        experiment_id: str,
        decided_by: str = "human",
    ) -> bool:
        """
        Approve an experiment's selected candidate for merge.

        Returns True if approval was successful.
        """
        experiment = self.store.get_experiment(experiment_id)
        if not experiment:
            logger.error("Experiment %s not found", experiment_id)
            return False

        if experiment.status != ExperimentStatus.AWAITING_APPROVAL:
            logger.warning(
                "Experiment %s is not awaiting approval (status: %s)",
                experiment_id, experiment.status,
            )
            return False

        # Update approval record
        approvals = self.store.list_approvals(experiment_id=experiment_id)
        for approval in approvals:
            if approval.status.value == "pending":
                approval.approve(decided_by=decided_by)
                self.store.save_approval(approval)

        # Update experiment
        experiment.transition_to(ExperimentStatus.APPROVED)
        self.store.save_experiment(experiment)

        # Record in evolution history
        if experiment.selected_candidate_id:
            candidate = self.store.get_candidate(experiment.selected_candidate_id)
            benchmarks = self.store.list_benchmark_results(
                candidate_id=experiment.selected_candidate_id
            )
            benchmark = benchmarks[0] if benchmarks else None
            if candidate:
                self.history.record_improvement(
                    experiment=experiment,
                    candidate=candidate,
                    benchmark=benchmark,
                    accepted_by=decided_by,
                )

        logger.info("Experiment %s approved by %s", experiment_id, decided_by)
        return True

    def reject_experiment(
        self,
        experiment_id: str,
        reason: str = "",
        decided_by: str = "human",
    ) -> bool:
        """Reject an experiment's selected candidate."""
        experiment = self.store.get_experiment(experiment_id)
        if not experiment:
            return False

        if experiment.status != ExperimentStatus.AWAITING_APPROVAL:
            return False

        # Update approval records
        approvals = self.store.list_approvals(experiment_id=experiment_id)
        for approval in approvals:
            if approval.status.value == "pending":
                approval.reject(reason=reason, decided_by=decided_by)
                self.store.save_approval(approval)

        experiment.transition_to(ExperimentStatus.REJECTED)
        experiment.error = reason or "Rejected by human review"
        self.store.save_experiment(experiment)

        # Mark selected candidate as rejected
        if experiment.selected_candidate_id:
            candidate = self.store.get_candidate(experiment.selected_candidate_id)
            if candidate:
                candidate.transition_to(CandidateStatus.REJECTED)
                self.store.save_candidate(candidate)

        logger.info("Experiment %s rejected by %s: %s", experiment_id, decided_by, reason)
        return True

    # -----------------------------------------------------------------------
    # Internal Helpers
    # -----------------------------------------------------------------------

    def _evaluate_candidate(
        self,
        candidate: Candidate,
        evaluate_fn: Optional[Callable[[Candidate], BenchmarkResult]] = None,
    ) -> Optional[BenchmarkResult]:
        """Evaluate a single candidate using the provided evaluator."""
        candidate.transition_to(CandidateStatus.EVALUATING)

        try:
            if evaluate_fn or self.evaluator:
                result = (evaluate_fn(candidate) if evaluate_fn
                          else self.evaluator.evaluate(candidate))
                # Counts, not an evaluator's claimed pass rate, are evidence.
                for total, passed in ((result.total_tests, result.passed_tests),
                                      (result.held_out_total, result.held_out_passed)):
                    if type(total) is not int or type(passed) is not int or not 0 <= passed <= total:
                        raise ValueError("Invalid benchmark test counts")
                if result.total_tests == 0:
                    raise ValueError("Benchmark contains no tests")
                if not math.isfinite(result.baseline_pass_rate) or not 0 <= result.baseline_pass_rate <= 1:
                    raise ValueError("Invalid baseline pass rate")
                held_out_baseline = (result.baseline_pass_rate if result.baseline_held_out_pass_rate is None
                                     else result.baseline_held_out_pass_rate)
                if not math.isfinite(held_out_baseline) or not 0 <= held_out_baseline <= 1:
                    raise ValueError("Invalid held-out baseline pass rate")
                result.candidate_id = candidate.candidate_id
                result.experiment_id = candidate.experiment_id
                result.failed_tests = result.total_tests - result.passed_tests
                result.compute_pass_rate()
                result.improvement_delta = result.pass_rate - result.baseline_pass_rate
                result.detect_overfitting()
                result.is_improvement = (
                    result.improvement_delta > 0 and not result.is_overfitting
                    and (result.held_out_total == 0
                         or result.held_out_pass_rate >= held_out_baseline)
                )
                return result

            logger.warning(
                "No evaluator for candidate %s; improvement cannot be established",
                candidate.candidate_id,
            )
            return None

        except Exception as exc:
            logger.warning(
                "Evaluation failed for candidate %s: %s",
                candidate.candidate_id, exc,
            )
            return None

    def get_status(self) -> Dict[str, Any]:
        """Return current evolution engine status."""
        stats = self.store.get_stats()
        stats["history_summary"] = self.history.get_improvement_summary()
        return stats
