"""
axen/evolution/history.py
=========================
Tracks parent-child version relationships for the evolution lineage graph.

Provides methods to record improvements, reconstruct ancestry paths,
and query the full evolution tree for any module.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .models import (
    ApprovalRequest,
    BenchmarkResult,
    Candidate,
    EvolutionRecord,
    Experiment,
)
from .store import EvolutionStore

logger = logging.getLogger("axen.evolution.history")


class EvolutionHistory:
    """
    Manages the version lineage graph for evolution improvements.

    Each accepted improvement creates an EvolutionRecord linking
    the parent version to the child version, forming a directed tree.
    """

    def __init__(self, store: EvolutionStore) -> None:
        self.store = store

    def record_improvement(
        self,
        experiment: Experiment,
        candidate: Candidate,
        benchmark: Optional[BenchmarkResult] = None,
        improvement_type: str = "code",
        accepted_by: str = "human",
    ) -> EvolutionRecord:
        """
        Record an accepted improvement in the evolution history.

        Args:
            experiment: The experiment that produced this improvement.
            candidate: The selected candidate that was approved.
            benchmark: Optional benchmark result for delta computation.
            improvement_type: Type of improvement (code, strategy, model_weights, etc.).
            accepted_by: Who approved this improvement.

        Returns:
            The created EvolutionRecord.
        """
        correctness_delta = 0.0
        performance_delta = 0.0
        if benchmark:
            correctness_delta = benchmark.improvement_delta
            performance_delta = (
                benchmark.baseline_pass_rate - benchmark.pass_rate
                if benchmark.baseline_pass_rate > 0
                else 0.0
            )

        record = EvolutionRecord(
            parent_id=candidate.parent_id,
            child_id=candidate.candidate_id,
            experiment_id=experiment.experiment_id,
            parent_version=f"gen_{max(0, candidate.generation - 1)}",
            child_version=f"gen_{candidate.generation}",
            target_module=experiment.target_module,
            target_file=experiment.target_file,
            improvement_type=improvement_type,
            improvement_summary=candidate.rationale,
            correctness_delta=correctness_delta,
            performance_delta=performance_delta,
            generation=candidate.generation,
            was_accepted=True,
            accepted_by=accepted_by,
        )

        self.store.save_evolution_record(record)
        logger.info(
            "Recorded evolution: %s -> %s (experiment %s, type %s)",
            record.parent_id or "root",
            record.child_id,
            experiment.experiment_id,
            improvement_type,
        )
        return record

    def get_ancestry(self, child_id: str) -> List[EvolutionRecord]:
        """Get the full ancestry chain for a given child version."""
        return self.store.get_ancestry_chain(child_id)

    def get_module_history(
        self,
        target_module: str,
        limit: int = 100,
    ) -> List[EvolutionRecord]:
        """Get all evolution records for a specific module."""
        return self.store.get_evolution_history(
            target_module=target_module, limit=limit
        )

    def get_generation_count(self, target_module: str) -> int:
        """Get the current generation number for a module."""
        records = self.store.get_evolution_history(target_module=target_module, limit=1)
        if records:
            return records[0].generation
        return 0

    def get_improvement_summary(self) -> Dict[str, Any]:
        """
        Get aggregate improvement statistics across all modules.

        Returns:
            Dictionary with total improvements, average deltas, etc.
        """
        records = self.store.get_evolution_history(limit=1000)
        if not records:
            return {
                "total_improvements": 0,
                "accepted_improvements": 0,
                "modules_improved": 0,
                "avg_correctness_delta": 0.0,
                "total_generations": 0,
            }

        accepted = [r for r in records if r.was_accepted]
        modules = set(r.target_module for r in accepted)
        avg_delta = (
            sum(r.correctness_delta for r in accepted) / len(accepted)
            if accepted
            else 0.0
        )
        max_gen = max((r.generation for r in records), default=0)

        return {
            "total_improvements": len(records),
            "accepted_improvements": len(accepted),
            "modules_improved": len(modules),
            "avg_correctness_delta": round(avg_delta, 4),
            "total_generations": max_gen,
            "improved_modules": sorted(modules),
        }
