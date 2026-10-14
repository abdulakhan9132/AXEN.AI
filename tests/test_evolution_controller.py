"""
tests/test_evolution_controller.py
==================================
Tests for end-to-end EvolutionController cycle:
Weakness detection -> candidate generation -> evaluation -> selection -> approval.
"""

import tempfile
from pathlib import Path
import pytest

from axen.evolution.candidate_generator import CandidateGenerator
from axen.evolution.candidate_selector import CandidateSelector
from axen.evolution.controller import EvolutionController
from axen.evolution.models import BenchmarkResult, Candidate, ExperimentStatus
from axen.evolution.store import EvolutionStore
from axen.evolution.weakness_detector import Weakness, WeaknessDetector


def test_evolution_controller_full_cycle():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "evo_ctrl.db"
        store = EvolutionStore(db_path=db_path)
        generator = CandidateGenerator(model=None)  # Uses AST/rule-based mutations
        selector = CandidateSelector()
        detector = WeaknessDetector()

        controller = EvolutionController(
            store=store,
            generator=generator,
            selector=selector,
            detector=detector,
        )

        original_code = "def compute_sum(a, b):\n    return a + b\n"
        weakness = Weakness(
            target_module="test_mod",
            target_file="test_mod.py",
            description="Lacks validation for None inputs",
        )

        def mock_evaluator(cand: Candidate) -> BenchmarkResult:
            return BenchmarkResult(
                candidate_id=cand.candidate_id,
                experiment_id=cand.experiment_id,
                total_tests=4,
                passed_tests=4,
                pass_rate=1.0,
                held_out_total=2,
                held_out_passed=2,
                held_out_pass_rate=1.0,
                is_improvement=True,
            )

        exp = controller.run_experiment(
            weakness=weakness,
            original_code=original_code,
            evaluate_fn=mock_evaluator,
            num_candidates=2,
        )

        assert exp.status == ExperimentStatus.AWAITING_APPROVAL
        assert exp.selected_candidate_id is not None
        assert len(exp.candidate_ids) == 2

        # Verify approval was created
        pending = store.get_pending_approvals()
        assert len(pending) == 1
        assert pending[0].candidate_id == exp.selected_candidate_id
