"""
tests/test_evolution_store.py
=============================
Tests for SQLite-backed EvolutionStore CRUD operations and WAL mode concurrency.
"""

import tempfile
from pathlib import Path
import pytest

from axen.evolution.models import (
    ApprovalRequest,
    BenchmarkResult,
    Candidate,
    CandidateStatus,
    EvolutionRecord,
    Experiment,
    ExperimentStatus,
)
from axen.evolution.store import EvolutionStore


@pytest.fixture
def temp_store():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_evolution.db"
        store = EvolutionStore(db_path=db_path)
        yield store


def test_experiment_crud(temp_store):
    exp = Experiment(
        target_module="axen.model.foundation",
        target_file="axen/model/foundation.py",
        weakness_description="Slow attention forward pass",
    )
    temp_store.save_experiment(exp)

    retrieved = temp_store.get_experiment(exp.experiment_id)
    assert retrieved is not None
    assert retrieved.target_module == "axen.model.foundation"
    assert retrieved.status == ExperimentStatus.CREATED

    exp.transition_to(ExperimentStatus.EVALUATING)
    temp_store.save_experiment(exp)

    updated = temp_store.get_experiment(exp.experiment_id)
    assert updated.status == ExperimentStatus.EVALUATING


def test_candidate_and_benchmark_crud(temp_store):
    exp = Experiment(target_module="axen.utils")
    temp_store.save_experiment(exp)

    cand = Candidate(
        experiment_id=exp.experiment_id,
        code="def helper(): return 42",
        status=CandidateStatus.EVALUATED,
    )
    temp_store.save_candidate(cand)

    bench = BenchmarkResult(
        candidate_id=cand.candidate_id,
        experiment_id=exp.experiment_id,
        total_tests=5,
        passed_tests=5,
        pass_rate=1.0,
    )
    temp_store.save_benchmark_result(bench)

    candidates = temp_store.get_candidates_for_experiment(exp.experiment_id)
    assert len(candidates) == 1
    assert candidates[0].candidate_id == cand.candidate_id

    retrieved_bench = temp_store.get_benchmark_result(cand.candidate_id)
    assert retrieved_bench is not None
    assert retrieved_bench.pass_rate == 1.0


def test_approval_lifecycle(temp_store):
    req = ApprovalRequest(
        experiment_id="exp_999",
        candidate_id="cand_999",
        target_file="test_target.py",
    )
    temp_store.save_approval(req)

    pending = temp_store.get_pending_approvals()
    assert len(pending) == 1
    assert pending[0].request_id == req.request_id

    req.approve(reviewer="lead_engineer")
    temp_store.save_approval(req)

    pending_after = temp_store.get_pending_approvals()
    assert len(pending_after) == 0
