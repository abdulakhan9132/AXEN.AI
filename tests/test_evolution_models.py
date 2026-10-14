"""
tests/test_evolution_models.py
==============================
Unit tests for evolution engine data models, lineage tracking, and serialization.
"""

import pytest
from axen.evolution.models import (
    ApprovalRequest,
    ApprovalStatus,
    BenchmarkResult,
    Candidate,
    CandidateStatus,
    EvolutionRecord,
    Experiment,
    ExperimentStatus,
)


def test_experiment_lifecycle():
    exp = Experiment(
        target_module="axen.utils.math",
        target_file="axen/utils/math.py",
        weakness_description="Division by zero edge case not handled",
    )
    assert exp.status == ExperimentStatus.CREATED
    assert exp.generation == 0

    exp.transition_to(ExperimentStatus.GENERATING)
    assert exp.status == ExperimentStatus.GENERATING

    exp.transition_to(ExperimentStatus.APPROVED)
    assert exp.status == ExperimentStatus.APPROVED
    assert exp.completed_at is not None

    d = exp.to_dict()
    assert d["status"] == "approved"
    assert d["target_module"] == "axen.utils.math"

    restored = Experiment.from_dict(d)
    assert restored.experiment_id == exp.experiment_id
    assert restored.status == ExperimentStatus.APPROVED


def test_candidate_lineage():
    parent = Candidate(
        experiment_id="exp_001",
        generation=0,
        code="def solve(x): return x + 1",
    )
    child = Candidate(
        experiment_id="exp_001",
        parent_id=parent.candidate_id,
        generation=1,
        code="def solve(x): return x + 2",
        rationale="Optimize constant term",
    )
    assert child.parent_id == parent.candidate_id
    assert child.generation == 1
    assert child.status == CandidateStatus.PENDING

    d = child.to_dict()
    assert d["parent_id"] == parent.candidate_id
    restored = Candidate.from_dict(d)
    assert restored.parent_id == parent.candidate_id


def test_benchmark_result_overfitting_detection():
    # Case 1: In-distribution passes, held-out passes -> No overfitting
    bench_good = BenchmarkResult(
        total_tests=10,
        passed_tests=9,
        held_out_total=4,
        held_out_passed=4,
        baseline_pass_rate=0.5,
    )
    bench_good.compute_pass_rate()
    assert bench_good.pass_rate == 0.9
    assert bench_good.held_out_pass_rate == 1.0
    assert not bench_good.detect_overfitting()

    # Case 2: In-distribution improves 100%, held-out drops -> Overfitting flagged
    bench_overfit = BenchmarkResult(
        total_tests=10,
        passed_tests=10,
        held_out_total=4,
        held_out_passed=1,
        baseline_pass_rate=0.5,
    )
    bench_overfit.compute_pass_rate()
    assert bench_overfit.pass_rate == 1.0
    assert bench_overfit.held_out_pass_rate == 0.25
    assert bench_overfit.detect_overfitting()


def test_approval_request_lifecycle():
    req = ApprovalRequest(
        experiment_id="exp_001",
        candidate_id="cand_001",
        target_file="test.py",
        diff="+ new line\n- old line",
    )
    assert req.status == ApprovalStatus.PENDING

    req.approve(reviewer="lead_dev", notes="Verified in sandbox")
    assert req.status == ApprovalStatus.APPROVED
    assert req.reviewer == "lead_dev"
    assert req.notes == "Verified in sandbox"
