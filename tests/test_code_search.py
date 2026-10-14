"""
tests/test_code_search.py
=========================
Tests for EvolutionaryCodeSearch, CandidateEvaluator, and anti-overfitting test partitioning.
"""

import pytest
from axen.evolution.benchmark_suite import BenchmarkSuite
from axen.evolution.candidate_generator import CandidateGenerator
from axen.evolution.candidate_selector import CandidateSelector
from axen.evolution.code_search import EvolutionaryCodeSearch
from axen.evolution.evaluator import CandidateEvaluator
from axen.evolution.models import Candidate, Experiment
from axen.evolution.weakness_detector import Weakness


def test_benchmark_suite_partitioning():
    suite = BenchmarkSuite(name="math_tests", split_ratio=0.7, seed=42)
    for i in range(10):
        suite.add_test(f"test_{i}", input_data=(i, i), expected_output=2 * i)

    train, held_out = suite.partition()
    assert len(train) == 7
    assert len(held_out) == 3
    # Ensure disjoint sets
    train_names = {t.name for t in train}
    held_out_names = {t.name for t in held_out}
    assert train_names.isdisjoint(held_out_names)


def test_candidate_evaluator():
    evaluator = CandidateEvaluator()
    suite = BenchmarkSuite(name="add_test", split_ratio=0.5, seed=123)
    suite.add_test("add_1", input_data=(1, 2), expected_output=3)
    suite.add_test("add_2", input_data=(3, 4), expected_output=7)

    cand = Candidate(
        experiment_id="exp_test",
        code="def solve(a, b):\n    return a + b\n",
    )

    result = evaluator.evaluate_candidate(
        candidate=cand,
        suite=suite,
        baseline_pass_rate=0.0,
        function_name="solve",
    )

    assert result.pass_rate == 1.0
    assert result.held_out_pass_rate == 1.0
    assert result.is_improvement is True
    assert result.is_overfitting is False


def test_evolutionary_code_search():
    generator = CandidateGenerator(model=None)
    evaluator = CandidateEvaluator()
    selector = CandidateSelector()
    search = EvolutionaryCodeSearch(generator=generator, evaluator=evaluator, selector=selector)

    suite = BenchmarkSuite(name="search_suite", split_ratio=0.6, seed=42)
    suite.add_test("t1", input_data=(2, 3), expected_output=5)
    suite.add_test("t2", input_data=(10, 5), expected_output=15)
    suite.add_test("t3", input_data=(-1, 1), expected_output=0)

    exp = Experiment(target_module="calc")
    weakness = Weakness(target_module="calc", description="Need fast addition")
    code = "def add(a, b):\n    return a + b\n"

    ranked = search.search_improvements(
        experiment=exp,
        weakness=weakness,
        original_code=code,
        benchmark_suite=suite,
        num_candidates=2,
        function_name="add",
    )

    assert isinstance(ranked, list)
    # The no-model fallback only adds comments. It cannot improve an already
    # correct baseline and must not be promoted as a successful experiment.
    assert ranked == []
