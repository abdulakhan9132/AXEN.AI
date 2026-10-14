"""Evidence and budget regressions for the self-improvement controller."""

import pytest

from axen.evolution.candidate_generator import CandidateGenerator
from axen.evolution.candidate_selector import CandidateSelector
from axen.evolution.controller import EvolutionController
from axen.evolution.models import BenchmarkResult, Candidate, CandidateStatus, ExperimentStatus
from axen.evolution.store import EvolutionStore
from axen.evolution.weakness_detector import Weakness


@pytest.fixture
def controller(tmp_path):
    return EvolutionController(
        EvolutionStore(tmp_path / "evolution.db"), CandidateGenerator(),
    )


def weakness(path="example.py"):
    return Weakness(target_module="example", target_file=path,
                    description="Wrong answer", evidence=["Expected 2, got 1"])


def result(passed=4, total=4, held_passed=2, held_total=2, baseline=0.5):
    return BenchmarkResult(total_tests=total, passed_tests=passed,
                           held_out_total=held_total, held_out_passed=held_passed,
                           baseline_pass_rate=baseline)


def test_missing_evaluator_never_creates_approval(controller):
    exp = controller.run_experiment(weakness(), "def solve(x): return x")
    assert exp.status == ExperimentStatus.FAILED
    assert not controller.store.get_pending_approvals()
    assert not controller.store.list_benchmark_results()


@pytest.mark.parametrize("bench", [
    result(total=0, passed=0), result(passed=5), result(baseline=float("nan")),
    result(held_passed=3), result(passed=2), result(held_passed=0),
])
def test_invalid_or_unimproved_evidence_cannot_advance(controller, bench):
    exp = controller.run_experiment(weakness(), "def solve(x): return x",
                                    lambda _: bench, num_candidates=1)
    assert exp.status == ExperimentStatus.FAILED
    assert not controller.store.get_pending_approvals()


def test_selector_does_not_trust_unmeasured_scores():
    candidate = Candidate(composite_score=1.0, status=CandidateStatus.PASSED)
    assert CandidateSelector().select_best([candidate], {}) is None


def test_loop_measures_original_and_records_improvement(controller):
    class Generator:
        def generate_candidates(self, experiment, original_code, **kwargs):
            return [Candidate(experiment_id=experiment.experiment_id,
                              original_code=original_code, code="def solve(x): return x + 1")]

    controller.generator = Generator()
    from axen.evolution.benchmark_suite import BenchmarkSuite
    from axen.evolution.evaluator import CandidateEvaluator
    suite = BenchmarkSuite(name="increment")
    for x in range(8):
        suite.add_test(str(x), x, x + 1)
    evaluator = CandidateEvaluator()
    # These are trusted test snippets; the existing evaluator executes in-process.
    experiments = controller.run_observation_loop(
        lambda: [weakness()], lambda _: "def solve(x): return x",
        lambda c: evaluator.evaluate_candidate(c, suite, function_name="solve"),
        num_candidates=1,
    )
    assert len(experiments) == 1  # Identical observations aren't retried.
    assert experiments[0].status == ExperimentStatus.AWAITING_APPROVAL
    benchmarks = controller.store.list_benchmark_results()
    assert len(benchmarks) == 2
    candidate_bench = next(b for b in benchmarks if b.metadata.get("baseline_result_id"))
    assert candidate_bench.pass_rate == 1.0
    assert candidate_bench.baseline_pass_rate == 0.0
    assert candidate_bench.baseline_held_out_pass_rate == 0.0
    assert controller.store.get_benchmark_result(candidate_bench.metadata["baseline_result_id"])


def test_comment_only_candidates_do_not_count_as_learning(controller):
    experiments = controller.run_observation_loop(
        lambda: [weakness()], lambda _: "def solve(x): return x",
        lambda _: result(), num_candidates=1,
    )
    assert len(experiments) == 1
    assert experiments[0].status == ExperimentStatus.FAILED
    assert not controller.store.get_pending_approvals()


def test_loop_enforces_experiment_limit_and_cancellation(controller):
    experiments = controller.run_observation_loop(
        lambda: [weakness("a.py"), weakness("b.py")], lambda _: "def solve(x): return x",
        lambda _: result(), max_experiments=1, num_candidates=1,
    )
    assert len(experiments) == 1
    assert controller.run_observation_loop(
        lambda: pytest.fail("Cancelled run must not observe"), lambda _: "",
        lambda _: result(), should_stop=lambda: True,
    ) == []


def test_loop_stops_at_time_budget(controller, monkeypatch):
    clock = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr("axen.evolution.controller.time.monotonic", lambda: next(clock))
    assert controller.run_observation_loop(
        lambda: [weakness()], lambda _: pytest.fail("Budget expired"),
        lambda _: result(), time_budget_seconds=1.0,
    ) == []


def test_loop_rejects_mismatched_baseline_suite(controller):
    def evaluate(candidate):
        return result(total=3 if candidate.candidate_id.startswith("baseline_") else 4, passed=2)
    experiments = controller.run_observation_loop(
        lambda: [weakness()], lambda _: "def solve(x): return x", evaluate, num_candidates=1,
    )
    assert experiments[0].status == ExperimentStatus.FAILED
    assert not controller.store.get_pending_approvals()


def test_distinct_held_out_baseline_detects_regression():
    bench = result(passed=4, held_passed=1)
    bench.baseline_held_out_pass_rate = 1.0
    bench.compute_pass_rate()
    assert bench.detect_overfitting()


@pytest.mark.parametrize("options", [{"max_cycles": 0}, {"max_experiments": -1},
                                    {"time_budget_seconds": float("inf")}])
def test_loop_rejects_invalid_budgets(controller, options):
    with pytest.raises(ValueError):
        controller.run_observation_loop(lambda: [], lambda _: "", lambda _: result(), **options)
