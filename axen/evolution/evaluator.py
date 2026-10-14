"""Fixed-suite evaluation through a capability-contained arithmetic interpreter."""
from __future__ import annotations
import ast
import time
from .models import BenchmarkResult
from .practice import ArithmeticSandbox


class CandidateEvaluator:
    def __init__(self, sandbox=None):
        self.sandbox = sandbox or ArithmeticSandbox()

    def evaluate_candidate(self, candidate, suite, baseline_pass_rate=0.0, function_name=None):
        train, held = suite.partition()
        result = BenchmarkResult(candidate_id=candidate.candidate_id, experiment_id=candidate.experiment_id,
            suite_name=suite.name, total_tests=len(train), held_out_total=len(held), baseline_pass_rate=baseline_pass_rate)
        latencies = []
        if not isinstance(self.sandbox, ArithmeticSandbox):
            raise ValueError("Candidate function evaluation requires arithmetic capability containment")
        try:
            if function_name and ast.parse(candidate.code).body[0].name != function_name:
                raise ValueError("Benchmark function mismatch")
            for cases, holdout in ((train, False), (held, True)):
                for case in cases:
                    start = time.perf_counter()
                    try:
                        inputs = case.input_data
                        if isinstance(inputs, dict): raise ValueError("Keyword inputs outside arithmetic curriculum")
                        args = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]
                        actual = self.sandbox.run(candidate.code, args)
                        if case.verify(actual):
                            if holdout: result.held_out_passed += 1
                            else: result.passed_tests += 1
                        else: result.failed_test_names.append(case.name)
                    except (ValueError, KeyError, ZeroDivisionError) as exc:
                        result.error_messages.append(str(exc)); result.failed_test_names.append(case.name)
                    latencies.append((time.perf_counter() - start) * 1000)
        except (ValueError, SyntaxError, AttributeError, IndexError) as exc:
            result.error_messages.append(str(exc))
        result.failed_tests = result.total_tests - result.passed_tests
        result.compute_pass_rate()
        result.avg_execution_time_ms = sum(latencies) / max(1, len(latencies))
        result.max_execution_time_ms = max(latencies, default=0)
        result.metadata["containment"] = self.sandbox.containment
        result.improvement_delta = result.pass_rate - baseline_pass_rate
        result.is_improvement = result.improvement_delta > 0 and (not held or result.held_out_pass_rate >= baseline_pass_rate)
        result.detect_overfitting()
        return result
