"""
tests/test_think_engine.py
==========================
Comprehensive Unit & Integration Test Suite for Think Engine 2.0.

Covers:
1. ThinkMode, ThinkBudget, and ThinkModeConfig definitions & YAML loading
2. ThinkModeSelector priority resolution (override > intent > multi-step > complexity > default)
3. ThinkRequestMetrics, PhaseMetric, ToolCallMetric, and ThinkLogger redaction
4. SubtaskDecomposer, constraint extraction, and dependency ordering
5. Dynamic Re-planning on tool/step failure
6. ClaimVerificationReport, AST syntax checks, numeric consistency, and contradiction detection
7. End-to-end AgentLoop and AssistantController integration with FAST/BALANCED/DEEP modes
8. Edge cases, resource budget exhaustion, and graceful degradation
"""

import time
from unittest.mock import MagicMock, patch
import pytest

from axen.agent import (
    AgentLoop,
    ClaimVerificationReport,
    DecomposedObjective,
    IntentInfo,
    Planner,
    Subtask,
    SubtaskDecomposer,
    ThinkBudget,
    ThinkEngine,
    ThinkEngineConfig,
    ThinkLogger,
    ThinkMode,
    ThinkModeConfig,
    ThinkModeSelector,
    ThinkRequestMetrics,
    ToolCallMetric,
    Verifier,
    VERIFIED,
    FAILED,
    UNCERTAIN,
)
from axen.assistant import AssistantController
from axen.tools import build_default_registry


# ===========================================================================
# 1. Configuration & Budget Tests
# ===========================================================================


class TestThinkConfig:
    def test_mode_from_string(self):
        assert ThinkMode.from_string("fast") == ThinkMode.FAST
        assert ThinkMode.from_string("FAST") == ThinkMode.FAST
        assert ThinkMode.from_string("Balanced") == ThinkMode.BALANCED
        assert ThinkMode.from_string("deep") == ThinkMode.DEEP
        assert ThinkMode.from_string("invalid_mode") == ThinkMode.BALANCED
        assert ThinkMode.from_string(None) == ThinkMode.BALANCED

    def test_budget_exhaustion_checks(self):
        budget = ThinkBudget(
            max_tokens=1000,
            max_runtime_seconds=10.0,
            max_iterations=5,
            max_tool_calls=3,
            max_retries=1,
        )
        assert budget.remaining_tokens(400) == 600
        assert budget.remaining_tokens(1200) == 0
        assert not budget.is_token_budget_exhausted(999)
        assert budget.is_token_budget_exhausted(1000)

        assert not budget.is_time_budget_exhausted(5.0)
        assert budget.is_time_budget_exhausted(10.5)

        assert not budget.is_tool_budget_exhausted(2)
        assert budget.is_tool_budget_exhausted(3)

        assert not budget.is_iteration_budget_exhausted(4)
        assert budget.is_iteration_budget_exhausted(5)

    def test_load_yaml_config(self):
        cfg = ThinkEngineConfig.from_yaml("configs/think_engine.yaml")
        assert cfg.enabled is True
        assert cfg.default_mode == ThinkMode.BALANCED
        assert ThinkMode.FAST in cfg.modes
        assert ThinkMode.BALANCED in cfg.modes
        assert ThinkMode.DEEP in cfg.modes

        # Mode-specific budgets
        fast_cfg = cfg.get_mode_config(ThinkMode.FAST)
        assert fast_cfg.skip_planning is True
        assert fast_cfg.budget.max_tool_calls == 1

        deep_cfg = cfg.get_mode_config(ThinkMode.DEEP)
        assert deep_cfg.enable_subtask_decomposition is True
        assert deep_cfg.enable_claim_verification is True
        assert deep_cfg.allow_replan is True
        assert deep_cfg.max_replan_attempts >= 2


# ===========================================================================
# 2. Intelligent Task Classification & Mode Selection Tests
# ===========================================================================


class TestThinkModeSelector:
    @pytest.fixture
    def selector(self):
        cfg = ThinkEngineConfig.from_yaml("configs/think_engine.yaml")
        return ThinkModeSelector(cfg)

    def test_explicit_override_takes_precedence(self, selector):
        intent = IntentInfo(intent="conversation", complexity="LOW")
        mode, rationale = selector.select("Hello", intent_info=intent, override="deep")
        assert mode == ThinkMode.DEEP
        assert "Explicit override" in rationale

        mode, rationale = selector.select("Complex multi-step task", intent_info=intent, override="fast")
        assert mode == ThinkMode.FAST

    def test_intent_overrides(self, selector):
        intent = IntentInfo(intent="conversation", complexity="LOW")
        mode, rationale = selector.select("Good morning", intent_info=intent)
        assert mode == ThinkMode.FAST

        code_intent = IntentInfo(intent="code", complexity="MEDIUM")
        mode, rationale = selector.select("Write a quicksort in Python", intent_info=code_intent)
        assert mode == ThinkMode.DEEP

    def test_multi_step_detection(self, selector):
        intent = IntentInfo(intent="general", complexity="MEDIUM")
        mode, rationale = selector.select(
            "Find the sales CSV, calculate the average column, and explain the trend.",
            intent_info=intent,
        )
        assert mode == ThinkMode.DEEP
        assert "Multi-step" in rationale

    def test_complexity_fallback(self, selector):
        low_intent = IntentInfo(intent="unlisted_intent", complexity="LOW")
        mode, _ = selector.select("Simple query", intent_info=low_intent)
        assert mode == ThinkMode.FAST

        high_intent = IntentInfo(intent="unlisted_intent", complexity="HIGH")
        mode, _ = selector.select("Hard query", intent_info=high_intent)
        assert mode == ThinkMode.DEEP


# ===========================================================================
# 3. Observability, Telemetry & Structured Logging Tests
# ===========================================================================


class TestThinkMetrics:
    def test_request_metrics_lifecycle(self):
        metrics = ThinkRequestMetrics(
            think_mode=ThinkMode.BALANCED,
            intent="calculation",
            complexity="MEDIUM",
        )
        pm = metrics.start_phase("planning")
        time.sleep(0.005)
        metrics.end_phase(pm)
        assert pm.duration_ms > 0

        tcm = metrics.start_tool_call("calculator")
        time.sleep(0.005)
        metrics.end_tool_call(tcm, success=True, verification="VERIFIED")
        assert tcm.duration_ms > 0
        assert tcm.success is True
        assert metrics.tool_calls_count == 1

        metrics.record_model_call(input_tokens=100, output_tokens=50)
        assert metrics.tokens_input == 100
        assert metrics.tokens_output == 50
        assert metrics.total_tokens == 150

        metrics.record_replan()
        assert metrics.replan_count == 1

        metrics.finish(status="completed")
        assert metrics.total_latency_ms > 0
        assert metrics.status == "completed"

        d = metrics.to_dict()
        assert d["think_mode"] == "balanced"
        assert d["total_tokens"] == 150
        assert d["tool_calls_count"] == 1

    def test_user_summary_hides_private_reasoning(self):
        metrics = ThinkRequestMetrics(think_mode=ThinkMode.DEEP)
        metrics.start_tool_call("python")
        metrics.finish()

        summary = metrics.to_user_summary()
        # Verify no hidden internal prompts, traces, or CoT in user-facing summary
        assert "chain_of_thought" not in summary
        assert "prompt" not in summary
        assert summary["mode"] == "deep"
        assert "python" in summary["tools_used"]

    def test_logger_redaction(self):
        logger = ThinkLogger(redact_fields=["secret_key", "password"])
        assert "password" in logger._redact
        assert "api_key" in logger._redact


# ===========================================================================
# 4. Subtask Decomposition & Dependency Tracking Tests
# ===========================================================================


class TestSubtaskDecomposer:
    def test_file_plus_math_decomposition(self):
        decomposer = SubtaskDecomposer()
        obj = decomposer.decompose(
            "Read data.csv and calculate the average score",
            intent_info=IntentInfo(intent="mixed", complexity="HIGH"),
        )
        assert len(obj.subtasks) >= 2
        # Dependency: calculation must depend on file read
        calc_subtask = next(st for st in obj.subtasks if st.tool in ("calculator", "math_solver"))
        assert len(calc_subtask.depends_on) > 0

    def test_constraint_detection(self):
        decomposer = SubtaskDecomposer()
        obj = decomposer.decompose(
            "Write a sorting function without using built-in sort strictly format as json",
        )
        assert any("sort" in c.lower() for c in obj.constraints)

    def test_dependency_readiness(self):
        st1 = Subtask(id=1, description="Step 1", depends_on=[])
        st2 = Subtask(id=2, description="Step 2", depends_on=[1])
        obj = DecomposedObjective(objective="Test", subtasks=[st1, st2])

        ready = obj.get_ready_subtasks()
        assert len(ready) == 1
        assert ready[0].id == 1

        obj.mark_completed(1, result="Done 1")
        ready_now = obj.get_ready_subtasks()
        assert len(ready_now) == 1
        assert ready_now[0].id == 2
        assert not obj.all_completed()

        obj.mark_completed(2, result="Done 2")
        assert obj.all_completed()


# ===========================================================================
# 5. Dynamic Re-planning & Self-Checking Tests
# ===========================================================================


class TestReplanAndVerification:
    def test_replan_calculator_to_python(self):
        engine = ThinkEngine()
        metrics = ThinkRequestMetrics()
        mode_cfg = engine.config.get_mode_config(ThinkMode.BALANCED)

        failed_step = {
            "step": 1,
            "step_id": 1,
            "tool": "calculator",
            "arguments": {"raw_input": "2 ** 100"},
        }
        alt = engine.attempt_replan(failed_step, "Overflow / SyntaxError", metrics, mode_cfg)
        assert alt is not None
        assert alt["tool"] == "python"
        assert metrics.replan_count == 1

    def test_replan_file_read_to_search(self):
        engine = ThinkEngine()
        metrics = ThinkRequestMetrics()
        mode_cfg = engine.config.get_mode_config(ThinkMode.DEEP)

        failed_step = {
            "step": 1,
            "step_id": 1,
            "tool": "file",
            "arguments": {"path": "missing_file.txt"},
        }
        alt = engine.attempt_replan(failed_step, "FileNotFoundError", metrics, mode_cfg)
        assert alt is not None
        assert alt["tool"] == "file_search"

    def test_replan_budget_exhaustion(self):
        engine = ThinkEngine()
        metrics = ThinkRequestMetrics()
        mode_cfg = engine.config.get_mode_config(ThinkMode.BALANCED)
        mode_cfg.max_replan_attempts = 1
        metrics.replan_count = 1  # Already reached max

        failed_step = {"step": 1, "tool": "calculator", "arguments": {}}
        alt = engine.attempt_replan(failed_step, "Error", metrics, mode_cfg)
        assert alt is None

    def test_claim_verification_numeric_matching(self):
        engine = ThinkEngine()
        observations = [
            {"step": 1, "tool": "calculator", "result": 1200, "success": True},
        ]
        # Text with matching number
        rep = engine.verify_claims("The calculated total is 1200.", observations, ["calculator"])
        assert rep.verified_claims >= 1
        assert rep.passed is True

    def test_claim_verification_python_syntax(self):
        engine = ThinkEngine()
        valid_code_ans = "Here is the code:\n```python\ndef add(a, b):\n    return a + b\n```"
        rep = engine.verify_claims(valid_code_ans, [], [])
        assert rep.passed is True
        assert rep.has_syntax_error is False

        invalid_code_ans = "Here is broken code:\n```python\ndef broken(:\n    return\n```"
        rep_bad = engine.verify_claims(invalid_code_ans, [], [])
        assert rep_bad.passed is False
        assert rep_bad.has_syntax_error is True

    def test_claim_verification_contradiction_detection(self):
        engine = ThinkEngine()
        observations = [
            {"step": 1, "tool": "calculator", "success": False, "error": "division by zero"},
        ]
        # Claims success despite failure
        rep = engine.verify_claims("I have successfully computed the verified result.", observations, ["calculator"])
        assert rep.has_contradiction is True
        assert rep.passed is False


# ===========================================================================
# 6. End-to-End AgentLoop & Controller Think Engine Tests
# ===========================================================================


class TestThinkEngineIntegration:
    @pytest.fixture
    def mock_deps(self, tmp_path):
        mem = MagicMock()
        mem.get_history.return_value = []
        mem.to_messages.return_value = [{"role": "system", "content": "You are AXEN."}]
        long_mem = MagicMock()
        long_mem.all_facts.return_value = {}
        long_mem.manager.handle_explicit_command.return_value = None
        long_mem.manager.retriever.is_task_relevant.return_value = False
        long_mem.retrieve_relevant.return_value = []
        model = MagicMock()
        model.is_loaded = False
        model.model_name = "test-model"
        registry = build_default_registry()
        return mem, long_mem, model, registry, tmp_path


    def test_agent_loop_runs_with_fast_mode(self, mock_deps):
        mem, long_mem, model, registry, tmp_path = mock_deps
        loop = AgentLoop(
            tools=registry,
            memory=mem,
            long_memory=long_mem,
            model=model,
            workspace_path=str(tmp_path),
        )
        answer, state = loop.run("Hello there", think_mode_override="fast")
        assert state.status == "completed"
        assert state.metadata.get("think_mode") == "fast"
        assert "think_metrics" in state.metadata
        assert state.metadata["think_metrics"]["think_mode"] == "fast"

    def test_agent_loop_runs_with_deep_mode(self, mock_deps):
        mem, long_mem, model, registry, tmp_path = mock_deps
        loop = AgentLoop(
            tools=registry,
            memory=mem,
            long_memory=long_mem,
            model=model,
            workspace_path=str(tmp_path),
        )
        answer, state = loop.run("Calculate 15 * 12.", think_mode_override="deep")
        assert state.status == "completed"
        assert "180" in answer
        assert state.metadata.get("think_mode") == "deep"
        assert "claim_verification" in state.metadata
        assert state.metadata["claim_verification"]["passed"] is True

    def test_controller_passes_think_mode_and_records_telemetry(self, mock_deps):
        mem, long_mem, model, registry, tmp_path = mock_deps
        ctrl = AssistantController(
            model=model,
            tool_registry=registry,
            conversation_memory=mem,
            long_term_memory=long_mem,
            workspace_path=str(tmp_path),
        )
        assert ctrl.think_engine is not None

        reply = ctrl.respond("Calculate 100 + 250.", think_mode="balanced")
        assert "350" in reply
        debug = ctrl.last_debug_info
        assert debug["think_mode"] == "balanced"
        assert "think_metrics" in debug
        assert debug["think_metrics"]["tool_calls_count"] >= 1
        assert "think_budget" in debug


# ===========================================================================
# 7. Edge Cases, Resource Budgets & Graceful Degradation
# ===========================================================================


class TestThinkEngineEdgeCases:
    def test_budget_exhaustion_terminates_loop(self):
        engine = ThinkEngine()
        metrics = ThinkRequestMetrics(think_mode=ThinkMode.FAST)
        # Fast budget max runtime is 10s
        cfg = engine.config.get_mode_config(ThinkMode.FAST)
        cfg.budget.max_runtime_seconds = 0.001  # extremely tight budget
        time.sleep(0.005)

        exceeded, reason = engine.check_budget(metrics, cfg)
        assert exceeded is True
        assert "Time budget exceeded" in reason

    def test_empty_request_handled_safely(self):
        engine = ThinkEngine()
        metrics, cfg = engine.start_request("")
        assert metrics.think_mode is not None
        engine.finish_request(metrics)
        assert metrics.status == "completed"
