"""
tests/test_axen_phase3.py
=========================
Comprehensive tests for AXEN Phase 3 — Agentic Reasoning Loop.

Validates:
  1. AgentState data structure and serialization
  2. Task classification for all 10 task types
  3. Tool decision logic (tool needed vs. no tool)
  4. Structured tool calls & results
  5. Deterministic tool execution (Calculator, MathSolver, Python, File)
  6. Observation capture
  7. Deterministic verification and failure handling
  8. Bounded retry logic (max 3 attempts)
  9. Step limit enforcement (max 8 steps)
  10. Multi-step mixed task planning and execution
  11. Controller integration and debug metadata
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
import pytest

from axen.agent.classifier import TaskClassifier
from axen.agent.loop import AgentLoop, MAX_STEPS, MAX_TOOL_ATTEMPTS
from axen.agent.planner import Planner
from axen.agent.state import AgentState
from axen.agent.verifier import Verifier
from axen.assistant.controller import AssistantController
from axen.memory.conversation_memory import ConversationMemory
from axen.memory.long_term_memory import LongTermMemory
from axen.tools import ToolRegistry, build_default_registry


# =======================================================================
# 1. AgentState Tests
# =======================================================================

class TestAgentState:
    def test_state_initialization_defaults(self):
        state = AgentState(user_request="Hello")
        assert state.user_request == "Hello"
        assert state.task_type == "conversation"
        assert state.status == "pending"
        assert state.plan == []
        assert state.current_step == 0
        assert state.tool_calls == []
        assert state.tool_results == []
        assert state.verification is None
        assert state.final_answer is None

    def test_state_serialization_matches_spec(self):
        state = AgentState(user_request="What is 25 * 48?")
        state.task_type = "calculation"
        state.status = "completed"
        state.plan = [{"step": 1, "tool": "calculator", "description": "calculate"}]
        state.current_step = 1
        state.add_tool_call("calculator", {"expression": "25 * 48"})
        state.add_tool_result("calculator", "1200", success=True)
        state.verification = {"passed": True, "status": "passed"}
        state.final_answer = "The answer is 1200."

        d = state.to_dict()
        expected_keys = {
            "user_request", "task_type", "status", "plan",
            "current_step", "tool_calls", "tool_results",
            "verification", "final_answer",
        }
        assert set(d.keys()) == expected_keys
        assert d["user_request"] == "What is 25 * 48?"
        assert d["task_type"] == "calculation"
        assert d["status"] == "completed"
        assert d["tool_calls"] == [{"tool": "calculator", "arguments": {"expression": "25 * 48"}}]
        assert d["tool_results"] == [{"tool": "calculator", "result": "1200", "success": True}]
        assert d["verification"]["passed"] is True
        assert d["final_answer"] == "The answer is 1200."

    def test_state_tool_result_failure(self):
        state = AgentState(user_request="25 / 0")
        state.add_tool_result("calculator", None, success=False, error="Division by zero")
        assert state.tool_results[0]["success"] is False
        assert state.tool_results[0]["error"] == "Division by zero"


# =======================================================================
# 2. Task Classification Tests (All 10 Task Types)
# =======================================================================

class TestTaskClassifier:
    def setup_method(self):
        self.classifier = TaskClassifier()

    def test_classify_conversation(self):
        ttype, need_tool, tool = self.classifier.classify("Hello AXEN")
        assert ttype == "conversation"
        assert need_tool is False
        assert tool is None

        ttype, need_tool, _ = self.classifier.classify("Good morning!")
        assert ttype == "conversation"
        assert need_tool is False

    def test_classify_calculation(self):
        ttype, need_tool, tool = self.classifier.classify("What is 25 * 48?")
        assert ttype == "calculation"
        assert need_tool is True
        assert tool == "calculator"

        ttype, need_tool, tool = self.classifier.classify("Calculate 12345 * 678")
        assert ttype == "calculation"
        assert need_tool is True
        assert tool == "calculator"

    def test_classify_math(self):
        ttype, need_tool, tool = self.classifier.classify("Differentiate x^3 + 2x")
        assert ttype == "math"
        assert need_tool is True
        assert tool == "math_solver"

        ttype, need_tool, tool = self.classifier.classify("What is the gcd of 48 and 18?")
        assert ttype == "math"
        assert need_tool is True
        assert tool == "math_solver"

    def test_classify_programming(self):
        ttype, need_tool, tool = self.classifier.classify("Explain Python recursion")
        assert ttype == "programming"
        assert need_tool is False
        assert tool is None

        ttype, need_tool, tool = self.classifier.classify("How does an algorithm work?")
        assert ttype == "programming"
        assert need_tool is False

    def test_classify_code_execution(self):
        ttype, need_tool, tool = self.classifier.classify("Run this Python code: print(2 + 3)")
        assert ttype == "code_execution"
        assert need_tool is True
        assert tool == "python"

        ttype, need_tool, tool = self.classifier.classify("print(2 + 3)")
        assert ttype == "code_execution"
        assert need_tool is True
        assert tool == "python"

    def test_classify_file_operation(self):
        ttype, need_tool, tool = self.classifier.classify("Read this file: data.txt")
        assert ttype == "file_operation"
        assert need_tool is True
        assert tool == "file"

    def test_classify_memory(self):
        ttype, need_tool, tool = self.classifier.classify("Remember that my name is Alice")
        assert ttype == "memory"
        assert need_tool is True
        assert tool == "memory"

        ttype, need_tool, tool = self.classifier.classify("Recall what you remember")
        assert ttype == "memory"
        assert need_tool is True

    def test_classify_knowledge_retrieval(self):
        ttype, need_tool, tool = self.classifier.classify("What is Python programming?")
        assert ttype == "knowledge_retrieval"
        assert need_tool is False
        assert tool is None

    def test_classify_mixed(self):
        ttype, need_tool, tool = self.classifier.classify("Read this file: data.txt and calculate the average")
        assert ttype == "mixed"
        assert need_tool is True
        assert tool == "multi_step"

    def test_classify_unknown(self):
        ttype, need_tool, tool = self.classifier.classify("??? !!! ###")
        assert ttype == "unknown"
        assert need_tool is False
        assert tool is None

        ttype, need_tool, tool = self.classifier.classify("")
        assert ttype == "unknown"
        assert need_tool is False


# =======================================================================
# 3. Planner Tests
# =======================================================================

class TestPlanner:
    def setup_method(self):
        self.planner = Planner()

    def test_plan_simple_conversation(self):
        plan = self.planner.create_plan("conversation", "Hello", None)
        assert len(plan) == 1
        assert plan[0]["tool"] is None
        assert plan[0]["step"] == 1

    def test_plan_calculator(self):
        plan = self.planner.create_plan("calculation", "What is 25 * 48?", "calculator")
        assert len(plan) == 1
        assert plan[0]["tool"] == "calculator"
        assert "25 * 48" in plan[0]["arguments"]["raw_input"]

    def test_plan_math_solver(self):
        plan = self.planner.create_plan("math", "Differentiate x^3 + 2x", "math_solver")
        assert len(plan) == 1
        assert plan[0]["tool"] == "math_solver"
        assert "Differentiate" in plan[0]["arguments"]["problem"]

    def test_plan_mixed_task(self):
        req = "Read this file: test.csv and calculate the total"
        plan = self.planner.create_plan("mixed", req, "multi_step")
        assert len(plan) >= 2
        tools_in_plan = [s["tool"] for s in plan]
        assert "file" in tools_in_plan
        assert "calculator" in tools_in_plan


# =======================================================================
# 4. Verifier Tests
# =======================================================================

class TestVerifier:
    def setup_method(self):
        self.verifier = Verifier()

    def test_verify_calculator_correct(self):
        args = {"expression": "25 * 48"}
        res = {"success": True, "result": 1200, "display": "1200", "expression": "25 * 48"}
        verif = self.verifier.verify_tool_result("calculator", args, res)
        assert verif["passed"] is True
        assert verif["status"] == "passed"

    def test_verify_calculator_mismatch_fails(self):
        args = {"expression": "25 * 48"}
        res = {"success": True, "result": 9999, "display": "9999", "expression": "25 * 48"}
        verif = self.verifier.verify_tool_result("calculator", args, res)
        assert verif["passed"] is False
        assert verif["status"] == "verification_failed"

    def test_verify_math_solver_flag(self):
        res_ok = {"success": True, "verified": True, "verification": "verified symbolically"}
        verif_ok = self.verifier.verify_tool_result("math_solver", {}, res_ok)
        assert verif_ok["passed"] is True

        res_bad = {"success": True, "verified": False, "verification": "cannot prove"}
        verif_bad = self.verifier.verify_tool_result("math_solver", {}, res_bad)
        assert verif_bad["passed"] is False
        assert verif_bad["status"] == "verification_failed"

    def test_verify_python_execution(self):
        res_ok = {"success": True, "exit_code": 0, "stdout": "5\n", "stderr": ""}
        verif_ok = self.verifier.verify_tool_result("python", {"code": "print(2+3)"}, res_ok)
        assert verif_ok["passed"] is True

        res_fail = {"success": True, "exit_code": 1, "stdout": "", "stderr": "Error"}
        verif_fail = self.verifier.verify_tool_result("python", {"code": "raise Error"}, res_fail)
        assert verif_fail["passed"] is False
        assert verif_fail["status"] == "verification_failed"


# =======================================================================
# 5. AgentLoop Tests (Execution, Observation, Limits, Retries)
# =======================================================================

class TestAgentLoop:
    def setup_method(self):
        self.tools = build_default_registry()
        self.conv_mem = ConversationMemory()
        self.long_mem = LongTermMemory(path=None)

        class DummyModel:
            is_loaded = False
            model_name = "test-model"
            def chat(self, messages):
                return "Mock LLM response"
            def chat_with_stats(self, messages):
                return "Mock LLM response", {"input_tokens": 10, "generated_tokens": 5, "roles": ["user"]}

        self.model = DummyModel()
        self.loop = AgentLoop(
            tools=self.tools,
            memory=self.conv_mem,
            long_memory=self.long_mem,
            model=self.model,
        )

    def test_loop_constants(self):
        assert MAX_STEPS == 8
        assert MAX_TOOL_ATTEMPTS == 3

    def test_calculator_execution_and_observation(self):
        answer, state = self.loop.run("What is 25 * 48?")
        assert state.task_type == "calculation"
        assert state.status == "completed"
        assert state.verification is not None
        assert state.verification["passed"] is True
        assert "1200" in answer
        assert len(state.tool_calls) == 1
        assert state.tool_calls[0]["tool"] == "calculator"
        assert state.tool_results[0]["result"] == "1200"
        assert state.tool_results[0]["success"] is True

    def test_math_solver_execution_and_observation(self):
        answer, state = self.loop.run("Differentiate x^3 + 2x")
        assert state.task_type == "math"
        assert state.status == "completed"
        assert state.verification is not None
        assert state.verification["passed"] is True
        assert "3*x**2 + 2" in answer
        assert len(state.tool_calls) == 1
        assert state.tool_calls[0]["tool"] == "math_solver"

    def test_python_tool_execution(self):
        answer, state = self.loop.run("Run this Python code: print(2 + 3)")
        assert state.task_type == "code_execution"
        assert state.status == "completed"
        assert state.verification["passed"] is True
        assert "5" in answer
        assert len(state.tool_calls) == 1
        assert state.tool_calls[0]["tool"] == "python"

    def test_deliberate_invalid_calculation_fails_gracefully(self):
        answer, state = self.loop.run("What is 25 / 0?")
        assert state.task_type == "calculation"
        assert state.status == "verification_failed"
        assert "Division by zero" in answer or "Error" in answer
        # Never fabricate a fake numeric answer
        assert state.final_answer.startswith("Error:")

    def test_no_tool_for_concepts(self):
        # Model is not loaded, but tool should not be called
        answer, state = self.loop.run("Explain Python recursion")
        assert state.task_type == "programming"
        assert len(state.tool_calls) == 0
        assert "My local language model isn't loaded yet" in answer

    def test_memory_store_and_recall_loop(self):
        # Store
        ans_store, state_store = self.loop.run("Remember that my project is AXEN")
        assert state_store.task_type == "memory"
        assert state_store.status == "completed"
        assert "AXEN" in ans_store

        # Recall
        ans_recall, state_recall = self.loop.run("Recall notes")
        assert state_recall.task_type == "memory"
        assert "my_project" in ans_recall or "AXEN" in ans_recall

    def test_file_read_loop(self):
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt") as tf:
            tf.write("Hello AXEN File Tool!")
            tf.flush()
            temp_path = tf.name

        try:
            answer, state = self.loop.run(f"Read this file: {temp_path}")
            assert state.task_type == "file_operation"
            assert state.status == "completed"
            assert "Hello AXEN File Tool!" in answer
            assert state.verification["passed"] is True
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_bounded_retry_recoverable_failure(self):
        call_count = 0
        def flaky_run(args, req, state):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return None, False, "transient network glitch", None
            return "recovered_data", True, None, {"passed": True, "status": "passed"}

        self.loop._run_file = flaky_run
        obs, ok, err, verif = self.loop._execute_with_retry("file", {}, "read file test.txt", AgentState("test"))
        assert ok is True
        assert obs == "recovered_data"
        assert call_count == 2

    def test_bounded_retry_stops_at_max_attempts(self):
        call_count = 0
        def permanent_fail(args, req, state):
            nonlocal call_count
            call_count += 1
            return None, False, "transient error that never resolves", None

        self.loop._run_file = permanent_fail
        obs, ok, err, verif = self.loop._execute_with_retry("file", {}, "read file test.txt", AgentState("test"))
        assert ok is False
        assert call_count == MAX_TOOL_ATTEMPTS

    def test_max_steps_limit(self):
        # Construct a synthetic 12-step plan
        plan = [{"step": i, "tool": "calculator", "arguments": {"raw_input": "1 + 1"}} for i in range(1, 13)]
        self.loop.planner.create_plan = lambda *args: plan

        ans, state = self.loop.run("What is 1 + 1?")
        # Must not exceed MAX_STEPS (8)
        assert state.current_step == MAX_STEPS



# =======================================================================
# 6. AssistantController Integration Tests
# =======================================================================

class TestAssistantControllerPhase3:
    def setup_method(self):
        class DummyModel:
            is_loaded = False
            model_name = "test-model"
            _device = "cpu"
            def chat(self, messages):
                return "Mock conversation"
            def chat_with_stats(self, messages):
                return "Mock conversation", {"input_tokens": 5, "generated_tokens": 4, "roles": ["user"]}

        self.controller = AssistantController(
            model=DummyModel(),
            tool_registry=build_default_registry(),
            conversation_memory=ConversationMemory(),
            long_term_memory=LongTermMemory(path=None),
        )

    def test_respond_calculation(self):
        res = self.controller.respond("What is 25 * 48?")
        assert res == "The answer is 1200."
        assert self.controller.last_agent_state is not None
        assert self.controller.last_agent_state.task_type == "calculation"
        assert self.controller.last_agent_state.status == "completed"

        dbg = self.controller.last_debug_info
        assert dbg["task_type"] == "calculation"
        assert dbg["selected_tool"] == "calculator"
        assert dbg["tool_decision"] == "tool_required"
        assert dbg["verification"] == "passed"
        assert dbg["final_status"] == "completed"

    def test_respond_symbolic_math(self):
        res = self.controller.respond("Differentiate x^3 + 2x")
        assert "3*x**2 + 2" in res
        assert self.controller.last_debug_info["task_type"] == "math"
        assert self.controller.last_debug_info["selected_tool"] == "math_solver"
        assert self.controller.last_debug_info["verification"] == "passed"

    def test_respond_conversation_no_model(self):
        res = self.controller.respond("Hello!")
        assert "My local language model isn't loaded yet" in res
        assert self.controller.last_debug_info["task_type"] == "conversation"
        assert self.controller.last_debug_info["tool_decision"] == "no_tool"

    def test_respond_conversation_with_model(self):
        self.controller.model.is_loaded = True
        res = self.controller.respond("Hello!")
        assert res == "Mock conversation"
        assert self.controller.last_debug_info["task_type"] == "conversation"
        assert self.controller.last_debug_info["tool_decision"] == "no_tool"
        assert self.controller.last_debug_info["final_status"] == "completed"
