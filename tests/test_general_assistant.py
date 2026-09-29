"""
tests/test_general_assistant.py
===============================
Comprehensive test suite for AXEN General-Purpose AI Assistant Brain.

Validates Section 18 categories:
1. CONVERSATION: "Hi, how are you?"
2. KNOWLEDGE: "What is an API?"
3. REASONING: "Why does increasing RAM help some applications but not others?"
4. MATH: "What is 238 × 47?"
5. SYMBOLIC MATH: "Differentiate f(x)=(2x^3-5x+4)(x^2+3x-1)"
6. PROGRAMMING: "Write a Python function to reverse a string."
7. DEBUGGING: "Why does this Python code produce an error?"
8. WRITING: "Write a professional leave email."
9. TRANSLATION: "Translate this sentence into Japanese."
10. PLANNING: "Create a 30-day plan to learn Python."
11. MEMORY: "What do you remember about my AXEN project?"
12. FOLLOW-UP: "Make it shorter."
13. MULTI-STEP: "Read this file, calculate the totals, and explain the result."
14. AMBIGUOUS: "Make it better."

Also validates:
- Structured Intent schema matching Section 1 spec
- Adaptive thinking levels (simple, moderate, complex, multi_step)
- Relevance-based memory retrieval (no dumping unrelated facts)
- Multi-tool reasoning & observation propagation
- Error recovery & fallback
- Verification integrity
- Debug mode formatting
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
import pytest

from axen.agent.classifier import TaskClassifier
from axen.agent.context_manager import ContextManager
from axen.agent.intent import IntentInfo
from axen.agent.loop import AgentLoop
from axen.agent.planner import Planner
from axen.agent.state import AgentState
from axen.agent.verifier import Verifier
from axen.assistant.controller import AssistantController
from axen.memory.conversation_memory import ConversationMemory
from axen.memory.long_term_memory import LongTermMemory
from axen.tools import build_default_registry


class MockLoadedModel:
    """Mock language model representing a loaded TinyLlama inference engine."""
    is_loaded = True
    model_name = "MockTinyLlama"
    _device = "cpu"

    def chat(self, messages: list[dict[str, str]]) -> str:
        last_msg = messages[-1]["content"] if messages else ""
        if "API" in last_msg:
            return "An API (Application Programming Interface) allows different software applications to communicate."
        elif "RAM" in last_msg:
            return "Increasing RAM helps applications that are memory-constrained or paging to disk."
        elif "reverse a string" in last_msg:
            return "Here is a Python function:\n```python\ndef reverse_string(s: str) -> str:\n    return s[::-1]\n```"
        elif "leave email" in last_msg:
            return "Subject: Leave Application\n\nDear Manager,\nI am writing to request leave from..."
        elif "Japanese" in last_msg:
            return "In Japanese: こんにちは (Konnichiwa)."
        elif "30-day plan" in last_msg:
            return "30-Day Python Learning Plan:\nWeek 1: Fundamentals\nWeek 2: Data structures\nWeek 3: OOP\nWeek 4: Projects"
        elif "error" in last_msg:
            return "This error occurs because the variable was referenced before assignment."
        elif "Bangalore" in last_msg:
            return "Survival budget for 15 days in Bangalore on ₹3000 (₹200/day): Food ₹140/day, Transport ₹40/day, Buffer ₹20/day."
        elif "portfolio website" in last_msg:
            return "A modern portfolio website structure should include: Hero, About, Projects showcase, Skills, and Contact form."
        return "Helpful response to: " + last_msg

    def chat_with_stats(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        ans = self.chat(messages)
        return ans, {"input_tokens": 12, "generated_tokens": 8, "roles": [m["role"] for m in messages]}


# =======================================================================
# 1. Structured Intent & General Intelligence Router Tests
# =======================================================================

class TestGeneralIntelligenceRouter:
    def setup_method(self):
        self.context_mgr = ContextManager()
        self.classifier = TaskClassifier(context_manager=self.context_mgr)

    def test_structured_intent_schema(self):
        intent = self.classifier.analyze_intent("What is 25 * 48?")
        d = intent.to_dict()
        required_keys = {
            "intent", "requires_reasoning", "requires_tool",
            "requires_memory", "requires_verification", "complexity",
        }
        assert required_keys.issubset(set(d.keys()))
        assert d["intent"] == "calculation"
        assert d["requires_tool"] is True
        assert d["requires_verification"] is True
        assert d["complexity"] == "simple"

    def test_category_1_conversation(self):
        intent = self.classifier.analyze_intent("Hi, how are you?")
        assert intent.intent == "conversation"
        assert intent.requires_tool is False
        assert intent.complexity == "simple"

    def test_category_2_knowledge(self):
        intent = self.classifier.analyze_intent("What is an API?")
        assert intent.intent == "knowledge"
        assert intent.requires_tool is False
        assert intent.requires_reasoning is True

    def test_category_3_reasoning(self):
        intent = self.classifier.analyze_intent("Why does increasing RAM help some applications but not others?")
        assert intent.intent == "reasoning"
        assert intent.requires_tool is False
        assert intent.requires_reasoning is True
        assert intent.complexity == "moderate"

    def test_category_4_math(self):
        intent = self.classifier.analyze_intent("What is 238 * 47?")
        assert intent.intent == "calculation"
        assert intent.requires_tool is True
        assert intent.primary_tool == "calculator"
        assert intent.requires_verification is True

    def test_category_5_symbolic_math(self):
        intent = self.classifier.analyze_intent("Differentiate f(x)=(2x^3-5x+4)(x^2+3x-1)")
        assert intent.intent == "math"
        assert intent.requires_tool is True
        assert intent.primary_tool == "math_solver"
        assert intent.requires_verification is True

    def test_category_6_programming(self):
        intent = self.classifier.analyze_intent("Write a Python function to reverse a string.")
        assert intent.intent == "programming"
        assert intent.requires_reasoning is True
        assert intent.requires_verification is True

    def test_category_7_debugging(self):
        intent = self.classifier.analyze_intent("Why does this Python code produce an error?")
        assert intent.intent == "debugging"
        assert intent.requires_reasoning is True
        assert intent.complexity == "complex"

    def test_category_8_writing(self):
        intent = self.classifier.analyze_intent("Write a professional leave email.")
        assert intent.intent == "writing"
        assert intent.requires_reasoning is True
        assert intent.requires_tool is False

    def test_category_9_translation(self):
        intent = self.classifier.analyze_intent("Translate this sentence into Japanese.")
        assert intent.intent == "translation"
        assert intent.requires_reasoning is True
        assert intent.extracted_data.get("target_language") == "Japanese"

    def test_category_10_planning(self):
        intent1 = self.classifier.analyze_intent("Create a 30-day plan to learn Python.")
        assert intent1.intent == "planning"
        assert intent1.requires_reasoning is True

        intent2 = self.classifier.analyze_intent("I have 3000 rupees and need to survive for 15 days in Bangalore")
        assert intent2.intent == "planning"
        assert intent2.requires_reasoning is True

    def test_category_11_memory(self):
        intent = self.classifier.analyze_intent("What do you remember about my AXEN project?")
        assert intent.intent == "memory"
        assert intent.requires_memory is True
        assert intent.requires_tool is True

    def test_category_12_follow_up(self):
        # With prior history, "Make it shorter" is recognized as a follow-up
        history = [
            {"role": "user", "content": "Explain APIs"},
            {"role": "assistant", "content": "An API is an interface..."},
        ]
        intent = self.classifier.analyze_intent("Make it shorter.", history=history)
        assert intent.intent == "follow_up"
        assert intent.extracted_data.get("action") == "shorten"

    def test_category_13_multi_step(self):
        intent = self.classifier.analyze_intent("Read this file, calculate the totals, and explain the result.")
        assert intent.intent == "multi_step"
        assert intent.requires_tool is True
        assert "file" in intent.tools_needed
        assert "calculator" in intent.tools_needed
        assert intent.complexity == "multi_step"

    def test_category_14_ambiguous_without_context(self):
        # Without context: asks clarification
        intent = self.classifier.analyze_intent("Make it better.", history=[])
        assert intent.intent == "ambiguous"
        assert intent.needs_clarification is True
        assert intent.clarification_prompt is not None
        assert "improve" in intent.clarification_prompt.lower()

    def test_ambiguous_with_context(self):
        # With context: infers active target
        history = [
            {"role": "user", "content": "Write a poem"},
            {"role": "assistant", "content": "The gentle wind sighs..."},
        ]
        intent = self.classifier.analyze_intent("Make it better.", history=history)
        assert intent.intent == "follow_up"
        assert intent.needs_clarification is False


# =======================================================================
# 2. Relevance-Based Memory Tests (Memory Hierarchy)
# =======================================================================

class TestRelevanceBasedMemory:
    def test_relevant_facts_retrieved_and_unrelated_ignored(self):
        mem = LongTermMemory(path=None)
        mem.remember("ui_preference", "I prefer minimal dark UI with sleek typography.")
        mem.remember("favorite_snack", "Chocolate chip cookies.")
        mem.remember("axen_project_goal", "AXEN is an offline, general-purpose local AI.")

        # Query about UI/dashboard -> should retrieve ui_preference, NOT snack
        ui_facts = mem.retrieve_relevant("Design a dashboard for me")
        assert len(ui_facts) > 0
        keys = [f["key"] for f in ui_facts]
        assert "ui_preference" in keys
        assert "favorite_snack" not in keys

        # Query about AXEN project -> should retrieve axen_project_goal
        axen_facts = mem.retrieve_relevant("What is the goal of AXEN?")
        assert len(axen_facts) > 0
        axen_keys = [f["key"] for f in axen_facts]
        assert "axen_project_goal" in axen_keys
        assert "favorite_snack" not in axen_keys


# =======================================================================
# 3. Context & Follow-up Tracking Tests
# =======================================================================

class TestContextAndFollowUps:
    def setup_method(self):
        self.context_mgr = ContextManager()

    def test_pronoun_resolution_python_invention(self):
        history = [
            {"role": "user", "content": "Who invented Python?"},
            {"role": "assistant", "content": "Guido van Rossum invented Python."},
        ]
        res = self.context_mgr.analyze_context("When was it released?", history)
        assert res["is_follow_up"] is True
        assert res["resolved_subject"] == "Python"

    def test_feature_addition_calculator(self):
        history = [
            {"role": "user", "content": "Create a Python calculator."},
            {"role": "assistant", "content": "Here is a Python calculator..."},
        ]
        res = self.context_mgr.analyze_context("Add history.", history)
        assert res["is_follow_up"] is True
        assert "history" in res["resolved_subject"].lower()


# =======================================================================
# 4. Multi-Tool Pipeline & Observation Propagation Tests
# =======================================================================

class TestMultiToolReasoning:
    def setup_method(self):
        self.tools = build_default_registry()
        self.conv_mem = ConversationMemory()
        self.long_mem = LongTermMemory(path=None)
        self.model = MockLoadedModel()
        self.loop = AgentLoop(
            tools=self.tools,
            memory=self.conv_mem,
            long_memory=self.long_mem,
            model=self.model,
        )

    def test_file_read_and_calculate_multi_tool_flow(self):
        # Create temp file with numbers
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt") as tf:
            tf.write("Total Revenue: 150\nOperational Expenses: 50\nNet Savings: 100")
            tf.flush()
            tmp_path = tf.name

        try:
            req = f"Read this file: {tmp_path}, calculate the totals, and explain the result."
            answer, state = self.loop.run(req)

            assert state.task_type == "multi_step"
            assert state.status == "completed"
            assert len(state.tool_calls) >= 2
            # Both file and calculator tools should be called
            tool_names = [call["tool"] for call in state.tool_calls]
            assert "file" in tool_names
            assert "calculator" in tool_names
            # State verification should pass
            assert state.verification is not None
            assert state.verification["passed"] is True
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# =======================================================================
# 5. Error Recovery & Fallbacks Tests
# =======================================================================

class TestErrorRecovery:
    def setup_method(self):
        self.tools = build_default_registry()
        self.conv_mem = ConversationMemory()
        self.long_mem = LongTermMemory(path=None)
        self.model = MockLoadedModel()
        self.loop = AgentLoop(
            tools=self.tools,
            memory=self.conv_mem,
            long_memory=self.long_mem,
            model=self.model,
        )

    def test_complex_derivative_sympy_fallback(self):
        # A complex product derivative
        req = "Differentiate f(x)=(2x^3-5x+4)(x^2+3x-1)"
        answer, state = self.loop.run(req)
        assert state.status == "completed"
        assert state.verification is not None
        assert state.verification["passed"] is True
        # Output should contain terms of the expanded or factored derivative
        assert "x" in answer
        assert "10*x**4" in answer or "2*x" in answer


# =======================================================================
# 6. Syntax Verification Tests
# =======================================================================

class TestCodeSyntaxVerification:
    def test_valid_python_syntax(self):
        verifier = Verifier()
        code = "def reverse_string(s: str) -> str:\n    return s[::-1]\n"
        res = verifier.verify_code_syntax(code, "python")
        assert res["passed"] is True
        assert res["status"] == "passed"

    def test_invalid_python_syntax(self):
        verifier = Verifier()
        code = "def broken(:\n    return"
        res = verifier.verify_code_syntax(code, "python")
        assert res["passed"] is False
        assert res["status"] == "verification_failed"


# =======================================================================
# 7. Debug Mode & Controller Integration Tests
# =======================================================================

class TestDebugModeAndController:
    def setup_method(self):
        self.model = MockLoadedModel()
        self.controller = AssistantController(
            model=self.model,
            tool_registry=build_default_registry(),
            conversation_memory=ConversationMemory(),
            long_term_memory=LongTermMemory(path=None),
        )

    def test_debug_mode_output_format(self):
        self.controller.long_memory.remember("preferred_stack", "Python & React")
        resp = self.controller.respond("What is 238 * 47?", debug=True)
        assert "11186" in resp
        assert "=== AXEN DEBUG ===" in resp
        assert "Intent: calculation" in resp
        assert "Complexity: simple" in resp
        assert "Verification: passed" in resp

    def test_controller_last_debug_info_fields(self):
        self.controller.respond("What is 25 * 48?")
        d = self.controller.last_debug_info
        assert d["intent"] == "calculation"
        assert d["complexity"] == "simple"
        assert d["requires_reasoning"] is False
        assert d["verification"] == "passed"
        assert "calculator" in d["tools"]

    def test_ambiguous_question_returns_clarification(self):
        # Fresh conversation with no context
        self.controller.reset_conversation()
        resp = self.controller.respond("Make it better.")
        assert "What specific aspect" in resp or "improve" in resp
        assert self.controller.last_debug_info["intent"] == "ambiguous"

    def test_follow_up_shorten_after_explanation(self):
        self.controller.reset_conversation()
        resp1 = self.controller.respond("What is an API?")
        assert len(resp1) > 20

        # Follow-up: Make it shorter
        resp2 = self.controller.respond("Make it shorter.")
        assert len(resp2) <= len(resp1)
        assert self.controller.last_debug_info["intent"] == "follow_up"
