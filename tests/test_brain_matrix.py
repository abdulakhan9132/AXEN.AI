"""
tests/test_brain_matrix.py
==========================
Comprehensive Brain Test Matrix for AXEN AI Phase 2 Intelligence Core.

Covers all 12 Categories:
1. Conversation: "hi", "hello", "who are you?", "how are you?"
2. Calculator: "2+2", "25*48", "100/4"
3. Symbolic Math: "f(x)=(2x^3-5x+4)(x^2+3x-1)", "Differentiate x^3 + 2x", "Simplify (x+1)^2"
4. Computer Systems: "Convert 255 to binary", "Convert 101101 to decimal", "What is 0b1010 AND 0b1100?"
5. Programming: "Generate Python code to print Hello World."
6. Execution: "Run Python: print(2+3)"
7. Follow-up: "Convert 255 to binary." -> "Now convert that to hexadecimal."
8. Multi-step: "Calculate the derivative and simplify it."
9. Mixed: "Calculate 25*48 and explain the calculation."
10. Ambiguous: "Open it."
11. Tool failure: Bounded recovery
12. Verification failure: Truthful verification status
"""

import pytest
from unittest.mock import MagicMock

from axen.assistant.controller import AssistantController
from axen.memory.conversation_memory import ConversationMemory
from axen.memory.long_term_memory import LongTermMemory
from axen.tools import build_default_registry


class TestBrainMatrix:
    @pytest.fixture
    def mock_model(self):
        mock = MagicMock()
        mock.is_loaded = True
        mock.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        mock._device = "cuda"

        def _chat(messages, **kwargs):
            last_msg = messages[-1]["content"].lower()
            if "who are you" in last_msg:
                return "I am AXEN, an AI assistant running locally on your computer."
            if "hello world" in last_msg:
                return "```python\nprint('Hello World')\n```"
            if "multiplication works" in last_msg or "explain the calculation" in last_msg or "explain" in last_msg:
                return "25 * 48 = 1,200. You can break it down as 25 * 40 + 25 * 8 = 1000 + 200 = 1200."
            return "Here is the response from AXEN."

        mock.chat.side_effect = _chat
        mock.chat_with_stats.side_effect = lambda msgs, **kw: (_chat(msgs, **kw), {"input_tokens": 10, "generated_tokens": 15})
        return mock

    @pytest.fixture
    def controller(self, mock_model, tmp_path):
        return AssistantController(
            model=mock_model,
            tool_registry=build_default_registry(),
            conversation_memory=ConversationMemory(),
            long_term_memory=LongTermMemory(path=tmp_path / "long_term.json"),
        )

    # -----------------------------------------------------------------------
    # CATEGORY 1 — CONVERSATION
    # -----------------------------------------------------------------------

    def test_cat1_greetings_and_identity(self, controller):
        for greeting in ["hi", "hello", "how are you?"]:
            resp = controller.respond(greeting)
            lower = resp.lower()
            assert any(g in lower for g in ["hey", "hello", "hi", "help", "doing well"])
            assert controller.last_agent_state.task_type in ("conversation", "greeting")
            assert len(controller.last_agent_state.tool_calls) == 0

        # Identity
        id_resp = controller.respond("who are you?")
        assert "axen" in id_resp.lower() or "assistant" in id_resp.lower()
        assert len(controller.last_agent_state.tool_calls) == 0

    # -----------------------------------------------------------------------
    # CATEGORY 2 — CALCULATOR
    # -----------------------------------------------------------------------

    def test_cat2_calculator(self, controller):
        cases = [
            ("2+2", "4"),
            ("25*48", "1200"),
            ("100/4", "25"),
        ]
        for query, expected in cases:
            resp = controller.respond(query)
            assert expected in resp
            state = controller.last_agent_state
            assert state.task_type == "calculation"
            assert state.verification is not None
            assert state.verification.get("passed") is True
            assert state.verification.get("status") in ("VERIFIED", "passed")

    # -----------------------------------------------------------------------
    # CATEGORY 3 — SYMBOLIC MATH
    # -----------------------------------------------------------------------

    def test_cat3_symbolic_math(self, controller):
        # 1. f(x) polynomial expression
        resp1 = controller.respond("f(x)=(2x^3-5x+4)(x^2+3x-1)")
        assert controller.last_agent_state.task_type == "math"
        assert controller.last_debug_info["selected_tool"] == "math_solver"
        # Must NOT be routed to calculator
        assert controller.last_debug_info["selected_tool"] != "calculator"
        assert "2*x**5" in resp1 or "2x^5" in resp1 or "x**4" in resp1

        # 2. Differentiate
        resp2 = controller.respond("Differentiate x^3 + 2x")
        assert "3*x**2+2" in resp2.replace(" ", "") or "3*x**2 + 2" in resp2
        assert controller.last_agent_state.task_type == "math"
        assert controller.last_agent_state.verification.get("status") in ("VERIFIED", "passed")

        # 3. Simplify
        resp3 = controller.respond("Simplify (x+1)^2")
        assert "x**2+2*x+1" in resp3.replace(" ", "") or "x**2 + 2*x + 1" in resp3
        assert controller.last_agent_state.task_type == "math"
        assert controller.last_agent_state.verification.get("status") in ("VERIFIED", "passed")

    # -----------------------------------------------------------------------
    # CATEGORY 4 — COMPUTER SYSTEMS
    # -----------------------------------------------------------------------

    def test_cat4_computer_systems(self, controller):
        # 1. Convert 255 to binary
        res1 = controller.respond("Convert 255 to binary")
        assert "11111111" in res1
        assert controller.last_agent_state.task_type == "computer_systems"

        # 2. Convert 101101 to decimal
        res2 = controller.respond("Convert 101101 to decimal")
        assert "45" in res2
        assert controller.last_agent_state.task_type == "computer_systems"

        # 3. Bitwise AND
        res3 = controller.respond("What is 0b1010 AND 0b1100?")
        assert "1000" in res3
        assert controller.last_agent_state.task_type == "computer_systems"

    # -----------------------------------------------------------------------
    # CATEGORY 5 — PROGRAMMING
    # -----------------------------------------------------------------------

    def test_cat5_programming(self, controller):
        res = controller.respond("Generate Python code to print Hello World.")
        assert "print" in res
        assert "Hello World" in res
        assert controller.last_agent_state.task_type == "programming"
        assert len(controller.last_agent_state.tool_calls) == 0

    # -----------------------------------------------------------------------
    # CATEGORY 6 — EXECUTION
    # -----------------------------------------------------------------------

    def test_cat6_execution(self, controller):
        res = controller.respond("Run Python: print(2+3)")
        assert "5" in res
        assert controller.last_agent_state.task_type == "code_execution"
        assert controller.last_debug_info["selected_tool"] == "python"
        assert controller.last_agent_state.verification.get("status") in ("VERIFIED", "passed")

    # -----------------------------------------------------------------------
    # CATEGORY 7 — FOLLOW-UP CONTEXT
    # -----------------------------------------------------------------------

    def test_cat7_follow_up(self, controller):
        # First turn
        controller.respond("Convert 255 to binary")
        assert controller.agent_loop.context_manager.last_numeric_result == "11111111"

        # Second turn resolving "that"
        res2 = controller.respond("Now convert that to hexadecimal.")
        assert "FF" in res2.upper() or "0XFF" in res2.upper() or "255" in res2
        assert controller.last_agent_state.task_type == "computer_systems"

    # -----------------------------------------------------------------------
    # CATEGORY 8 — MULTI-STEP
    # -----------------------------------------------------------------------

    def test_cat8_multi_step_derivative_and_simplify(self, controller):
        res = controller.respond("Calculate the derivative of (x+1)^3 and simplify it.")
        assert "3*(x + 1)**2" in res or "3*x**2 + 6*x + 3" in res or "3*(x+1)^2" in res
        # State plan should have multiple steps
        assert len(controller.last_agent_state.plan) >= 2
        for step in controller.last_agent_state.plan:
            assert "step_id" in step
            assert "action" in step
            assert "status" in step

    # -----------------------------------------------------------------------
    # CATEGORY 9 — MIXED INTENT
    # -----------------------------------------------------------------------

    def test_cat9_mixed_intent(self, controller):
        res = controller.respond("Calculate 25*48 and explain the calculation.")
        assert "1200" in res or "1,200" in res
        assert controller.last_agent_state.task_type in ("mixed", "multi_step", "calculation")
        # Tool was called for arithmetic
        assert any(call["tool"] == "calculator" for call in controller.last_agent_state.tool_calls)

    # -----------------------------------------------------------------------
    # CATEGORY 10 — AMBIGUOUS
    # -----------------------------------------------------------------------

    def test_cat10_ambiguous(self, controller):
        controller.reset_conversation()
        res = controller.respond("Open it.")
        assert any(w in res.lower() for w in ["open", "specify", "what", "which", "clarify"])
        assert controller.last_agent_state.task_type in ("ambiguous", "clarification")

    # -----------------------------------------------------------------------
    # CATEGORY 11 — TOOL FAILURE & BOUNDED RECOVERY
    # -----------------------------------------------------------------------

    def test_cat11_tool_failure_bounded_recovery(self, controller):
        # Deliberate syntax/arithmetic failure: 25 / 0
        res = controller.respond("What is 25 / 0?")
        assert "Error" in res or "zero" in res.lower() or "reliably" in res.lower()
        assert controller.last_agent_state.status == "verification_failed"

    # -----------------------------------------------------------------------
    # CATEGORY 12 — VERIFICATION FAILURE
    # -----------------------------------------------------------------------

    def test_cat12_verification_failure_truthful(self, controller):
        # Deliberate failure should never claim verification
        controller.respond("What is 100 / 0?")
        verif = controller.last_agent_state.verification
        if verif:
            assert verif.get("passed") is False
            assert verif.get("status") in ("FAILED", "verification_failed")
