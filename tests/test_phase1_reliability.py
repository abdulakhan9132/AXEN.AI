"""
tests/test_phase1_reliability.py
================================
Comprehensive Phase 1 test suite covering:
  A. Greeting: 'hi' -> short greeting, no unrelated context
  B. Identity: 'Who are you?' -> AXEN identity, no fictional personal history
  C. Calculator: '25 * 48' -> 1200
  D. Math: Differentiate f(x)=(2x^3-5x+4)(x^2+3x-1) -> 10x^4 + 18x^3 - 21x^2 - 4x + 17
  E. Programming: 'Generate Python code to print Hello World.' -> valid Python code
  F. Memory isolation: Conversation A stores a task -> New Chat -> 'hi' -> clean
  G. Memory relevance: Stored 'I prefer dark UI.' -> 'What UI style do I prefer?' -> dark UI
  H. Greeting contamination: hi, hello, hey, good morning in sequence -> clean
  I. Thinking states: real state progression (UNDERSTANDING -> ... -> COMPLETED)
  J. Cancellation: cancel_event stops agent and marks cancelled
"""

import json
import threading
import time
import pytest
from unittest.mock import MagicMock

from app.ui.backend_service import AxenBackendService
from axen.assistant.controller import AssistantController
from axen.memory.conversation_memory import ConversationMemory
from axen.memory.long_term_memory import LongTermMemory
from axen.tools import build_default_registry


class TestPhase1Reliability:
    @pytest.fixture
    def mock_model(self):
        mock = MagicMock()
        mock.is_loaded = True
        mock.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        mock._device = "cuda"

        def _chat(messages, **kwargs):
            last_msg = messages[-1]["content"].lower()
            all_text = " ".join(m.get("content", "").lower() for m in messages)
            if "who are you" in last_msg:
                return "I am AXEN, an AI assistant running locally on your computer."
            if "hello world" in last_msg:
                return "```python\nprint('Hello World')\n```"
            if "dark" in all_text and ("prefer" in last_msg or "style" in last_msg or "ui" in last_msg):
                return "You prefer a dark UI with sleek typography."
            if any(g in last_msg for g in ["hi", "hello", "hey", "good morning"]):
                return "Hey! How can I help you today?"
            return "I am here to help you."

        mock.chat.side_effect = _chat
        mock.chat_with_stats.side_effect = lambda msgs, **kw: (_chat(msgs, **kw), {"input_tokens": 12, "generated_tokens": 16})
        return mock

    def test_a_greeting(self, mock_model, tmp_path):
        """A. Greeting: 'hi' -> short greeting, no unrelated context."""
        controller = AssistantController(
            model=mock_model,
            tool_registry=build_default_registry(),
            conversation_memory=ConversationMemory(),
            long_term_memory=LongTermMemory(path=tmp_path / "long.json"),
        )
        resp = controller.respond("hi")
        lower = resp.lower()
        assert any(g in lower for g in ["hey", "hello", "hi", "help"])
        assert "updated version" not in lower
        assert "chatbot" not in lower
        assert len(resp) < 150

    def test_b_identity(self, mock_model, tmp_path):
        """B. Identity: 'Who are you?' -> AXEN identity, no fictional personal history."""
        controller = AssistantController(
            model=mock_model,
            tool_registry=build_default_registry(),
            conversation_memory=ConversationMemory(),
            long_term_memory=LongTermMemory(path=tmp_path / "long.json"),
        )
        resp = controller.respond("Who are you?")
        lower = resp.lower()
        assert "axen" in lower or "assistant" in lower
        assert "work for google" not in lower
        assert "employed by" not in lower
        assert "my hobbies" not in lower

    def test_c_calculator(self, tmp_path):
        """C. Calculator: '25 * 48' -> 1200 deterministic."""
        controller = AssistantController(
            model=None,
            tool_registry=build_default_registry(),
            conversation_memory=ConversationMemory(),
            long_term_memory=LongTermMemory(path=tmp_path / "long.json"),
        )
        resp = controller.respond("25 * 48")
        assert "1200" in resp
        state = controller.last_agent_state
        assert state.task_type == "calculation"
        assert state.verification.get("passed") is True

    def test_d_math_differentiation(self, tmp_path):
        """D. Math: Differentiate f(x)=(2x^3-5x+4)(x^2+3x-1)."""
        from reasoning.math.solver import MathSolver
        solver = MathSolver()
        res = solver.solve("differentiate (2*x**3 - 5*x + 4)*(x**2 + 3*x - 1)")
        assert res.verified is True
        ans = str(res.answer).replace(" ", "")
        # Expected derivative: 10*x**4 + 24*x**3 - 21*x**2 - 22*x + 17
        assert "10*x**4" in ans
        assert "-21*x**2" in ans

    def test_e_programming_hello_world(self, mock_model, tmp_path):
        """E. Programming: 'Generate Python code to print Hello World.'"""
        controller = AssistantController(
            model=mock_model,
            tool_registry=build_default_registry(),
            conversation_memory=ConversationMemory(),
            long_term_memory=LongTermMemory(path=tmp_path / "long.json"),
        )
        resp = controller.respond("Generate Python code to print Hello World.")
        assert "print" in resp
        assert "Hello World" in resp

    def test_f_memory_isolation(self, mock_model, tmp_path):
        """F. Memory isolation: Conversation A stores a temporary task. New Chat. 'hi' -> no continuation."""
        service = AxenBackendService.get_instance()
        service.controller.model = mock_model

        sess_a = service.new_session(title="Task A")
        service.chat("Write Python code for a calculator with memory and square root", session_id=sess_a)

        # New chat
        sess_b = service.new_session(title="Chat B")
        res_b = service.chat("hi", session_id=sess_b)

        assert "square root" not in res_b["response"].lower()
        assert "calculator" not in res_b["response"].lower()
        assert any(g in res_b["response"].lower() for g in ["hey", "hello", "hi", "help"])

    def test_g_memory_relevance(self, mock_model, tmp_path):
        """G. Memory relevance: Stored 'I prefer dark UI.' -> Ask: 'What UI style do I prefer?' -> dark UI."""
        service = AxenBackendService.get_instance()
        service.controller.model = mock_model
        service.controller.long_memory = LongTermMemory(path=tmp_path / "ui_pref.json")

        sess = service.new_session(title="Store Pref")
        service.chat("Remember that I prefer dark UI with sleek typography", session_id=sess)

        # Inquire about UI preference
        sess2 = service.new_session(title="Ask Pref")
        res = service.chat("What UI style do I prefer?", session_id=sess2)
        assert "dark" in res["response"].lower() or "sleek" in res["response"].lower()

    def test_h_greeting_contamination_multiple_runs(self, mock_model, tmp_path):
        """H. Greeting contamination: Run hi, hello, hey, good morning multiple times -> all clean."""
        controller = AssistantController(
            model=mock_model,
            tool_registry=build_default_registry(),
            conversation_memory=ConversationMemory(),
            long_term_memory=LongTermMemory(path=tmp_path / "long.json"),
        )
        greetings = ["hi", "hello", "hey", "good morning"]
        for g in greetings:
            controller.memory.clear()
            resp = controller.respond(g)
            lower = resp.lower()
            assert any(term in lower for term in ["hey", "hello", "hi", "help", "good morning"])
            assert "updated version" not in lower
            assert "previous interactions" not in lower

    def test_i_thinking_states_progression(self, tmp_path):
        """I. Thinking states: Verify clean progression from UNDERSTANDING to COMPLETED."""
        states_emitted = []
        def _cb(st_name, disp, meta):
            states_emitted.append(st_name)

        controller = AssistantController(
            model=None,
            tool_registry=build_default_registry(),
            conversation_memory=ConversationMemory(),
            long_term_memory=LongTermMemory(path=tmp_path / "long.json"),
        )
        resp = controller.respond("25 * 48", state_callback=_cb)
        assert "1200" in resp
        assert "UNDERSTANDING" in states_emitted
        assert "USING_TOOL" in states_emitted
        assert "VERIFYING" in states_emitted
        assert "COMPLETED" in states_emitted
        # Ensure COMPLETED is the terminal state
        assert states_emitted[-1] == "COMPLETED"

    def test_j_cancellation(self, tmp_path):
        """J. Cancellation: cancel_event stops agent and marks cancelled."""
        cancel_event = threading.Event()
        cancel_event.set()  # Pre-cancel

        controller = AssistantController(
            model=None,
            tool_registry=build_default_registry(),
            conversation_memory=ConversationMemory(),
            long_term_memory=LongTermMemory(path=tmp_path / "long.json"),
        )
        resp = controller.respond("What is 123 * 456?", cancel_event=cancel_event)
        assert "cancelled" in resp.lower()
        assert controller.last_agent_state.status == "cancelled"
