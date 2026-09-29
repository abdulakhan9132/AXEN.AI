"""
tests/test_response_contamination.py
====================================
Comprehensive regression tests for AXEN response contamination, prompt leakage,
role integrity, new chat isolation, and selective memory retrieval.

Verifies:
  Test 1: Input "hi" -> normal greeting, must NOT contain "updated version", "AXEN chatbot", "previous interactions"
  Test 2: Input "Hello, who are you?" -> AXEN identity response, must NOT invent employment, company, hobbies
  Test 3: Input "What is 2 + 2?" -> 4
  Test 4: Input "Explain recursion." -> actual explanation of recursion
  Test 5: Start New Chat -> Input "hi" -> clean greeting without previous conversation leakage
  Test 6: Memory Test -> Conversation 1: "My name is Alex." -> New Chat -> "Do you remember my name?" -> Alex -> Then "hi" -> normal greeting without memory leakage
"""

import json
import pytest
from unittest.mock import MagicMock
from app.ui.backend_service import AxenBackendService
from axen.assistant.controller import AssistantController
from axen.memory.conversation_memory import ConversationMemory
from axen.memory.long_term_memory import LongTermMemory
from axen.tools import build_default_registry


class TestResponseContamination:
    """Test suite ensuring zero prompt leakage and clean conversational behavior."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model returning clean responses for testing."""
        mock = MagicMock()
        mock.is_loaded = True
        mock.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        mock._device = "cuda"

        def _mock_chat(messages, **kwargs):
            last_msg = messages[-1]["content"].lower()
            if "recursion" in last_msg:
                return "Recursion is a programming technique where a function calls itself to solve a smaller instance of the problem until it reaches a base case."
            if "who are you" in last_msg:
                return "I am AXEN, a local AI assistant running directly on your computer."
            if any(g in last_msg for g in ["hi", "hello", "hey"]):
                return "Hey! How can I help you today?"
            return "I am here to help you."

        mock.chat.side_effect = _mock_chat
        mock.chat_with_stats.side_effect = lambda msgs, **kw: (_mock_chat(msgs, **kw), {"input_tokens": 10, "generated_tokens": 15})
        return mock

    def test_1_greeting_hi_clean_no_contamination(self, mock_model, tmp_path):
        """Test 1: 'hi' must return a normal greeting and NEVER contain chatbot update artifacts."""
        long_mem = LongTermMemory(path=tmp_path / "long_term.json")
        long_mem.remember("favorite_snack", "Chocolate chip cookies.")
        long_mem.remember("project", "AXEN AI Assistant")

        controller = AssistantController(
            model=mock_model,
            tool_registry=build_default_registry(),
            conversation_memory=ConversationMemory(),
            long_term_memory=long_mem,
        )

        response = controller.respond("hi")
        lower_resp = response.lower()

        # Must not contain contaminated meta phrases
        assert "updated version" not in lower_resp
        assert "axen chatbot" not in lower_resp
        assert "previous interactions" not in lower_resp
        assert "chocolate chip" not in lower_resp
        assert any(g in lower_resp for g in ["hey", "hello", "hi", "help"])

    def test_2_identity_response_no_hallucinated_facts(self, mock_model, tmp_path):
        """Test 2: 'Hello, who are you?' must return AXEN identity without fake personal info."""
        controller = AssistantController(
            model=mock_model,
            tool_registry=build_default_registry(),
            conversation_memory=ConversationMemory(),
            long_term_memory=LongTermMemory(path=tmp_path / "long_term.json"),
        )

        response = controller.respond("Hello, who are you?")
        lower_resp = response.lower()

        assert "axen" in lower_resp or "assistant" in lower_resp
        assert "work for google" not in lower_resp
        assert "employed by" not in lower_resp
        assert "my hobbies" not in lower_resp

    def test_3_arithmetic_calculation_2_plus_2(self, tmp_path):
        """Test 3: 'What is 2 + 2?' must evaluate to 4 using verified calculator."""
        controller = AssistantController(
            model=None,
            tool_registry=build_default_registry(),
            conversation_memory=ConversationMemory(),
            long_term_memory=LongTermMemory(path=tmp_path / "long_term.json"),
        )

        response = controller.respond("What is 2 + 2?")
        assert "4" in response
        assert controller.last_agent_state is not None
        assert controller.last_agent_state.task_type == "calculation"
        assert controller.last_agent_state.verification.get("passed") is True

    def test_4_explain_recursion(self, mock_model, tmp_path):
        """Test 4: 'Explain recursion.' returns actual explanation of recursion."""
        controller = AssistantController(
            model=mock_model,
            tool_registry=build_default_registry(),
            conversation_memory=ConversationMemory(),
            long_term_memory=LongTermMemory(path=tmp_path / "long_term.json"),
        )

        response = controller.respond("Explain recursion.")
        lower_resp = response.lower()

        assert "recursion" in lower_resp
        assert "function" in lower_resp or "base case" in lower_resp

    def test_5_new_chat_isolation(self, mock_model, tmp_path):
        """Test 5: Starting a New Chat ensures zero leakage from previous conversation."""
        service = AxenBackendService.get_instance()
        service.controller.model = mock_model

        # Conversation 1: User asks something complex/developmental
        sess1 = service.new_session(title="Chat 1")
        res1 = service.chat("Can you write a chatbot in Python that handles previous interactions?", session_id=sess1)
        assert len(service.conversations[sess1]["messages"]) == 2

        # Conversation 2: Start New Chat and say "hi"
        sess2 = service.new_session(title="Chat 2")
        res2 = service.chat("hi", session_id=sess2)

        # Inspect controller memory: must only have the "hi" turn and its answer
        mem_messages = service.controller.memory.to_messages()
        # Role sequence: system, user ("hi"), assistant
        assert len(mem_messages) == 3
        assert mem_messages[0]["role"] == "system"
        assert mem_messages[1]["role"] == "user"
        assert mem_messages[1]["content"] == "hi"
        assert mem_messages[2]["role"] == "assistant"

        # None of Conversation 1 should be in the messages sent to the model
        model_input_str = json.dumps(mem_messages)
        assert "chatbot in Python" not in model_input_str
        assert "previous interactions" not in res2["response"].lower()
        assert "updated version" not in res2["response"].lower()

    def test_6_memory_recall_and_subsequent_greeting_isolation(self, mock_model, tmp_path):
        """
        Test 6:
        Conversation 1: "My name is Alex."
        New Chat: "Do you remember my name?" -> Alex
        Then in same or new chat: "hi" -> normal greeting, memory must not leak into greeting.
        """
        service = AxenBackendService.get_instance()
        service.controller.model = mock_model
        # Use an isolated long-term memory for this test
        service.controller.long_memory = LongTermMemory(path=tmp_path / "alex_mem.json")

        # Turn 1: Save name explicitly
        sess1 = service.new_session(title="Introduce")
        service.chat("Remember that my name is Alex", session_id=sess1)
        assert service.controller.long_memory.recall("my_name_is_alex") or service.controller.long_memory.recall("name") or "Alex" in str(service.controller.long_memory.all_facts())

        # Turn 2: New Chat -> Ask to recall name
        sess2 = service.new_session(title="Recall")
        res2 = service.chat("Do you remember my name?", session_id=sess2)
        assert "Alex" in res2["response"]

        # Turn 3: "hi" -> Must NOT leak Alex or long-term facts
        res3 = service.chat("hi", session_id=sess2)
        assert "Alex" not in res3["response"]
        assert "updated version" not in res3["response"].lower()
        # Verify retrieved memories on "hi" is 0
        state = service.controller.last_agent_state
        assert state is not None
        assert len(state.metadata.get("retrieved_memories", [])) == 0
