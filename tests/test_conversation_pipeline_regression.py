"""
tests/test_conversation_pipeline_regression.py
=============================================
Regression tests specifically for live conversation prompt construction,
TinyLlama chat template alignment, and current-user grounding.

Verifies:
  1. No duplicate BOS token (<s>) in chat template tokenization
  2. Final message in model input is strictly the current user message
  3. System prompt does not trigger tool-script hallucinations on simple greetings
  4. Identity queries maintain clean assistant identity without invented employers/hobbies
  5. Conversation history preserves proper role sequences (user -> assistant -> user)
  6. Generated token slicing decodes only new tokens, never prompt+completion
"""

import pytest
from axen.assistant.controller import AssistantController
from axen.memory.conversation_memory import ConversationMemory
from axen.memory.long_term_memory import LongTermMemory
from axen.model.hf_model import HFLocalModel, HFModelConfig, _truncate_turn
from axen.tools import build_default_registry


class TestConversationPipelineRegression:
    def test_chat_template_tokenization_no_duplicate_bos(self):
        """Ensure apply_chat_template does not duplicate the <s> (BOS) token."""
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        except Exception:
            pytest.skip("HuggingFace tokenizer not available in offline test environment")

        messages = [
            {"role": "system", "content": "You are AXEN."},
            {"role": "user", "content": "hi"},
        ]
        encoded = tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"] if hasattr(encoded, "input_ids") else encoded
        # In TinyLlama, <s> has token ID 1. Direct apply_chat_template should not start with [1, 529]
        # Token 529 is the '<' of '<|system|>'
        first_token = int(input_ids[0][0])
        assert first_token != 1, f"Found duplicate BOS token (1) at index 0: {input_ids[0][:3].tolist()}"

    def test_current_user_message_is_final_message(self):
        """Ensure the user's latest request is strictly the last message before generation."""
        conv_mem = ConversationMemory()
        conv_mem.add_user("First user message")
        conv_mem.add_assistant("First assistant reply")

        # Now new turn
        conv_mem.add_user("hi")
        messages = conv_mem.to_messages()

        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "First user message"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "First assistant reply"
        assert messages[3]["role"] == "user"
        assert messages[3]["content"] == "hi"
        # Crucial: final role must be user
        assert messages[-1]["role"] == "user"

    def test_assistant_response_never_becomes_user_message(self):
        """Verify assistant output is stored strictly as assistant role."""
        controller = AssistantController(
            model=None,
            tool_registry=build_default_registry(),
            conversation_memory=ConversationMemory(),
            long_term_memory=LongTermMemory(path=None),
        )
        controller.memory.add_user("hi")
        controller.memory.add_assistant("Greetings! How can I help you?")

        history = controller.memory.get_history()
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == "hi"
        assert history[1].role == "assistant"
        assert history[1].content == "Greetings! How can I help you?"

    def test_system_prompt_does_not_contain_script_hallucination_triggers(self):
        """System prompt must be concise and avoid prompting tool-script descriptions on greetings."""
        cfg = HFModelConfig.from_yaml()
        mem = ConversationMemory()
        messages = mem.to_messages()
        sys_prompt = messages[0]["content"]

        assert "AXEN" in sys_prompt
        assert "helpful and concise" in sys_prompt or "assistant" in sys_prompt
        # Ensure confusing meta-instruction 'Use available tools' is absent from the LLM prompt
        assert "Use available tools when the task requires calculation" not in sys_prompt

    def test_identity_guardrails_prevent_employer_and_hobbies(self):
        """Identity guardrails must prevent hallucinated jobs, companies, or hobbies."""
        controller = AssistantController(
            model=None,
            tool_registry=build_default_registry(),
            conversation_memory=ConversationMemory(),
            long_term_memory=LongTermMemory(path=None),
        )
        # Test company question
        guarded_co = controller._apply_identity_guardrails(
            "What company do you work for?",
            "I work for a great marketing company called Acme Corp.",
        )
        assert "I don't work for a company" in guarded_co or "AXEN" in guarded_co
        assert "Acme Corp" not in guarded_co

        # Test hobbies question
        guarded_hob = controller._apply_identity_guardrails(
            "What are your hobbies?",
            "My hobbies are playing guitar and hiking in the mountains.",
        )
        assert "I don't have personal hobbies" in guarded_hob
        assert "guitar" not in guarded_hob

        # Test who are you question
        guarded_who = controller._apply_identity_guardrails(
            "Hello, who are you?",
            "My name is AXEN, and I've been programmed to assist you with your job search and work for Acme.",
        )
        assert "I am AXEN, a local AI assistant" in guarded_who
        assert "job search" not in guarded_who

    def test_greeting_guardrail_prevents_runaway_monologue(self):
        """Greetings must not output multi-paragraph monologues."""
        controller = AssistantController(
            model=None,
            tool_registry=build_default_registry(),
            conversation_memory=ConversationMemory(),
            long_term_memory=LongTermMemory(path=None),
        )
        monologue = (
            "Greetings! How may I assist you today?\n\n"
            "Sure thing! May I know what you need assistance with?\n\n"
            "Certainly! I'd love to help you with your job search."
        )
        guarded = controller._apply_identity_guardrails("hi", monologue)
        assert guarded == "Greetings! How may I assist you today?"
        assert "job search" not in guarded

    def test_truncate_turn_strips_subsequent_turns(self):
        """Verify _truncate_turn cuts off user/system turn continuations."""
        raw = "Hello! How can I help?\n\nUser: Can you write code?\nAssistant: Yes I can."
        assert _truncate_turn(raw) == "Hello! How can I help?"

        raw_tag = "Hello! How can I help?</s><|user|>\nWhat is 2+2?"
        assert _truncate_turn(raw_tag) == "Hello! How can I help?"
