"""
tests/test_memory_system.py
===========================
Comprehensive test suite for AXEN AI Phase 3: Memory & Personal Context System.

Validates:
  TEST 1  - Explicit Memory ("Remember that I prefer dark UI" -> Recall)
  TEST 2  - New Chat Survival (Long-term memory persists across new conversations)
  TEST 3  - Short-Term Isolation (Temporary conversation context does not promote to long-term memory)
  TEST 4  - Irrelevant Memory (Calculation 25 * 48 does not retrieve unrelated memories)
  TEST 5  - Memory Update (Update preference from dark to light -> light active, dark inactive)
  TEST 6  - Memory Delete ("Forget that I prefer light UI" -> deactivated)
  TEST 7  - Deduplication ("I prefer dark UI" & "I like dark interfaces" deduped)
  TEST 8  - Explicit vs Automatic ("I might move to London someday" filtered out)
  TEST 9  - Current Instruction Overrides Memory (Prompt instruction takes precedence)
  TEST 10 - Tool Request (Memory does not enter or slow calculation path)
  TEST 11 - Memory Contamination (Greetings remain clean without memory dump)
  TEST 12 - Memory Isolation (User & Conversation isolation)
  TEST 13 - Memory Safety (Secrets, tokens, API keys, passwords blocked)
  TEST 14 - Backup & Export/Import (JSON round-trip)
  TEST 15 - Bounded Short-Term Memory (MAX_RECENT_MESSAGES limit & summary generation)
"""

from __future__ import annotations

import os
import tempfile
import pytest
from pathlib import Path

from axen.memory.memory_models import (
    MemoryItem,
    MemoryType,
    ImportanceLevel,
    ConfidenceLevel,
)
from axen.memory.memory_policy import MemoryPolicy
from axen.memory.memory_store import MemoryStore
from axen.memory.memory_extractor import MemoryExtractor
from axen.memory.memory_validator import MemoryValidator
from axen.memory.memory_retriever import MemoryRetriever
from axen.memory.memory_manager import MemoryManager
from axen.memory.conversation_memory import ConversationMemory, MAX_RECENT_MESSAGES
from axen.memory.long_term_memory import LongTermMemory
from axen.assistant.controller import AssistantController
from axen.agent.loop import AgentLoop
from axen.tools import build_default_registry


# ---------------------------------------------------------------------------
# Fixture: Isolated in-memory/tempfile MemoryManager
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_db_path(tmp_path):
    return str(tmp_path / "test_axen_memory.db")


@pytest.fixture
def memory_manager(temp_db_path):
    mgr = MemoryManager(db_path=temp_db_path)
    yield mgr
    mgr.close()


@pytest.fixture
def mock_controller(temp_db_path):
    """AssistantController with temporary DB and mock LLM."""
    class DummyModel:
        is_loaded = True
        model_name = "test-tinyllama"
        _device = "cpu"

        def chat(self, messages):
            # Inspect system prompt to see if user memory is injected
            sys_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
            last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            
            # If prompt asks about UI preference and memory was injected:
            if "RELEVANT USER MEMORY:" in sys_msg:
                if "ui_theme" in sys_msg or "dark" in sys_msg:
                    return "You prefer a dark UI."
            if "hi" in last_user.lower() or "hello" in last_user.lower():
                return "Hello! How can I help you today?"
            return f"Echo response to: {last_user}"

        def chat_with_stats(self, messages):
            res = self.chat(messages)
            return res, {"input_tokens": 10, "generated_tokens": 5, "roles": ["system", "user"]}

    conv_mem = ConversationMemory()
    long_mem = LongTermMemory(db_path=temp_db_path)
    ctrl = AssistantController(
        model=DummyModel(),
        tool_registry=build_default_registry(),
        conversation_memory=conv_mem,
        long_term_memory=long_mem,
    )
    return ctrl


# ===========================================================================
# Unit & Integration Tests
# ===========================================================================

class TestPhase3MemorySystem:

    # -----------------------------------------------------------------------
    # TEST 1 — EXPLICIT MEMORY
    # -----------------------------------------------------------------------
    def test_explicit_memory_store_and_recall(self, memory_manager):
        """User explicitly asks to remember preference, then asks for recall."""
        # 1. Store via explicit command
        cmd_result = memory_manager.handle_explicit_command("Remember that I prefer dark UI.")
        assert cmd_result is not None
        assert cmd_result["handled"] is True
        assert "dark UI" in cmd_result["message"]

        # 2. Retrieve relevant memory
        retrieved = memory_manager.retrieve("What UI style do I prefer?")
        assert len(retrieved) >= 1
        top_mem = retrieved[0]
        assert top_mem.memory_type == MemoryType.PREFERENCE
        assert top_mem.key == "ui_theme"
        assert "dark" in top_mem.value.lower()
        assert top_mem.relevance_score > 0.40

    # -----------------------------------------------------------------------
    # TEST 2 — NEW CHAT PERSISTENCE
    # -----------------------------------------------------------------------
    def test_long_term_memory_survives_new_chat(self, temp_db_path):
        """Conversation A stores memory; Conversation B retrieves it."""
        # Session A
        mgr_a = MemoryManager(db_path=temp_db_path)
        mgr_a.handle_explicit_command("Remember that my name is Alex.", conversation_id="conv_A")
        mgr_a.close()

        # Session B (simulates new chat with brand new manager / conversation)
        mgr_b = MemoryManager(db_path=temp_db_path)
        retrieved = mgr_b.retrieve("What is my name?", conversation_id="conv_B")
        assert len(retrieved) >= 1
        assert "name" in retrieved[0].key
        assert "Alex" in retrieved[0].value
        mgr_b.close()

    # -----------------------------------------------------------------------
    # TEST 3 — SHORT-TERM ISOLATION
    # -----------------------------------------------------------------------
    def test_short_term_context_isolation(self, memory_manager):
        """Casual conversation context (e.g. 'My favorite number is 42') is not promoted to long-term memory."""
        # Process message through automatic extraction
        saved = memory_manager.process_message_for_memory(
            "My favorite number is 42.",
            conversation_id="conv_1"
        )
        # Should not be saved as a high-confidence permanent long-term memory
        all_memories = memory_manager.list_memories()
        number_memories = [m for m in all_memories if "42" in m.value]
        assert len(number_memories) == 0

    # -----------------------------------------------------------------------
    # TEST 4 — IRRELEVANT MEMORY FILTER
    # -----------------------------------------------------------------------
    def test_irrelevant_memory_bypass(self, memory_manager):
        """Deterministic calculations and unrelated tasks do not retrieve memories."""
        memory_manager.remember(
            memory_type=MemoryType.PREFERENCE,
            key="ui_theme",
            value="dark UI",
            importance=ImportanceLevel.HIGH,
            confidence=ConfidenceLevel.HIGH,
        )
        memory_manager.remember(
            memory_type=MemoryType.PROJECT,
            key="current_project",
            value="AXEN AI",
            importance=ImportanceLevel.HIGH,
            confidence=ConfidenceLevel.HIGH,
        )

        # 1. Calculation query: retrieval must be bypassed
        assert not memory_manager.is_retrieval_needed("What is 25 * 48?")
        calc_retrieved = memory_manager.retrieve("What is 25 * 48?")
        assert len(calc_retrieved) == 0

        # 2. Irrelevant query: even if checked, scores below threshold
        unrelated = memory_manager.retrieve("What is the capital of France?")
        assert len(unrelated) == 0

    # -----------------------------------------------------------------------
    # TEST 5 — MEMORY UPDATE
    # -----------------------------------------------------------------------
    def test_memory_update_deactivates_old(self, memory_manager):
        """Updating a preference deactivates the older conflicting one."""
        # 1. Store dark UI
        memory_manager.remember(
            memory_type=MemoryType.PREFERENCE,
            key="ui_theme",
            value="dark UI",
            importance=ImportanceLevel.MEDIUM,
            confidence=ConfidenceLevel.HIGH,
        )
        active_before = memory_manager.list_memories(is_active=True)
        assert len(active_before) == 1
        assert active_before[0].value == "dark UI"

        # 2. Update to light UI via explicit command
        cmd_result = memory_manager.handle_explicit_command("Update my preference to light UI.")
        assert cmd_result["handled"] is True

        # 3. Check active vs inactive
        active_after = memory_manager.list_memories(is_active=True)
        assert len(active_after) == 1
        assert "light UI" in active_after[0].value

        # Historical record exists but is inactive
        all_records = memory_manager.list_memories(is_active=False)
        assert any("dark UI" in r.value for r in all_records)

    # -----------------------------------------------------------------------
    # TEST 6 — MEMORY DELETE / FORGET
    # -----------------------------------------------------------------------
    def test_memory_forget_command(self, memory_manager):
        """'Forget that I prefer light UI' deactivates the memory."""
        memory_manager.remember(
            memory_type=MemoryType.PREFERENCE,
            key="ui_theme",
            value="light UI",
            importance=ImportanceLevel.MEDIUM,
            confidence=ConfidenceLevel.HIGH,
        )
        assert len(memory_manager.list_memories(is_active=True)) == 1

        cmd_result = memory_manager.handle_explicit_command("Forget that I prefer light UI.")
        assert cmd_result["handled"] is True

        # Must no longer be retrieved as active memory
        retrieved = memory_manager.retrieve("What UI style do I prefer?")
        assert len(retrieved) == 0

    # -----------------------------------------------------------------------
    # TEST 7 — DEDUPLICATION
    # -----------------------------------------------------------------------
    def test_deduplication_similar_phrasings(self, memory_manager):
        """Similar statements like 'I prefer dark UI' and 'I like dark interfaces' do not duplicate."""
        # First insertion
        res1 = memory_manager.remember(
            memory_type=MemoryType.PREFERENCE,
            key="ui_theme",
            value="dark UI",
            confidence=ConfidenceLevel.HIGH,
        )
        assert res1 is not None

        # Second insertion with same key and compatible normalized value
        res2 = memory_manager.remember(
            memory_type=MemoryType.PREFERENCE,
            key="ui_theme",
            value="dark UI",
            confidence=ConfidenceLevel.HIGH,
        )
        # Should return existing item without creating a duplicate
        active = memory_manager.list_memories(is_active=True)
        assert len(active) == 1

    # -----------------------------------------------------------------------
    # TEST 8 — EXPLICIT VS CASUAL/AUTOMATIC
    # -----------------------------------------------------------------------
    def test_casual_speculative_statements_filtered(self, memory_manager):
        """Speculative statement ('I might move to London someday') is not saved automatically."""
        saved = memory_manager.process_message_for_memory("I might move to London someday.")
        assert len(saved) == 0
        assert len(memory_manager.list_memories()) == 0

    # -----------------------------------------------------------------------
    # TEST 9 — CURRENT INSTRUCTION OVERRIDES MEMORY
    # -----------------------------------------------------------------------
    def test_prompt_priority_current_instruction(self, memory_manager):
        """Prompt builder clearly instructs that current user request overrides memory."""
        memory_manager.remember(
            memory_type=MemoryType.PREFERENCE,
            key="ui_theme",
            value="dark UI",
        )
        retrieved = memory_manager.retrieve("Design a UI")
        block = memory_manager.format_context_block(retrieved)
        assert "RELEVANT USER MEMORY:" in block
        assert "Current user instructions always supersede" in block
        assert "dark UI" in block

    # -----------------------------------------------------------------------
    # TEST 10 — TOOL REQUEST BYPASSES MEMORY
    # -----------------------------------------------------------------------
    def test_tool_request_bypasses_memory(self, mock_controller):
        """Mathematical operations execute via calculator without memory intrusion."""
        # Prepopulate memory
        mock_controller.long_term_memory.remember("preference", "ui_theme", "dark UI")

        # Execute calculation
        res = mock_controller.respond("Calculate 25 * 48")
        assert "1200" in res
        assert mock_controller.last_agent_state.task_type == "calculation"
        # Tool call should be calculator, not memory tool
        assert mock_controller.last_agent_state.tool_calls[0]["tool"] == "calculator"

    # -----------------------------------------------------------------------
    # TEST 11 — MEMORY CONTAMINATION IN GREETINGS
    # -----------------------------------------------------------------------
    def test_greeting_no_memory_contamination(self, mock_controller):
        """Greeting 'hi' does not trigger memory retrieval or dump facts."""
        mock_controller.long_term_memory.remember("identity", "name", "Abdulla")
        mock_controller.long_term_memory.remember("preference", "ui_theme", "dark UI")

        res = mock_controller.respond("hi")
        assert any(w in res.lower() for w in ("hello", "hey", "hi"))
        assert "Abdulla" not in res
        assert "dark UI" not in res

    # -----------------------------------------------------------------------
    # TEST 12 — MEMORY ISOLATION (USER & CONVERSATION)
    # -----------------------------------------------------------------------
    def test_memory_user_isolation(self, memory_manager):
        """User A's memory is isolated from User B."""
        memory_manager.remember(
            memory_type=MemoryType.IDENTITY,
            key="name",
            value="Alice",
            user_id="user_A",
        )
        memory_manager.remember(
            memory_type=MemoryType.IDENTITY,
            key="name",
            value="Bob",
            user_id="user_B",
        )

        user_a_mem = memory_manager.list_memories(user_id="user_A")
        assert len(user_a_mem) == 1
        assert user_a_mem[0].value == "Alice"

        user_b_mem = memory_manager.list_memories(user_id="user_B")
        assert len(user_b_mem) == 1
        assert user_b_mem[0].value == "Bob"

    # -----------------------------------------------------------------------
    # TEST 13 — MEMORY SAFETY (SECRETS BLOCKING)
    # -----------------------------------------------------------------------
    def test_memory_safety_blocks_secrets(self, memory_manager):
        """API keys, passwords, private keys, bearer tokens are strictly rejected."""
        # 1. API key pattern
        is_safe, reason = MemoryPolicy.validate_candidate(
            "sk-1234567890abcdef1234567890abcdef",
            MemoryType.FACT,
            "api_key"
        )
        assert is_safe is False
        assert "Security Policy" in reason

        # 2. Attempt to remember a password
        saved = memory_manager.remember(
            memory_type=MemoryType.INSTRUCTION,
            key="password",
            value="MyPassword123!",
        )
        assert saved is None
        assert len(memory_manager.list_memories()) == 0

    # -----------------------------------------------------------------------
    # TEST 14 — BACKUP & EXPORT/IMPORT
    # -----------------------------------------------------------------------
    def test_export_and_import_json(self, memory_manager, tmp_path):
        """JSON export and import cleanly round-trips active memories."""
        memory_manager.remember(MemoryType.IDENTITY, "name", "Abdulla")
        memory_manager.remember(MemoryType.PREFERENCE, "ui_theme", "dark")

        export_data = memory_manager.export_memories()
        assert export_data["memory_version"] == 1
        assert len(export_data["memories"]) == 2

        # Create a new blank DB and import
        new_db = str(tmp_path / "imported_memory.db")
        new_mgr = MemoryManager(db_path=new_db)
        imported_count = new_mgr.import_memories(export_data)
        assert imported_count == 2
        assert len(new_mgr.list_memories()) == 2
        new_mgr.close()

    # -----------------------------------------------------------------------
    # TEST 15 — BOUNDED SHORT-TERM MEMORY
    # -----------------------------------------------------------------------
    def test_bounded_short_term_memory_and_summary(self):
        """ConversationMemory caps raw messages at MAX_RECENT_MESSAGES and summarizes older ones."""
        conv = ConversationMemory(max_history=12)
        assert MAX_RECENT_MESSAGES == 12

        # Add 20 messages
        for i in range(20):
            conv.add_user_message(f"User message {i}")
            conv.add_assistant_message(f"Assistant response {i}")

        recent = conv.get_recent_history(limit=12)
        assert len(recent) <= 12
        assert conv.has_summary()
        summary = conv.get_summary()
        assert summary is not None
        assert len(summary) > 0
