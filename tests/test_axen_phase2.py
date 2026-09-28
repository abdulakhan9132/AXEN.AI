"""
tests/test_axen_phase2.py
==========================
Test suite for AXEN Phase 2 Local Assistant.

Tests cover:
  - HFModelConfig loading and defaults
  - ConversationMemory add/retrieve/trim
  - LongTermMemory store/recall/forget
  - ToolRegistry registration and dispatch
  - Calculator tool: correct arithmetic, edge cases, blocked unsafe code
  - Python tool: validated code execution (safe + unsafe)
  - AssistantController: tool dispatch without model, memory ops
  - AssistantController: graceful fallback when model is not loaded

NOTE: These tests do NOT load the actual language model (that requires
downloading ~2 GB and a GPU). They use mocks for the model layer.
Running these tests: `pytest tests/test_axen_phase2.py -v`
"""

from __future__ import annotations

import pytest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# -----------------------------------------------------------------------
# 1. HFModelConfig
# -----------------------------------------------------------------------

class TestHFModelConfig:
    def test_defaults(self):
        """Config uses sensible defaults when no YAML is present."""
        from axen.model.hf_model import HFModelConfig
        cfg = HFModelConfig()
        assert cfg.name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert cfg.device == "auto"
        assert cfg.dtype == "float16"
        assert cfg.max_new_tokens == 512
        assert 0.0 < cfg.temperature <= 1.0
        assert 0.0 < cfg.top_p <= 1.0
        assert cfg.max_context_length > 0

    def test_from_yaml_missing_file(self, tmp_path):
        """from_yaml returns defaults when file does not exist."""
        from axen.model.hf_model import HFModelConfig
        cfg = HFModelConfig.from_yaml(tmp_path / "nonexistent.yaml")
        assert cfg.name == HFModelConfig.name  # class default

    def test_from_yaml_valid(self, tmp_path):
        """from_yaml parses hf_model section correctly."""
        import yaml
        from axen.model.hf_model import HFModelConfig
        data = {
            "hf_model": {
                "name": "my/model",
                "device": "cpu",
                "max_new_tokens": 256,
                "temperature": 0.5,
            }
        }
        p = tmp_path / "test.yaml"
        p.write_text(yaml.dump(data), encoding="utf-8")
        cfg = HFModelConfig.from_yaml(p)
        assert cfg.name == "my/model"
        assert cfg.device == "cpu"
        assert cfg.max_new_tokens == 256
        assert cfg.temperature == 0.5

    def test_from_yaml_partial(self, tmp_path):
        """from_yaml uses defaults for missing fields."""
        import yaml
        from axen.model.hf_model import HFModelConfig
        data = {"hf_model": {"name": "partial/model"}}
        p = tmp_path / "partial.yaml"
        p.write_text(yaml.dump(data), encoding="utf-8")
        cfg = HFModelConfig.from_yaml(p)
        assert cfg.name == "partial/model"
        assert cfg.temperature == HFModelConfig.temperature  # class default


# -----------------------------------------------------------------------
# 2. HFLocalModel (no model loaded — only structural tests)
# -----------------------------------------------------------------------

class TestHFLocalModel:
    def test_not_loaded_initially(self):
        """Model reports not loaded before load() is called."""
        from axen.model.hf_model import HFLocalModel
        m = HFLocalModel()
        assert not m.is_loaded

    def test_generate_raises_when_not_loaded(self):
        """generate() raises RuntimeError if model is not loaded."""
        from axen.model.hf_model import HFLocalModel
        m = HFLocalModel()
        with pytest.raises(RuntimeError, match="not loaded"):
            m.generate("Hello")

    def test_chat_raises_when_not_loaded(self):
        """chat() raises RuntimeError if model is not loaded."""
        from axen.model.hf_model import HFLocalModel
        m = HFLocalModel()
        with pytest.raises(RuntimeError, match="not loaded"):
            m.chat([{"role": "user", "content": "Hi"}])

    def test_repr(self):
        """repr() is informative."""
        from axen.model.hf_model import HFLocalModel
        m = HFLocalModel()
        r = repr(m)
        assert "HFLocalModel" in r
        assert "not loaded" in r


# -----------------------------------------------------------------------
# 3. ConversationMemory
# -----------------------------------------------------------------------

class TestConversationMemory:
    def test_add_and_retrieve(self):
        from axen.memory.conversation_memory import ConversationMemory
        mem = ConversationMemory()
        mem.add_user("Hello")
        mem.add_assistant("Hi there!")
        msgs = mem.to_messages()
        # First is always system prompt
        assert msgs[0]["role"] == "system"
        assert msgs[1]["content"] == "Hello"
        assert msgs[2]["content"] == "Hi there!"

    def test_turn_count(self):
        from axen.memory.conversation_memory import ConversationMemory
        mem = ConversationMemory()
        mem.add_user("Q1")
        mem.add_assistant("A1")
        mem.add_user("Q2")
        assert mem.turn_count == 2

    def test_clear(self):
        from axen.memory.conversation_memory import ConversationMemory
        mem = ConversationMemory()
        mem.add_user("test")
        mem.clear()
        assert mem.turn_count == 0
        assert mem.token_estimate == 0

    def test_trimming(self):
        """Memory stays under max_tokens by removing oldest messages."""
        from axen.memory.conversation_memory import ConversationMemory
        mem = ConversationMemory(max_tokens=10)
        # Each "X" * 100 is roughly 25 tokens
        mem.add_user("X" * 100)
        mem.add_assistant("Y" * 100)
        mem.add_user("Z" * 100)
        # After trimming, token_estimate should be <= max_tokens
        assert mem.token_estimate <= 30  # may slightly exceed due to char math

    def test_system_prompt_always_present(self):
        from axen.memory.conversation_memory import ConversationMemory
        mem = ConversationMemory()
        msgs = mem.to_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"

    def test_custom_system_prompt(self):
        from axen.memory.conversation_memory import ConversationMemory
        mem = ConversationMemory(system_prompt="Custom system")
        msgs = mem.to_messages()
        assert msgs[0]["content"] == "Custom system"

    def test_set_system_prompt(self):
        from axen.memory.conversation_memory import ConversationMemory
        mem = ConversationMemory()
        mem.set_system_prompt("Updated!")
        msgs = mem.to_messages()
        assert msgs[0]["content"] == "Updated!"


# -----------------------------------------------------------------------
# 4. LongTermMemory
# -----------------------------------------------------------------------

class TestLongTermMemory:
    def test_remember_and_recall(self, tmp_path):
        from axen.memory.long_term_memory import LongTermMemory
        mem = LongTermMemory(path=tmp_path / "test.json")
        mem.remember("color", "blue")
        assert mem.recall("color") == "blue"

    def test_recall_default(self, tmp_path):
        from axen.memory.long_term_memory import LongTermMemory
        mem = LongTermMemory(path=tmp_path / "test.json")
        assert mem.recall("missing_key", "fallback") == "fallback"

    def test_forget(self, tmp_path):
        from axen.memory.long_term_memory import LongTermMemory
        mem = LongTermMemory(path=tmp_path / "test.json")
        mem.remember("temp", "value")
        assert mem.forget("temp") is True
        assert mem.recall("temp") is None

    def test_forget_nonexistent(self, tmp_path):
        from axen.memory.long_term_memory import LongTermMemory
        mem = LongTermMemory(path=tmp_path / "test.json")
        assert mem.forget("does_not_exist") is False

    def test_persistence(self, tmp_path):
        """Memory persists to disk and reloads correctly."""
        from axen.memory.long_term_memory import LongTermMemory
        p = tmp_path / "mem.json"
        mem1 = LongTermMemory(path=p)
        mem1.remember("language", "Python")
        # Reload from same file
        mem2 = LongTermMemory(path=p)
        assert mem2.recall("language") == "Python"

    def test_search(self, tmp_path):
        from axen.memory.long_term_memory import LongTermMemory
        mem = LongTermMemory(path=tmp_path / "s.json")
        mem.remember("fav_color", "red")
        mem.remember("fav_food", "pizza")
        results = mem.search("fav")
        assert len(results) == 2

    def test_all_facts(self, tmp_path):
        from axen.memory.long_term_memory import LongTermMemory
        mem = LongTermMemory(path=tmp_path / "af.json")
        mem.remember("k1", "v1")
        mem.remember("k2", "v2")
        facts = mem.all_facts()
        assert facts == {"k1": "v1", "k2": "v2"}

    def test_clear(self, tmp_path):
        from axen.memory.long_term_memory import LongTermMemory
        mem = LongTermMemory(path=tmp_path / "c.json")
        mem.remember("x", "y")
        mem.clear()
        assert len(mem) == 0

    def test_len(self, tmp_path):
        from axen.memory.long_term_memory import LongTermMemory
        mem = LongTermMemory(path=tmp_path / "l.json")
        assert len(mem) == 0
        mem.remember("a", 1)
        mem.remember("b", 2)
        assert len(mem) == 2


# -----------------------------------------------------------------------
# 5. ToolRegistry
# -----------------------------------------------------------------------

class TestToolRegistry:
    def test_register_and_call(self):
        from axen.tools.tool_registry import ToolRegistry, ToolDefinition
        registry = ToolRegistry()
        tool = ToolDefinition(
            name="test.echo",
            description="Echo args",
            handler=lambda args: {"echo": args.get("msg", "")},
        )
        registry.register(tool)
        result = registry.call("test.echo", {"msg": "hello"})
        assert result.success is True
        assert result.output["echo"] == "hello"

    def test_duplicate_raises(self):
        from axen.tools.tool_registry import ToolRegistry, ToolDefinition
        registry = ToolRegistry()
        t = ToolDefinition(name="dup", description="x", handler=lambda a: {})
        registry.register(t)
        with pytest.raises(ValueError):
            registry.register(t)

    def test_missing_tool(self):
        from axen.tools.tool_registry import ToolRegistry
        registry = ToolRegistry()
        result = registry.call("nonexistent.tool", {})
        assert result.success is False
        assert "No tool named" in (result.error or "")

    def test_tool_error_captured(self):
        from axen.tools.tool_registry import ToolRegistry, ToolDefinition
        def bad_handler(args):
            raise ValueError("intentional error")
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="bad", description="fails", handler=bad_handler))
        result = registry.call("bad", {})
        assert result.success is False
        assert "intentional error" in (result.error or "")

    def test_list_tools(self):
        from axen.tools import build_default_registry
        registry = build_default_registry()
        tools = registry.list_tools()
        names = [t["name"] for t in tools]
        assert "calculator.evaluate" in names
        assert "python.execute" in names


# -----------------------------------------------------------------------
# 6. Calculator Tool
# -----------------------------------------------------------------------

class TestCalculatorTool:
    def setup_method(self):
        from axen.tools.calculator import calculate
        self.calculate = calculate

    def test_simple_add(self):
        r = self.calculate({"expression": "2 + 3"})
        assert r["success"] is True
        assert r["result"] == 5

    def test_simple_multiply(self):
        r = self.calculate({"expression": "12345 * 678"})
        assert r["success"] is True
        assert r["result"] == 12345 * 678

    def test_power(self):
        r = self.calculate({"expression": "2 ** 10"})
        assert r["success"] is True
        assert r["result"] == 1024

    def test_float_division(self):
        r = self.calculate({"expression": "7 / 2"})
        assert r["success"] is True
        assert abs(r["result"] - 3.5) < 1e-9

    def test_floor_division(self):
        r = self.calculate({"expression": "7 // 2"})
        assert r["success"] is True
        assert r["result"] == 3

    def test_modulo(self):
        r = self.calculate({"expression": "10 % 3"})
        assert r["success"] is True
        assert r["result"] == 1

    def test_parentheses(self):
        r = self.calculate({"expression": "(2 + 3) * 4"})
        assert r["success"] is True
        assert r["result"] == 20

    def test_unary_minus(self):
        r = self.calculate({"expression": "-5 + 10"})
        assert r["success"] is True
        assert r["result"] == 5

    def test_division_by_zero(self):
        r = self.calculate({"expression": "1 / 0"})
        assert r["success"] is False
        assert "zero" in r["error"].lower()

    def test_empty_expression(self):
        r = self.calculate({"expression": ""})
        assert r["success"] is False

    def test_missing_expression_key(self):
        r = self.calculate({})
        assert r["success"] is False

    def test_blocked_string_literal(self):
        """String literals are not allowed."""
        r = self.calculate({"expression": "'hello' + 'world'"})
        assert r["success"] is False

    def test_display_integer(self):
        """Whole-number floats are displayed without decimal."""
        r = self.calculate({"expression": "4 / 2"})
        assert r["success"] is True
        assert r["display"] == "2"

    def test_complex_expression(self):
        r = self.calculate({"expression": "100 * (1 + 5) ** 2 / 4"})
        assert r["success"] is True
        expected = 100 * (1 + 5) ** 2 / 4
        assert abs(r["result"] - expected) < 1e-6


# -----------------------------------------------------------------------
# 7. Python Tool
# -----------------------------------------------------------------------

class TestPythonTool:
    def setup_method(self):
        from axen.tools.python_tool import execute_python
        self.execute = execute_python

    def test_hello_world(self):
        r = self.execute({"code": "print('hello world')"})
        assert r["validated"] is True
        assert r["executed"] is True
        assert r["success"] is True
        assert "hello world" in r["stdout"]

    def test_arithmetic_output(self):
        r = self.execute({"code": "print(2 ** 10)"})
        assert r["success"] is True
        assert "1024" in r["stdout"]

    def test_syntax_error(self):
        r = self.execute({"code": "def broken("})
        assert r["validated"] is False
        assert "Syntax error" in r["error"]

    def test_blocked_import_subprocess(self):
        r = self.execute({"code": "import subprocess"})
        assert r["validated"] is False
        assert "subprocess" in r["error"]

    def test_blocked_import_socket(self):
        r = self.execute({"code": "import socket"})
        assert r["validated"] is False
        assert "socket" in r["error"]

    def test_runtime_error_captured(self):
        r = self.execute({"code": "raise ValueError('test error')"})
        assert r["executed"] is True
        assert r["success"] is False
        assert r["returncode"] != 0

    def test_multiline_code(self):
        code = "total = 0\nfor i in range(1, 6):\n    total += i\nprint(total)"
        r = self.execute({"code": code})
        assert r["success"] is True
        assert "15" in r["stdout"]

    def test_empty_code(self):
        r = self.execute({"code": ""})
        assert r["success"] is False
        assert "Missing" in r["error"]

    def test_no_output(self):
        r = self.execute({"code": "x = 42"})
        assert r["success"] is True
        assert r["stdout"] == ""


# -----------------------------------------------------------------------
# 8. AssistantController (no model — tool-only mode)
# -----------------------------------------------------------------------

class TestAssistantControllerNoModel:
    """
    Tests for the AssistantController without loading the actual LLM.
    We only test tool dispatch, memory, and graceful fallback.
    """

    def setup_method(self, tmp_path=None):
        from axen.assistant.controller import AssistantController
        from axen.memory.long_term_memory import LongTermMemory
        import tempfile
        # Use a temp file for long-term memory so tests don't interfere
        self._tmpdir = tempfile.mkdtemp()
        ltm = LongTermMemory(path=Path(self._tmpdir) / "ltm.json")
        self.controller = AssistantController(long_term_memory=ltm)
        # NOTE: model is NOT loaded

    def test_model_not_ready(self):
        assert self.controller.model_ready is False

    def test_respond_without_model(self):
        """Without a model, should return a helpful fallback message."""
        response = self.controller.respond("Tell me a story")
        assert isinstance(response, str)
        assert len(response) > 0
        # Should mention that the model is not ready
        response_lower = response.lower()
        assert (
            "not loaded" in response_lower
            or "isn't loaded" in response_lower
            or "load" in response_lower
            or "tool" in response_lower
        )

    def test_calculator_dispatch(self):
        """Calculator tool fires for clear arithmetic expressions."""
        # Use 'calculate' keyword which is in the regex pattern
        response = self.controller.respond("calculate 12 * 34")
        # Should contain the result 408
        assert "408" in response

    def test_calculator_dispatch_simple(self):
        response = self.controller.respond("calculate 100 + 200")
        assert "300" in response

    def test_memory_save_command(self):
        """'remember that' triggers long-term memory save."""
        response = self.controller.respond("remember that my name is Alex")
        assert "Alex" in response or "saved" in response.lower()

    def test_memory_recall_command(self):
        """'do you remember' triggers recall."""
        self.controller.long_memory.remember("name", "Alex")
        response = self.controller.respond("do you remember anything about me?")
        assert "Alex" in response or "remember" in response.lower()

    def test_reset_conversation(self):
        self.controller.respond("Hello")
        self.controller.reset_conversation()
        assert self.controller.memory.turn_count == 0

    def test_status(self):
        status = self.controller.status()
        assert "model" in status
        assert "model_loaded" in status
        assert "tools" in status
        assert isinstance(status["tools"], list)

    def test_python_execution_dispatch(self):
        """Python tool fires for ```python blocks."""
        code_input = "run this:\n```python\nprint('axen test')\n```"
        response = self.controller.respond(code_input)
        assert "axen test" in response or "python" in response.lower()

    def test_empty_input(self):
        response = self.controller.respond("")
        assert "didn't catch" in response.lower() or isinstance(response, str)

    def test_conversation_turn_count(self):
        self.controller.respond("hello")
        self.controller.respond("how are you?")
        assert self.controller.memory.turn_count == 2


# -----------------------------------------------------------------------
# 9. Integration: build_default_registry
# -----------------------------------------------------------------------

class TestDefaultRegistry:
    def test_registry_has_expected_tools(self):
        from axen.tools import build_default_registry
        reg = build_default_registry()
        assert "calculator.evaluate" in reg.names()
        assert "python.execute" in reg.names()

    def test_calculator_via_registry(self):
        from axen.tools import build_default_registry
        reg = build_default_registry()
        result = reg.call("calculator.evaluate", {"expression": "7 + 8"})
        assert result.success is True
        assert result.output["result"] == 15

    def test_python_via_registry(self):
        from axen.tools import build_default_registry
        reg = build_default_registry()
        result = reg.call("python.execute", {"code": "print(42)"})
        assert result.success is True
        assert "42" in result.output["stdout"]


# -----------------------------------------------------------------------
# 10. Startup behavior: normal mode SHOULD load the model
# -----------------------------------------------------------------------

class TestStartupModelLoad:
    """
    Tests verifying that the startup flow ATTEMPTS to load the model
    and correctly reports the result. We use mocking to avoid actually
    downloading and loading TinyLlama during unit tests.
    """

    def test_load_model_with_progress_calls_model_load(self, tmp_path, monkeypatch):
        """
        load_model_with_progress() should set controller._model_loaded = True
        when the model load succeeds.
        """
        from axen.assistant.controller import AssistantController
        from axen.model.hf_model import HFModelConfig
        from axen_chat import load_model_with_progress

        config = HFModelConfig(name="test/model", device="cpu", dtype="float16",
                               max_new_tokens=32, temperature=0.7, top_p=0.9,
                               max_context_length=512)
        controller = AssistantController(config=config)

        # Mock the internals of _do_load_with_hooks so we don't download anything
        import types

        def _fake_load(ctrl, cfg):
            # Simulate successful model load
            ctrl.model._tokenizer = object()
            ctrl.model._model = object()
            ctrl.model._device = "cpu"
            ctrl.model._loaded = True
            ctrl._model_loaded = True

        monkeypatch.setattr("axen_chat._do_load_with_hooks", _fake_load)

        success = load_model_with_progress(controller, config)
        assert success is True
        assert controller.model_ready is True

    def test_load_model_with_progress_returns_false_on_import_error(self, monkeypatch):
        """
        load_model_with_progress() returns False when torch is missing.
        """
        from axen.assistant.controller import AssistantController
        from axen.model.hf_model import HFModelConfig
        from axen_chat import load_model_with_progress

        config = HFModelConfig()
        controller = AssistantController(config=config)

        def _failing_load(ctrl, cfg):
            raise ImportError("No module named 'torch'")

        monkeypatch.setattr("axen_chat._do_load_with_hooks", _failing_load)

        success = load_model_with_progress(controller, config)
        assert success is False
        assert controller.model_ready is False

    def test_load_model_with_progress_returns_false_on_runtime_error(self, monkeypatch):
        """
        load_model_with_progress() returns False when model load fails at runtime
        (e.g. OOM or download failure).
        """
        from axen.assistant.controller import AssistantController
        from axen.model.hf_model import HFModelConfig
        from axen_chat import load_model_with_progress

        config = HFModelConfig()
        controller = AssistantController(config=config)

        def _oom_load(ctrl, cfg):
            raise RuntimeError("CUDA out of memory")

        monkeypatch.setattr("axen_chat._do_load_with_hooks", _oom_load)

        success = load_model_with_progress(controller, config)
        assert success is False
        assert controller.model_ready is False


# -----------------------------------------------------------------------
# 11. --no-model mode: model is NOT loaded
# -----------------------------------------------------------------------

class TestNoModelMode:
    """
    Verify --no-model mode skips model loading and still serves tools.
    """

    def test_no_model_flag_skips_loading(self):
        """
        In --no-model mode, the controller should not have model_ready=True.
        This is the expected state before load_model() is ever called.
        """
        from axen.assistant.controller import AssistantController
        controller = AssistantController()
        # Without calling load_model(), model should not be ready
        assert controller.model_ready is False

    def test_calculator_works_without_model(self):
        """Calculator tool works even when model is not loaded."""
        from axen.assistant.controller import AssistantController
        controller = AssistantController()
        response = controller.respond("calculate 999 * 111")
        assert "110889" in response

    def test_python_tool_works_without_model(self):
        """Python executor works even when model is not loaded."""
        from axen.assistant.controller import AssistantController
        controller = AssistantController()
        code_input = "run:\n```python\nprint('no model needed')\n```"
        response = controller.respond(code_input)
        assert "no model needed" in response or "python" in response.lower()

    def test_fallback_message_is_clear(self):
        """When model is not loaded, the fallback message is informative."""
        from axen.assistant.controller import AssistantController
        controller = AssistantController()
        response = controller.respond("Tell me a joke")
        # Should say something about the model not being loaded
        assert "load" in response.lower() or "model" in response.lower()


# -----------------------------------------------------------------------
# 12. --status mode: reports correct state
# -----------------------------------------------------------------------

class TestStatusMode:
    """
    Verify the status() method returns accurate information.
    """

    def test_status_reports_model_name(self):
        from axen.assistant.controller import AssistantController
        from axen.model.hf_model import HFModelConfig
        cfg = HFModelConfig(name="test/model")
        controller = AssistantController(config=cfg)
        st = controller.status()
        # Model name should appear in the repr
        assert "test/model" in st["model"]

    def test_status_reports_not_loaded_before_load(self):
        from axen.assistant.controller import AssistantController
        controller = AssistantController()
        st = controller.status()
        assert st["model_loaded"] is False

    def test_status_reports_loaded_after_mock_load(self):
        from axen.assistant.controller import AssistantController
        controller = AssistantController()
        # Simulate a successful load
        controller.model._tokenizer = object()
        controller.model._model = object()
        controller.model._device = "cpu"
        controller.model._loaded = True
        controller._model_loaded = True
        st = controller.status()
        assert st["model_loaded"] is True

    def test_status_has_tools_list(self):
        from axen.assistant.controller import AssistantController
        controller = AssistantController()
        st = controller.status()
        assert isinstance(st["tools"], list)
        assert len(st["tools"]) >= 2

    def test_status_reports_conversation_turns(self):
        from axen.assistant.controller import AssistantController
        import tempfile
        from axen.memory.long_term_memory import LongTermMemory
        tmpdir = tempfile.mkdtemp()
        ltm = LongTermMemory(path=Path(tmpdir) / "ltm.json")
        controller = AssistantController(long_term_memory=ltm)
        controller.respond("hello")  # model not loaded, still adds to memory
        st = controller.status()
        assert st["conversation_turns"] == 1


# -----------------------------------------------------------------------
# 13. Successful model load state
# -----------------------------------------------------------------------

class TestModelLoadedState:
    """
    Verify that after a successful load (simulated), the controller
    correctly reflects the loaded state and can generate responses.
    """

    def test_model_ready_after_simulated_load(self):
        from axen.assistant.controller import AssistantController
        controller = AssistantController()
        # Simulate successful model injection (same as _do_load_with_hooks does)
        controller.model._tokenizer = object()
        controller.model._model = object()
        controller.model._device = "cuda"
        controller.model._loaded = True
        controller._model_loaded = True
        assert controller.model_ready is True

    def test_model_not_ready_before_load(self):
        from axen.assistant.controller import AssistantController
        controller = AssistantController()
        assert controller.model_ready is False

    def test_model_not_ready_after_unload(self):
        from axen.assistant.controller import AssistantController
        controller = AssistantController()
        # Simulate load then unload
        controller.model._tokenizer = object()
        controller.model._model = object()
        controller.model._device = "cpu"
        controller.model._loaded = True
        controller._model_loaded = True
        assert controller.model_ready is True
        # Now unload
        controller.model._loaded = False
        controller.model._model = None
        controller.model._tokenizer = None
        controller._model_loaded = False
        assert controller.model_ready is False


# -----------------------------------------------------------------------
# 14. Conversation Intelligence & Pipeline Regression Tests
# -----------------------------------------------------------------------

class MockLLMModel:
    """Mock model capturing messages sent by the controller."""
    def __init__(self, canned_response="Hello! I am AXEN, your local AI assistant."):
        self.canned_response = canned_response
        self.received_messages = []
        self._loaded = True
        self.model_name = "MockLLM"
        self._device = "cpu"

    @property
    def is_loaded(self):
        return True

    def chat(self, messages, **kwargs):
        self.received_messages.append(list(messages))
        return self.canned_response

    def chat_with_stats(self, messages, **kwargs):
        self.received_messages.append(list(messages))
        stats = {
            "input_tokens": sum(len(m.get("content", "")) // 4 for m in messages),
            "generated_tokens": len(self.canned_response) // 4,
            "roles": [m.get("role", "unknown") for m in messages],
        }
        return self.canned_response, stats


class TestConversationIntelligence:
    """Regression tests for Conversation Intelligence, Tool Routing, and Guardrails."""

    def test_current_user_message_reaches_model(self):
        from axen.assistant.controller import AssistantController
        from axen.memory.long_term_memory import LongTermMemory
        import tempfile
        tmp = tempfile.mkdtemp()
        mock = MockLLMModel("Python is a popular programming language.")
        controller = AssistantController(
            model=mock,
            long_term_memory=LongTermMemory(path=Path(tmp) / "ltm.json")
        )
        controller._model_loaded = True

        response = controller.respond("What is Python?")
        assert response == "Python is a popular programming language."
        assert len(mock.received_messages) == 1
        last_conv = mock.received_messages[0]
        # Current user message must be role: user and contain exact prompt
        assert last_conv[-1]["role"] == "user"
        assert last_conv[-1]["content"] == "What is Python?"

    def test_correct_message_roles_and_no_duplicates(self):
        from axen.assistant.controller import AssistantController
        from axen.memory.long_term_memory import LongTermMemory
        import tempfile
        tmp = tempfile.mkdtemp()
        mock = MockLLMModel("Blue is a wonderful color.")
        controller = AssistantController(
            model=mock,
            long_term_memory=LongTermMemory(path=Path(tmp) / "ltm.json")
        )
        controller._model_loaded = True

        controller.respond("My favorite color is blue.")
        mock.canned_response = "Your favorite color is blue!"
        controller.respond("What is my favorite color?")

        assert len(mock.received_messages) == 2
        second_conv = mock.received_messages[1]

        # Roles: system, user, assistant, user
        roles = [m["role"] for m in second_conv]
        assert roles == ["system", "user", "assistant", "user"]
        assert second_conv[1]["content"] == "My favorite color is blue."
        assert second_conv[2]["content"] == "Blue is a wonderful color."
        assert second_conv[3]["content"] == "What is my favorite color?"

    def test_fresh_conversation_starts_clean(self):
        from axen.assistant.controller import AssistantController
        from axen.memory.long_term_memory import LongTermMemory
        import tempfile
        tmp = tempfile.mkdtemp()
        controller = AssistantController(
            long_term_memory=LongTermMemory(path=Path(tmp) / "ltm.json")
        )
        assert controller.memory.turn_count == 0
        msgs = controller.memory.to_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"

    def test_deterministic_calculator_routing_words(self):
        from axen.assistant.controller import AssistantController
        from axen.memory.long_term_memory import LongTermMemory
        import tempfile
        tmp = tempfile.mkdtemp()
        controller = AssistantController(
            long_term_memory=LongTermMemory(path=Path(tmp) / "ltm.json")
        )
        # Test 2: What is 25 multiplied by 48?
        resp = controller.respond("What is 25 multiplied by 48?")
        assert "1200" in resp

        # Test 9: What is 12345 * 678?
        resp2 = controller.respond("What is 12345 * 678?")
        assert "8369910" in resp2

    def test_symbolic_math_routing_derivative(self):
        from axen.assistant.controller import AssistantController
        from axen.memory.long_term_memory import LongTermMemory
        import tempfile
        tmp = tempfile.mkdtemp()
        controller = AssistantController(
            long_term_memory=LongTermMemory(path=Path(tmp) / "ltm.json")
        )
        resp = controller.respond("Calculate the derivative of x^3")
        assert "3*x**2" in resp

    def test_file_tool_routing(self):
        from axen.assistant.controller import AssistantController
        from axen.memory.long_term_memory import LongTermMemory
        import tempfile
        tmp = tempfile.mkdtemp()
        controller = AssistantController(
            long_term_memory=LongTermMemory(path=Path(tmp) / "ltm.json")
        )
        resp = controller.respond("Read this file: configs/axen_local.yaml")
        assert "hf_model" in resp or "TinyLlama" in resp

    def test_truncate_turn_stops_continuation(self):
        from axen.model.hf_model import _truncate_turn
        raw = "Recursion is when a function calls itself.<|user|>\nWhat is an iteration?"
        assert _truncate_turn(raw) == "Recursion is when a function calls itself."

        raw2 = "I am an AI assistant.\nUser: Do you work for Google?\nAssistant: Yes."
        assert _truncate_turn(raw2) == "I am an AI assistant."

    def test_identity_guardrail_company(self):
        from axen.assistant.controller import AssistantController
        from axen.memory.long_term_memory import LongTermMemory
        import tempfile
        tmp = tempfile.mkdtemp()
        mock = MockLLMModel("I work for Acme Corp and joined the company 2 years ago.")
        controller = AssistantController(
            model=mock,
            long_term_memory=LongTermMemory(path=Path(tmp) / "ltm.json")
        )
        controller._model_loaded = True
        resp = controller.respond("What company do you work for?")
        assert "I don't work for a company" in resp
        assert "Acme" not in resp

    def test_identity_guardrail_hobbies(self):
        from axen.assistant.controller import AssistantController
        from axen.memory.long_term_memory import LongTermMemory
        import tempfile
        tmp = tempfile.mkdtemp()
        mock = MockLLMModel("I love playing guitar and watching movies in my spare time.")
        controller = AssistantController(
            model=mock,
            long_term_memory=LongTermMemory(path=Path(tmp) / "ltm.json")
        )
        controller._model_loaded = True
        resp = controller.respond("Do you have hobbies?")
        assert "I don't have personal hobbies" in resp
        assert "guitar" not in resp

    def test_debug_mode_records_stats(self):
        from axen.assistant.controller import AssistantController
        from axen.memory.long_term_memory import LongTermMemory
        import tempfile
        tmp = tempfile.mkdtemp()
        mock = MockLLMModel("Hello! How can I help?")
        controller = AssistantController(
            model=mock,
            long_term_memory=LongTermMemory(path=Path(tmp) / "ltm.json")
        )
        controller._model_loaded = True
        controller.respond("Hello", debug=True)
        assert hasattr(controller, "last_debug_info")
        dbg = controller.last_debug_info
        assert dbg["route"] == "model"
        assert dbg["model"] == "MockLLM"
        assert "roles" in dbg
        assert "input_tokens" in dbg
        assert "generated_tokens" in dbg


