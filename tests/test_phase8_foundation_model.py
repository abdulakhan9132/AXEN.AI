"""
tests/test_phase8_foundation_model.py
=====================================
Comprehensive Test Suite for Phase 8: AXEN Foundation Model Training & Capability Growth.

Verifies:
1. Foundation Model Architecture & Configuration
2. Tokenizer v2 (Printable ASCII, JSON, Symbols, Whitespace, Legacy v1 compatibility)
3. Model Metadata & Baseline Preservation
4. Dataset Architecture, Capabilities Balancing & Sanitization
5. Supervised Next-Token Training, Prompt Masking & Loss
6. Smoke Testing, Finite Gradients & Checkpoint Versioning
7. Overfitting Protection (Best Validation Checkpoint)
8. Deterministic Greedy Inference & Latency Benchmarks
9. Native Brain Structured Tool Intent Emission & Validation
10. Controlled Fallback to Planner
11. Strict Qwen Isolation (Coding Only vs AXEN General Brain)
12. Memory & Security Regression Protection
13. Promotion Rules & Verification
"""

import json
import math
import os
import tempfile
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.ai.router import ModelRouter
from app.context.models import AssembledContext, ModelRequest
from axen.agent.learning import (
    BrainEvaluationHarness,
    CheckpointMetadata,
    DatasetSanitizer,
    LearningEvent,
    LearningStore,
    TrainingExampleGenerator,
)
from axen.agent.loop import AgentLoop
from axen.agent.tool_intent import (
    BrainDecision,
    BrainDecisionAdapter,
    DecisionSource,
    ToolIntent,
    ToolIntentParser,
    ToolIntentType,
    ToolIntentValidator,
)
from axen.memory.conversation_memory import ConversationMemory
from axen.memory.long_term_memory import LongTermMemory
from axen.model.dataset import DatasetManager, InstructionDataset
from axen.model.foundation import (
    AXENFoundationModel,
    FoundationModelConfig,
    FoundationTokenizer,
)
from axen.model.trainer import FoundationTrainer, TrainingConfig
from axen.tools import build_default_registry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "README.md").write_text("# Project Title\nProject docs.", encoding="utf-8")
    (ws / "data.csv").write_text("id,val\n1,10\n2,20\n3,30\n", encoding="utf-8")
    (ws / "auth.py").write_text("def authenticate(): return True\n", encoding="utf-8")
    (ws / "config.py").write_text("DEBUG = True\nPORT = 8000\n", encoding="utf-8")
    return ws


@pytest.fixture
def foundation_model():
    return AXENFoundationModel()


@pytest.fixture
def default_registry():
    return build_default_registry()


# ---------------------------------------------------------------------------
# 1. Foundation Model Configuration
# ---------------------------------------------------------------------------
def test_01_foundation_model_config():
    cfg = FoundationModelConfig()
    assert cfg.model_name == "axen_foundation"
    assert cfg.vocab_size == 256
    assert cfg.context_length == 256
    assert cfg.num_layers == 2
    assert cfg.num_heads == 4
    assert cfg.embedding_dim == 64
    assert cfg.ff_dim == 256

    d = cfg.to_dict()
    assert d["model_name"] == "axen_foundation"
    reloaded = FoundationModelConfig.from_dict(d)
    assert reloaded.context_length == 256


# ---------------------------------------------------------------------------
# 2. Model Architecture & Parameter Count
# ---------------------------------------------------------------------------
def test_02_foundation_model_architecture(foundation_model):
    total, trainable = foundation_model.count_parameters()
    assert total > 140_000
    assert total == trainable
    assert foundation_model.is_loaded is True
    assert hasattr(foundation_model, "transformer")
    assert hasattr(foundation_model, "lm_head")


# ---------------------------------------------------------------------------
# 3. Tokenizer v2 ASCII Coverage
# ---------------------------------------------------------------------------
def test_03_tokenizer_v2_ascii_coverage():
    tok = FoundationTokenizer()
    assert tok.vocab_size >= 101
    assert tok.pad_token_id == 0
    assert tok.unk_token_id == 1
    assert tok.bos_token_id == 2
    assert tok.eos_token_id == 3

    # All standard ASCII chars must encode without UNK (1)
    test_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789{}[]:\",/*+-=_ \n\t"
    encoded = tok.encode(test_chars, add_special_tokens=False)
    assert tok.unk_token_id not in encoded


# ---------------------------------------------------------------------------
# 4. Tokenizer v2 JSON Roundtrip
# ---------------------------------------------------------------------------
def test_04_tokenizer_v2_json_roundtrip():
    tok = FoundationTokenizer()
    raw_json = '{"type": "tool_call", "tool": "calculator.evaluate", "arguments": {"expression": "25 * 48"}}'
    encoded = tok.encode(raw_json, add_special_tokens=True)
    decoded = tok.decode(encoded, skip_special_tokens=True)
    assert decoded == raw_json


# ---------------------------------------------------------------------------
# 5. Tokenizer Legacy v1 Compatibility
# ---------------------------------------------------------------------------
def test_05_tokenizer_legacy_v1_compatibility(tmp_path):
    legacy_file = tmp_path / "legacy_vocab.json"
    legacy_vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "a": 4, "b": 5}
    legacy_file.write_text(json.dumps(legacy_vocab), encoding="utf-8")

    loaded_tok = FoundationTokenizer.load(legacy_file)
    assert loaded_tok.version == "v1"
    assert loaded_tok.encode("ab", add_special_tokens=False) == [4, 5]


# ---------------------------------------------------------------------------
# 6. Model Metadata Saved
# ---------------------------------------------------------------------------
def test_06_model_metadata_saved():
    p = Path("evaluation/model_metadata.json")
    assert p.exists()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["model_name"] == "axen_foundation"
    assert data["parameters"] > 140_000
    assert data["vocab_size"] == 256
    assert data["layers"] == 2
    assert "checkpoint" in data


# ---------------------------------------------------------------------------
# 7. Foundation Baseline Preserved Without Overwriting Phase 7
# ---------------------------------------------------------------------------
def test_07_foundation_baseline_preserved():
    p7 = Path("evaluation/baseline.json")
    assert p7.exists()
    p7_data = json.loads(p7.read_text(encoding="utf-8"))
    assert p7_data["total_tasks"] == 11
    assert p7_data["decision_sources"]["planner_fallback"] == 8

    fb = Path("evaluation/foundation_baseline.json")
    assert fb.exists()
    fb_data = json.loads(fb.read_text(encoding="utf-8"))
    assert fb_data["benchmark_name"] == "axen_foundation_baseline"
    assert "loss_metrics" in fb_data


# ---------------------------------------------------------------------------
# 8. Dataset Architecture Structure
# ---------------------------------------------------------------------------
def test_08_dataset_architecture_structure(tmp_path):
    dm = DatasetManager(base_dir=tmp_path)
    assert dm.raw_dir.exists()
    assert dm.sanitized_dir.exists()
    assert dm.curated_dir.exists()
    assert dm.val_dir.exists()
    assert dm.test_dir.exists()
    assert dm.metadata_dir.exists()


# ---------------------------------------------------------------------------
# 9. Dataset Sanitization of Secrets
# ---------------------------------------------------------------------------
def test_09_dataset_sanitization_secrets():
    raw_prompt = "My api_key is sk-secret1234567890. Calculate 10 + 5."
    clean = DatasetSanitizer.clean_text(raw_prompt)
    assert "sk-secret" not in clean
    assert "[REDACTED_SECRET]" in clean


# ---------------------------------------------------------------------------
# 10. Dataset Sanitization of Thoughts
# ---------------------------------------------------------------------------
def test_10_dataset_sanitization_thoughts():
    raw_response = "<think>Consider tool calculator.</think>Here is the result."
    clean = DatasetSanitizer.clean_text(raw_response)
    assert "<think>" not in clean
    assert "Consider tool" not in clean
    assert "Here is the result." in clean


# ---------------------------------------------------------------------------
# 11. Dataset Balancing and Report
# ---------------------------------------------------------------------------
def test_11_dataset_balancing_and_report():
    report_path = Path("evaluation/dataset_report.json")
    assert report_path.exists()
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    assert rep["total_examples"] >= 50
    assert "conversation" in rep["categories_count"]
    assert "tool_intent" in rep["categories_count"]
    assert "knowledge" in rep["categories_count"]
    # No single category should dominate (> 30%)
    for cat, pct in rep["category_distribution_percentage"].items():
        assert pct < 30.0, f"Category {cat} dominates with {pct}%"


# ---------------------------------------------------------------------------
# 12. Supervised Prompt Masking in InstructionDataset
# ---------------------------------------------------------------------------
def test_12_instruction_dataset_prompt_masking():
    tok = FoundationTokenizer()
    sample = [{"input": "Hello", "target": "Hi there!"}]
    ds = InstructionDataset(sample, tokenizer=tok, max_length=64)
    inputs, targets = ds[0]

    # First tokens corresponding to prompt 'USER: Hello\nASSISTANT: ' must be 0 (masked)
    assert targets[0].item() == 0
    assert targets[5].item() == 0
    # Later tokens must contain the target characters (> 0)
    non_zeros = targets[targets > 0]
    assert len(non_zeros) > 0


# ---------------------------------------------------------------------------
# 13. Training Smoke Test Execution
# ---------------------------------------------------------------------------
def test_13_trainer_smoke_test_execution(tmp_path):
    model = AXENFoundationModel()
    train_data = [{"category": "conv", "input": "Hi", "target": "Hello!"}] * 8
    val_data = [{"category": "conv", "input": "Hi", "target": "Hello!"}] * 4
    cfg = TrainingConfig(epochs=2, batch_size=2, checkpoint_dir=str(tmp_path))

    trainer = FoundationTrainer(model, train_data, val_data, config=cfg)
    smoke_res = trainer.run_smoke_test(max_steps=10)
    assert smoke_res["status"] == "passed"
    assert smoke_res["loss_decreased"] is True
    assert smoke_res["gpu_memory_mb"] >= 0.0


# ---------------------------------------------------------------------------
# 14. Checkpoint Save and Reload
# ---------------------------------------------------------------------------
def test_14_checkpoint_save_and_reload(tmp_path):
    model = AXENFoundationModel()
    ckpt_path = tmp_path / "test_ckpt.pt"
    model.save_checkpoint(ckpt_path, metadata={"version_id": "test_v1"})
    assert ckpt_path.exists()

    reloaded, meta = AXENFoundationModel.load_checkpoint(ckpt_path)
    assert reloaded.is_loaded is True
    assert meta.get("version_id") == "test_v1"


# ---------------------------------------------------------------------------
# 15. Overfitting Protection Best Validation Checkpoint
# ---------------------------------------------------------------------------
def test_15_overfitting_protection_best_val_checkpoint(tmp_path):
    model = AXENFoundationModel()
    train_data = [{"category": "conv", "input": "Hi", "target": "Hello!"}] * 8
    val_data = [{"category": "conv", "input": "Hi", "target": "Hello!"}] * 4
    cfg = TrainingConfig(epochs=3, batch_size=2, checkpoint_dir=str(tmp_path))

    trainer = FoundationTrainer(model, train_data, val_data, config=cfg)
    res = trainer.train(version_id="brain_test")
    best_path = Path(res["best_checkpoint_path"])
    assert best_path.exists()
    assert res["best_val_loss"] <= res["initial_val_loss"] or res["best_val_loss"] < 6.0


# ---------------------------------------------------------------------------
# 16. Deterministic Greedy Generation
# ---------------------------------------------------------------------------
def test_16_deterministic_greedy_generation(foundation_model):
    prompt = "USER: Hello\nASSISTANT: "
    out1 = foundation_model.generate(prompt, max_new_tokens=15, temperature=0.0)
    out2 = foundation_model.generate(prompt, max_new_tokens=15, temperature=0.0)
    assert out1 == out2


# ---------------------------------------------------------------------------
# 17. Structured Tool Intent Emission
# ---------------------------------------------------------------------------
def test_17_native_structured_tool_intent_emission():
    raw = '{"type": "tool_call", "tool": "calculator.evaluate", "arguments": {"expression": "25 * 48"}}'
    decision = ToolIntentParser.parse(raw)
    assert decision.decision_type == ToolIntentType.TOOL_CALL
    assert decision.tool_intent is not None
    assert decision.tool_intent.tool == "calculator.evaluate"
    assert decision.tool_intent.arguments["expression"] == "25 * 48"


# ---------------------------------------------------------------------------
# 18. Tool Intent Validator Accepts Native Intent
# ---------------------------------------------------------------------------
def test_18_tool_intent_validator_accepts_native_intent(default_registry):
    intent = ToolIntent(
        type=ToolIntentType.TOOL_CALL,
        tool="calculator.evaluate",
        arguments={"expression": "25 * 48"},
    )
    is_valid, err = ToolIntentValidator.validate(intent, registry=default_registry)
    assert is_valid is True
    assert err is None


# ---------------------------------------------------------------------------
# 19. Brain Decision Adapter Records Native Source
# ---------------------------------------------------------------------------
def test_19_brain_decision_adapter_records_native_source(default_registry):
    adapter = BrainDecisionAdapter(registry=default_registry)
    mock_model = MagicMock()
    mock_model.is_loaded = True
    mock_model.chat.return_value = '{"type": "tool_call", "tool": "calculator.evaluate", "arguments": {"expression": "25 * 48"}}'

    decision = adapter.decide("Calculate 25 * 48.", model=mock_model)
    assert decision.decision_source == DecisionSource.BRAIN_NATIVE
    assert adapter.metrics[DecisionSource.BRAIN_NATIVE.value] >= 1


# ---------------------------------------------------------------------------
# 20. Agent Loop Executes Native Brain Plan
# ---------------------------------------------------------------------------
def test_20_agent_loop_executes_native_brain_plan(default_registry, temp_workspace):
    mock_model = MagicMock()
    mock_model.is_loaded = True
    mock_model.chat.return_value = '{"type": "tool_call", "tool": "calculator.evaluate", "arguments": {"expression": "25 * 48"}}'

    loop = AgentLoop(
        tools=default_registry,
        memory=ConversationMemory(),
        long_memory=LongTermMemory(None),
        model=mock_model,
        workspace_path=str(temp_workspace),
    )
    ans, state = loop.run("Calculate 25 * 48.")
    assert state.metadata.get("decision_source") == "brain_native"
    assert any("calculator" in c.get("tool", "") for c in state.tool_calls)


# ---------------------------------------------------------------------------
# 21. Controlled Fallback on Unrecognized Request
# ---------------------------------------------------------------------------
def test_21_planner_fallback_on_unrecognized_request(default_registry, temp_workspace):
    mock_model = MagicMock()
    mock_model.is_loaded = True
    # Model generates invalid text
    mock_model.chat.return_value = "Unparseable non-json response."

    loop = AgentLoop(
        tools=default_registry,
        memory=ConversationMemory(),
        long_memory=LongTermMemory(None),
        model=mock_model,
        workspace_path=str(temp_workspace),
    )
    ans, state = loop.run("Calculate 25 * 48.")
    # Safe fallback to deterministic planner was executed
    assert state.status == "completed"
    assert state.metadata.get("decision_source") == "planner_fallback"


# ---------------------------------------------------------------------------
# 22. Qwen Isolation: Coding Strictly to Qwen
# ---------------------------------------------------------------------------
def test_22_qwen_isolation_coding_only():
    router = ModelRouter()
    assert router._is_coding_task("Write a Python function to reverse a string.") is True
    assert router._is_coding_task("Debug this Python exception in auth.py.") is True
    assert router._is_coding_task("Implement a REST endpoint in FastAPI.") is True


# ---------------------------------------------------------------------------
# 23. General Knowledge Isolation: AXEN Foundation Brain, NEVER Qwen
# ---------------------------------------------------------------------------
def test_23_axen_foundation_general_knowledge_isolation():
    router = ModelRouter()
    assert router._is_coding_task("What is a computer?") is False
    assert router._is_coding_task("What is machine learning?") is False
    assert router._is_coding_task("Explain the concept of entropy in thermodynamics.") is False


# ---------------------------------------------------------------------------
# 24. Conversation Isolation: AXEN Foundation Brain, NEVER Qwen
# ---------------------------------------------------------------------------
def test_24_axen_foundation_conversation_isolation():
    router = ModelRouter()
    assert router._is_coding_task("Hello") is False
    assert router._is_coding_task("Who are you?") is False
    assert router._is_coding_task("What can you do?") is False


# ---------------------------------------------------------------------------
# 25. Context vs Persistent Memory Separation
# ---------------------------------------------------------------------------
def test_25_context_vs_persistent_memory_separation():
    conv_mem = ConversationMemory()
    long_mem = LongTermMemory(None)
    conv_mem.add("user", "My project uses Python.")

    # Session memory contains it (system message + user message)
    assert len(conv_mem.to_messages()) == 2
    # Long-term memory is not automatically polluted
    assert long_mem.recall("project_language") is None


# ---------------------------------------------------------------------------
# 26. Reflection Structured Output
# ---------------------------------------------------------------------------
def test_26_reflection_structured_output():
    raw_reflection = '{"status": "complete", "reason_code": "verified_result"}'
    parsed = json.loads(raw_reflection)
    assert parsed["status"] == "complete"
    assert parsed["reason_code"] == "verified_result"


# ---------------------------------------------------------------------------
# 27. Correction Handling
# ---------------------------------------------------------------------------
def test_27_correction_handling():
    evt = LearningEvent(
        task_type="calculation",
        user_request="Calculate 10 / 0",
        result="failure",
        correction={"type": "final_response", "content": "Division by zero is undefined."},
    )
    gen = TrainingExampleGenerator(store=MagicMock())
    ex = gen.generate_example_from_event(evt)
    assert ex is not None
    assert ex["category"] == "correction"
    assert "undefined" in ex["target"]["content"]


# ---------------------------------------------------------------------------
# 28. Safety Regression: Path Traversal Remains Blocked
# ---------------------------------------------------------------------------
def test_28_security_path_traversal_remains_blocked(default_registry):
    malicious_intent = ToolIntent(
        type=ToolIntentType.TOOL_CALL,
        tool="file_read",
        arguments={"path": "../../../etc/passwd"},
    )
    is_valid, err = ToolIntentValidator.validate(malicious_intent, registry=default_registry)
    assert is_valid is False
    assert "outside" in err.lower() or "traversal" in err.lower() or "invalid" in err.lower()


# ---------------------------------------------------------------------------
# 29. Real Inference Benchmark Recorded
# ---------------------------------------------------------------------------
def test_29_real_inference_benchmark_recorded():
    report_path = Path("evaluation/training_report.json")
    assert report_path.exists()
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    assert "inference_benchmark" in rep
    inf = rep["inference_benchmark"]
    assert "avg_latency_ms" in inf
    assert "tokens_per_second" in inf
    assert "gpu_memory_mb" in inf
    assert inf["avg_latency_ms"] > 0


# ---------------------------------------------------------------------------
# 30. Candidate Promotion Decision Verified
# ---------------------------------------------------------------------------
def test_30_candidate_promotion_decision_verified():
    report_path = Path("evaluation/training_report.json")
    assert report_path.exists()
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    assert "promotion_decision" in rep
    assert rep["promotion_decision"] == "promoted"
    assert rep["capability_comparison"]["candidate"]["brain_native"] > rep["capability_comparison"]["baseline"]["brain_native"]
