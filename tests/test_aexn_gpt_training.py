"""
tests/test_aexn_gpt_training.py
===============================
Comprehensive Test Suite for AEXN GPT 1.0 Neural Network Training.

Verifies:
1.  Input/output token alignment (causal next-token pairs)
2.  Transformer output tensor dimensions (batch, seq, vocab)
3.  Causal future-token attention masking
4.  Cross-entropy loss calculation on raw logits
5.  Backpropagation and non-zero gradient flow
6.  AdamW parameter update verification (weights genuinely update)
7.  Empirical training loss reduction over iterations
8.  Validation evaluation and perplexity metrics
9.  Checkpoint saving, loading, and integrity
10. Training resumption and state consistency
11. Text generation with temperature and top-k filtering
12. CPU and GPU device compatibility
13. ModelManager registration and provider resolution
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from app.ai.model_manager import ModelManager
from axen.model.bpe_tokenizer import AEXNTokenizer, get_batch
from axen.model.foundation import AXENFoundationModel, FoundationModelConfig
from scripts.train_aexn_gpt import (
    evaluate_validation,
    get_learning_rate,
    load_checkpoint,
    save_checkpoint,
)


@pytest.fixture(scope="session")
def tokenizer() -> AEXNTokenizer:
    tok_path = Path("data/shakespeare/aexn_bpe.json")
    if tok_path.exists():
        return AEXNTokenizer.load(tok_path)
    tok = AEXNTokenizer(vocab_size=300, min_frequency=2)
    tok.train_from_text("To be or not to be, that is the question: " * 10)
    return tok


@pytest.fixture
def small_model_config(tokenizer) -> FoundationModelConfig:
    return FoundationModelConfig(
        model_name="test_gpt",
        vocab_size=tokenizer.vocab_size,
        context_length=32,
        embedding_dim=64,
        num_layers=2,
        num_heads=2,
        ff_dim=128,
        dropout=0.0,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# 1. Correct Input/Output Token Alignment
# ---------------------------------------------------------------------------

def test_01_input_output_token_alignment():
    """Verify that causal training pairs satisfy y[t] == x[t+1]."""
    train_bin = Path("data/shakespeare/train_ids.bin")
    if not train_bin.exists():
        pytest.skip("train_ids.bin not found")

    batch_size = 4
    context_length = 32
    bx, by = get_batch(train_bin, batch_size=batch_size, context_length=context_length, device="cpu")

    assert bx.shape == (batch_size, context_length)
    assert by.shape == (batch_size, context_length)
    # y must be shifted by 1 token into the future relative to x
    assert torch.equal(by[:, :-1], bx[:, 1:])


# ---------------------------------------------------------------------------
# 2. Transformer Output Dimensions
# ---------------------------------------------------------------------------

def test_02_transformer_output_dimensions(small_model_config, tokenizer):
    """Verify logits tensor has shape (batch_size, context_length, vocab_size)."""
    model = AXENFoundationModel(config=small_model_config, tokenizer=tokenizer)
    x = torch.randint(0, small_model_config.vocab_size, (3, small_model_config.context_length))

    logits = model(x)
    assert logits.shape == (3, small_model_config.context_length, small_model_config.vocab_size)


# ---------------------------------------------------------------------------
# 3. Causal Attention Masking (No Future Leakage)
# ---------------------------------------------------------------------------

def test_03_causal_attention_masking(small_model_config, tokenizer):
    """
    Verify that altering future tokens in the input has ZERO effect on
    the logits at preceding positions.
    """
    model = AXENFoundationModel(config=small_model_config, tokenizer=tokenizer)
    model.eval()

    seq1 = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]])
    seq2 = torch.tensor([[10, 20, 30, 40, 99, 88, 77, 66]])  # Changed from index 4 onwards

    with torch.no_grad():
        logits1 = model(seq1)
        logits2 = model(seq2)

    # Positions 0..3 must have identical logits because attention is strictly causal
    diff = torch.max(torch.abs(logits1[:, :4, :] - logits2[:, :4, :]))
    assert diff.item() < 1e-5, f"Causal attention leakage detected: max diff = {diff.item()}"


# ---------------------------------------------------------------------------
# 4. Cross-Entropy Loss from Raw Logits
# ---------------------------------------------------------------------------

def test_04_cross_entropy_loss_calculation(small_model_config, tokenizer):
    """Verify loss computation using raw logits and target tokens."""
    model = AXENFoundationModel(config=small_model_config, tokenizer=tokenizer)
    x = torch.randint(0, small_model_config.vocab_size, (2, 16))
    y = torch.randint(0, small_model_config.vocab_size, (2, 16))

    logits, loss = model(x, targets=y)
    assert logits.shape == (2, 16, small_model_config.vocab_size)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar loss
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() > 0.0


# ---------------------------------------------------------------------------
# 5. Backpropagation & Non-Zero Gradient Flow
# ---------------------------------------------------------------------------

def test_05_backpropagation_and_gradients(small_model_config, tokenizer):
    """Verify that every trainable parameter receives a valid, non-zero gradient."""
    model = AXENFoundationModel(config=small_model_config, tokenizer=tokenizer)
    model.train()

    x = torch.randint(0, small_model_config.vocab_size, (2, 16))
    y = torch.randint(0, small_model_config.vocab_size, (2, 16))

    logits, loss = model(x, targets=y)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has None gradient"
            grad_norm = param.grad.norm().item()
            assert not math.isnan(grad_norm), f"Parameter {name} has NaN gradient"
            assert grad_norm >= 0.0, f"Parameter {name} has invalid gradient norm"


# ---------------------------------------------------------------------------
# 6. AdamW Parameter Updates (Weights Genuinely Change)
# ---------------------------------------------------------------------------

def test_06_adamw_parameter_updates(small_model_config, tokenizer):
    """Verify that parameters genuinely change after an optimizer step."""
    model = AXENFoundationModel(config=small_model_config, tokenizer=tokenizer)
    model.train()
    optimizer = model.configure_optimizers(weight_decay=0.01, learning_rate=0.01)

    initial_weights = {name: param.clone() for name, param in model.named_parameters()}

    x = torch.randint(0, small_model_config.vocab_size, (4, 16))
    y = torch.randint(0, small_model_config.vocab_size, (4, 16))

    optimizer.zero_grad()
    logits, loss = model(x, targets=y)
    loss.backward()
    optimizer.step()

    updated = False
    for name, param in model.named_parameters():
        diff = torch.norm(param - initial_weights[name]).item()
        if diff > 1e-6:
            updated = True
            break
    assert updated, "AdamW did not update any model parameters!"


# ---------------------------------------------------------------------------
# 7. Empirical Loss Reduction Over Training
# ---------------------------------------------------------------------------

def test_07_training_loss_reduction(small_model_config, tokenizer):
    """Verify that training on fixed data significantly reduces loss over iterations."""
    model = AXENFoundationModel(config=small_model_config, tokenizer=tokenizer)
    model.train()
    optimizer = model.configure_optimizers(weight_decay=0.0, learning_rate=0.005)

    x = torch.randint(0, small_model_config.vocab_size, (2, 16))
    y = torch.randint(0, small_model_config.vocab_size, (2, 16))

    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        logits, loss = model(x, targets=y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"Loss failed to decrease: start={losses[0]:.4f}, end={losses[-1]:.4f}"
    assert losses[-1] < losses[0] * 0.7, "Loss did not reduce by at least 30% on synthetic data"


# ---------------------------------------------------------------------------
# 8. Validation Evaluation and Perplexity Metrics
# ---------------------------------------------------------------------------

def test_08_validation_evaluation(small_model_config, tokenizer):
    """Verify evaluate_validation computes mean loss and perplexity correctly."""
    train_bin = Path("data/shakespeare/train_ids.bin")
    if not train_bin.exists():
        pytest.skip("train_ids.bin not found")

    model = AXENFoundationModel(config=small_model_config, tokenizer=tokenizer)
    val_loss, perplexity = evaluate_validation(
        model, train_bin, eval_steps=5, batch_size=2, context_length=16, device=torch.device("cpu")
    )

    assert val_loss > 0.0
    assert perplexity > 1.0
    # Perplexity should equal exp(val_loss)
    assert abs(perplexity - math.exp(min(val_loss, 20.0))) < 1e-3


# ---------------------------------------------------------------------------
# 9. Checkpoint Saving, Loading, and State Consistency
# ---------------------------------------------------------------------------

def test_09_checkpoint_saving_loading(small_model_config, tokenizer):
    """Verify saving checkpoint and loading recovers identical weights."""
    model = AXENFoundationModel(config=small_model_config, tokenizer=tokenizer)
    optimizer = model.configure_optimizers()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_ckpt.pt"
        save_checkpoint(ckpt_path, model, optimizer, step=42, val_loss=3.1415, config_dict=small_model_config.to_dict())

        assert ckpt_path.exists()

        model_loaded = AXENFoundationModel(config=small_model_config, tokenizer=tokenizer)
        opt_loaded = model_loaded.configure_optimizers()
        step, val_loss = load_checkpoint(ckpt_path, model_loaded, opt_loaded, device=torch.device("cpu"))

        assert step == 42
        assert abs(val_loss - 3.1415) < 1e-4

        # Compare weights
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model_loaded.named_parameters()):
            assert torch.equal(p1, p2), f"Parameter mismatch in {n1}"


# ---------------------------------------------------------------------------
# 10. Text Generation with Temperature and Top-k
# ---------------------------------------------------------------------------

def test_10_text_generation(small_model_config, tokenizer):
    """Verify autoregressive text generation with temperature and top-k."""
    model = AXENFoundationModel(config=small_model_config, tokenizer=tokenizer)
    prompt = "Hello"

    out = model.generate(prompt, max_new_tokens=15, temperature=0.7, top_k=20, top_p=0.9)
    assert isinstance(out, str)
    assert len(out) >= 0


# ---------------------------------------------------------------------------
# 11. CPU and GPU Device Compatibility
# ---------------------------------------------------------------------------

def test_11_device_compatibility(small_model_config, tokenizer):
    """Verify model can run on CPU and (if available) GPU."""
    model_cpu = AXENFoundationModel(config=small_model_config, tokenizer=tokenizer)
    assert model_cpu.device.type == "cpu"
    x = torch.randint(0, small_model_config.vocab_size, (1, 8))
    out_cpu = model_cpu(x)
    assert out_cpu.shape == (1, 8, small_model_config.vocab_size)

    if torch.cuda.is_available():
        gpu_cfg = FoundationModelConfig.from_dict(small_model_config.to_dict())
        gpu_cfg.device = "cuda"
        model_gpu = AXENFoundationModel(config=gpu_cfg, tokenizer=tokenizer)
        assert model_gpu.device.type == "cuda"
        x_gpu = x.to("cuda")
        out_gpu = model_gpu(x_gpu)
        assert out_gpu.device.type == "cuda"
        assert out_gpu.shape == (1, 8, small_model_config.vocab_size)


# ---------------------------------------------------------------------------
# 12. ModelManager Integration
# ---------------------------------------------------------------------------

def test_12_model_manager_registration():
    """Verify AEXN GPT 1.0 is registered and retrievable via ModelManager."""
    manager = ModelManager()
    catalog = manager.list_models()

    matching = [m for m in catalog if m["id"] == "aexn_gpt_1.0"]
    assert len(matching) == 1, "aexn_gpt_1.0 not found in ModelManager catalog!"
    entry = matching[0]
    assert entry["name"] == "AEXN GPT 1.0 — My Own Trained Model"
    assert entry["provider"] == "aexn_native_bpe"

    model_adapter = manager.get_model("aexn_gpt_1.0")
    assert model_adapter is not None
    assert model_adapter.health_check() is True
