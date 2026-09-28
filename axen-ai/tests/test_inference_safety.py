from pathlib import Path

import pytest
import torch

from axen.config.loader import ModelConfig
from axen.inference.chat_engine import AXENChatEngine
from axen.inference.generator import generate_text
from axen.model.language_model import AXENLanguageModel
from axen.tokenizer.tokenizer import Tokenizer
from axen.training.overfit import load_checkpoint


def test_token_id_out_of_range():
    model = AXENLanguageModel(8, 4, 8, 1, 2, 16, dropout=0.0)

    with pytest.raises(ValueError, match="Token ID out of range"):
        model(torch.tensor([[8]]))


def test_vocab_size_consistency():
    model, tokenizer, payload = load_checkpoint("checkpoints/overfit.pt")
    config = ModelConfig(**payload["config"])

    assert config.vocab_size == model.transformer.token_embedding.num_embeddings
    assert config.vocab_size == model.lm_head.out_features
    assert tokenizer.vocab_size <= config.vocab_size


def test_context_length_limit_and_position_embeddings():
    model, tokenizer, _ = load_checkpoint("checkpoints/overfit.pt")
    prompt = "hello world " * 20

    output = generate_text(model, tokenizer, prompt, max_new_tokens=2, temperature=0.8)

    assert output


def test_generation_token_range():
    model, tokenizer, _ = load_checkpoint("checkpoints/overfit.pt")
    device = next(model.parameters()).device
    generated = model.generate(
        torch.tensor([tokenizer.encode("hi", add_special_tokens=True)], device=device),
        max_new_tokens=8,
    )

    assert int(generated.min()) >= 0
    assert int(generated.max()) < model.lm_head.out_features


def test_tokenizer_model_compatibility_error(tmp_path: Path):
    engine = AXENChatEngine(tmp_path / "missing.pt")
    tokenizer = Tokenizer().fit(["hello"])

    assert tokenizer.vocab_size > 0
    assert not engine.is_model_ready()
