from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from axen.config.loader import ModelConfig
from axen.data.dataset import TextDataset
from axen.model.attention import MultiHeadAttention
from axen.model.language_model import AXENLanguageModel
from axen.tokenizer.tokenizer import Tokenizer
from axen.training.overfit import load_checkpoint, run_overfit
from axen.utils.device import detect_device
from axen.utils.reproducibility import set_seed


CORPUS = "hello world\nhello axen\naxen is learning\nthe cat is small\nthe dog is running\n"


def _config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        context_length=32,
        embedding_dim=32,
        num_layers=1,
        num_heads=2,
        ff_dim=64,
        dropout=0.0,
        learning_rate=0.003,
        batch_size=4,
        max_steps=80,
        gradient_clip=1.0,
        seed=123,
        device="auto",
    )


def test_gradients_are_finite_and_parameters_update():
    set_seed(123)
    tokenizer = Tokenizer().fit([CORPUS])
    tokens = tokenizer.encode(CORPUS, add_special_tokens=False)
    dataset = TextDataset(tokens, context_length=32)
    x, y = next(iter(DataLoader(dataset, batch_size=4, shuffle=False)))
    model = AXENLanguageModel(64, 32, 32, 1, 2, 64, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    loss = nn.functional.cross_entropy(model(x).reshape(-1, 64), y.reshape(-1))
    assert loss.requires_grad
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert any(torch.any(gradient != 0) for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    optimizer.step()

    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))


def test_causal_attention_blocks_future_tokens():
    set_seed(123)
    attention = MultiHeadAttention(embedding_dim=8, num_heads=2, dropout=0.0).eval()
    prefix = torch.randn(1, 4, 8)
    changed_future = prefix.clone()
    changed_future[:, 2:, :] += 100.0

    first_output = attention(prefix)[:, :2, :]
    changed_output = attention(changed_future)[:, :2, :]

    assert torch.allclose(first_output, changed_output, atol=1e-5)


def test_overfit_checkpoint_and_generation(tmp_path: Path):
    checkpoint = tmp_path / "overfit.pt"
    result = run_overfit(CORPUS, _config(), checkpoint, print_every=0)

    assert result.final_loss < result.initial_loss * 0.5
    assert checkpoint.exists()
    assert all(torch.isfinite(torch.tensor(value)) for value in result.gradient_norms)

    model, tokenizer, payload = load_checkpoint(checkpoint, device=detect_device())
    assert payload["training_step"] == _config().max_steps
    prompt = torch.tensor(
        [tokenizer.encode("hello", add_special_tokens=False)],
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
    generated = model.generate(prompt, max_new_tokens=4, temperature=0.8)

    assert generated.shape[1] == prompt.shape[1] + 4
    assert tokenizer.decode(generated[0].tolist())
