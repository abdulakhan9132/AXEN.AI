from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from axen.config.loader import ModelConfig
from axen.data.dataset import TextDataset
from axen.model.language_model import AXENLanguageModel
from axen.tokenizer.tokenizer import Tokenizer
from axen.utils.device import detect_device
from axen.utils.reproducibility import set_seed


@dataclass
class OverfitResult:
    initial_loss: float
    final_loss: float
    steps: int
    duration_seconds: float
    device: str
    gradient_norms: list[float]
    learning_rates: list[float]
    checkpoint_path: str
    gpu_memory_mb: int


def build_model(config: ModelConfig) -> AXENLanguageModel:
    return AXENLanguageModel(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        ff_dim=config.ff_dim,
        dropout=config.dropout,
        activation=config.activation,
    )


def _gradient_norm(model: nn.Module) -> float:
    squared_norm = torch.zeros((), device=next(model.parameters()).device)
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared_norm = squared_norm + parameter.grad.detach().pow(2).sum()
    return float(torch.sqrt(squared_norm).item())


def _assert_finite(value: torch.Tensor, *, step: int, learning_rate: float, gradient_norm: float, config: ModelConfig) -> None:
    if not torch.isfinite(value).all() or not math.isfinite(gradient_norm):
        raise FloatingPointError(
            f"Non-finite training state at step {step}: "
            f"loss={value.detach().item()}, learning_rate={learning_rate}, "
            f"gradient_norm={gradient_norm}, config={config}"
        )


def save_checkpoint(
    path: str | Path,
    model: AXENLanguageModel,
    optimizer: torch.optim.Optimizer,
    config: ModelConfig,
    tokenizer: Tokenizer,
    step: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "training_step": step,
            "tokenizer_state": tokenizer.vocab,
            "metadata": metadata or {},
        },
        checkpoint_path,
    )


def load_checkpoint(path: str | Path, device: torch.device | None = None) -> tuple[AXENLanguageModel, Tokenizer, dict[str, Any]]:
    target_device = device or detect_device()
    payload = torch.load(path, map_location=target_device, weights_only=False)
    config = ModelConfig(**payload["config"])
    tokenizer = Tokenizer(vocab=payload["tokenizer_state"])
    token_ids = list(tokenizer.vocab.values())
    if token_ids and (min(token_ids) < 0 or max(token_ids) >= config.vocab_size):
        raise ValueError(
            "Tokenizer/model vocabulary mismatch: "
            f"tokenizer IDs range from {min(token_ids)} to {max(token_ids)}, "
            f"but model vocabulary size is {config.vocab_size}."
        )
    model = build_model(config).to(target_device)
    if model.transformer.token_embedding.num_embeddings != config.vocab_size:
        raise ValueError("Model configuration does not match token embedding vocabulary size.")
    if model.lm_head.out_features != config.vocab_size:
        raise ValueError("Model configuration does not match language-model head vocabulary size.")
    model.load_state_dict(payload["model"])
    model.eval()
    return model, tokenizer, payload


def run_overfit(
    corpus: str,
    config: ModelConfig,
    checkpoint_path: str | Path,
    *,
    print_every: int = 50,
) -> OverfitResult:
    set_seed(config.seed)
    device = detect_device() if config.device == "auto" else torch.device(config.device)
    tokenizer = Tokenizer().fit([corpus])
    tokens = tokenizer.encode(corpus, add_special_tokens=False)
    dataset = TextDataset(tokens, context_length=config.context_length)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    model.train()
    iterator = iter(loader)
    initial_loss: float | None = None
    final_loss = float("inf")
    gradient_norms: list[float] = []
    learning_rates: list[float] = []
    started_at = time.perf_counter()

    for step in range(config.max_steps):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        if not loss.requires_grad:
            raise RuntimeError(f"Loss does not require gradients at step {step}.")
        loss.backward()
        gradient_norm = _gradient_norm(model)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        _assert_finite(loss, step=step, learning_rate=learning_rate, gradient_norm=gradient_norm, config=config)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()

        final_loss = float(loss.detach().item())
        initial_loss = final_loss if initial_loss is None else initial_loss
        gradient_norms.append(gradient_norm)
        learning_rates.append(learning_rate)
        if print_every > 0 and (step == 0 or (step + 1) % print_every == 0):
            print(f"Step {step}\nLoss: {final_loss:.4f}\nGradient norm: {gradient_norm:.4f}\nLearning rate: {learning_rate:.6f}")

    save_checkpoint(checkpoint_path, model, optimizer, config, tokenizer, config.max_steps)
    duration = time.perf_counter() - started_at
    gpu_memory_mb = round(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0
    return OverfitResult(
        initial_loss=initial_loss if initial_loss is not None else float("nan"),
        final_loss=final_loss,
        steps=config.max_steps,
        duration_seconds=duration,
        device=str(device),
        gradient_norms=gradient_norms,
        learning_rates=learning_rates,
        checkpoint_path=str(checkpoint_path),
        gpu_memory_mb=gpu_memory_mb,
    )
