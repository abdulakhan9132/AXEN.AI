from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from axen.config.loader import ModelConfig
from axen.data.dataset import TextDataset
from axen.model.language_model import AXENLanguageModel
from axen.tokenizer.tokenizer import Tokenizer
from axen.training.overfit import build_model, save_checkpoint
from axen.utils.device import detect_device
from axen.utils.reproducibility import set_seed


@dataclass
class TrainingResult:
    rows: list[dict[str, float | int | str]]
    steps: int
    best_validation_loss: float
    device: str
    best_checkpoint: str
    latest_checkpoint: str


def _loss_for_batch(model: AXENLanguageModel, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = model(inputs)
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=0,
    )


class Trainer:
    def __init__(
        self,
        model: AXENLanguageModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: ModelConfig,
        tokenizer: Tokenizer,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.tokenizer = tokenizer
        self.device = detect_device() if config.device == "auto" else torch.device(config.device)
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate)

    def train_epoch(self, max_steps: int | None = None) -> tuple[float, int]:
        self.model.train()
        total_loss = 0.0
        batches = 0
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            loss = _loss_for_batch(self.model, inputs, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss: {loss.item()}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"Non-finite gradient norm: {gradient_norm.item()}")
            self.optimizer.step()
            total_loss += float(loss.detach().item())
            batches += 1
            if max_steps is not None and batches >= max_steps:
                break
        if batches == 0:
            raise ValueError("Training split produced no batches.")
        return total_loss / batches, batches

    def evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        batches = 0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                loss = _loss_for_batch(self.model, inputs, targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite validation loss: {loss.item()}")
                total_loss += float(loss.item())
                batches += 1
        if batches == 0:
            raise ValueError("Validation split produced no batches.")
        return total_loss / batches

    def fit(
        self,
        epochs: int,
        checkpoint_dir: str | Path,
        *,
        validation_frequency: int = 1,
        max_steps: int | None = None,
        on_epoch: Callable[[dict[str, float | int | str]], None] | None = None,
    ) -> TrainingResult:
        if epochs <= 0 or validation_frequency <= 0:
            raise ValueError("epochs and validation_frequency must be positive.")
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        best_path = checkpoint_path / "best.pt"
        latest_path = checkpoint_path / "latest.pt"
        log_path = checkpoint_path / "training_log.csv"
        rows: list[dict[str, float | int | str]] = []
        best_validation_loss = math.inf
        total_steps = 0
        started_at = time.perf_counter()
        for epoch in range(1, epochs + 1):
            remaining_steps = None if max_steps is None else max_steps - total_steps
            if remaining_steps is not None and remaining_steps <= 0:
                break
            train_loss, epoch_steps = self.train_epoch(remaining_steps)
            total_steps += epoch_steps
            should_validate = epoch % validation_frequency == 0 or epoch == epochs or (
                max_steps is not None and total_steps >= max_steps
            )
            validation_loss = self.evaluate() if should_validate else float("nan")
            elapsed = time.perf_counter() - started_at
            row: dict[str, float | int | str] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": elapsed,
                "steps": total_steps,
                "device": str(self.device),
            }
            rows.append(row)
            save_checkpoint(
                latest_path,
                self.model,
                self.optimizer,
                self.config,
                self.tokenizer,
                total_steps,
                metadata={"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss},
            )
            if should_validate and validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                save_checkpoint(
                    best_path,
                    self.model,
                    self.optimizer,
                    self.config,
                    self.tokenizer,
                    total_steps,
                    metadata={"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss},
                )
            if on_epoch is not None:
                on_epoch(row)
            if max_steps is not None and total_steps >= max_steps:
                break
        if not rows or not best_path.exists() or not latest_path.exists():
            raise RuntimeError("Training did not complete an optimizer step and checkpoint both outputs.")
        with log_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return TrainingResult(
            rows=rows,
            steps=total_steps,
            best_validation_loss=best_validation_loss,
            device=str(self.device),
            best_checkpoint=str(best_path),
            latest_checkpoint=str(latest_path),
        )


def train_prepared(
    config: ModelConfig,
    *,
    on_epoch: Callable[[dict[str, float | int | str]], None] | None = None,
) -> TrainingResult:
    set_seed(config.seed)
    processed = Path(config.processed_dir)
    tokenizer_path = processed / "tokenizer.json"
    train_path = processed / "train.json"
    validation_path = processed / "validation.json"
    missing = [str(path) for path in (tokenizer_path, train_path, validation_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing prepared training files: " + ", ".join(missing))
    tokenizer = Tokenizer.load(tokenizer_path)
    if tokenizer.vocab_size > config.vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary ({tokenizer.vocab_size}) exceeds model vocabulary ({config.vocab_size})."
        )
    train_dataset = TextDataset.from_file(train_path, config.context_length)
    validation_dataset = TextDataset.from_file(validation_path, config.context_length)
    if len(train_dataset.tokens) <= config.context_length:
        raise ValueError("Training split is too short for the configured sequence length.")
    if len(validation_dataset.tokens) <= config.context_length:
        raise ValueError("Validation split is too short for the configured sequence length.")
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)
    trainer = Trainer(build_model(config), train_loader, validation_loader, config, tokenizer)
    return trainer.fit(
        config.epochs,
        config.checkpoint_dir,
        validation_frequency=config.validation_frequency,
        max_steps=config.max_steps if config.max_steps > 0 else None,
        on_epoch=on_epoch,
    )
