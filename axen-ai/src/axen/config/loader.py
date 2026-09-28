from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 256
    context_length: int = 32
    embedding_dim: int = 32
    num_layers: int = 2
    num_heads: int = 4
    ff_dim: int = 128
    dropout: float = 0.1
    activation: str = "gelu"
    learning_rate: float = 3e-4
    batch_size: int = 4
    epochs: int = 5
    max_steps: int = 100
    gradient_clip: float = 1.0
    validation_frequency: int = 1
    checkpoint_dir: str = "checkpoints"
    seed: int = 42
    device: str = "auto"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    train_ratio: float = 0.9
    validation_ratio: float = 0.05

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        model = raw.get("model", {})
        training = raw.get("training", {})
        system = raw.get("system", {})
        data = raw.get("data", {})
        return cls(
            vocab_size=int(model.get("vocab_size", 256)),
            context_length=int(model.get("context_length", 32)),
            embedding_dim=int(model.get("embedding_dim", 32)),
            num_layers=int(model.get("num_layers", 2)),
            num_heads=int(model.get("num_heads", 4)),
            ff_dim=int(model.get("ff_dim", 128)),
            dropout=float(model.get("dropout", 0.1)),
            activation=str(model.get("activation", "gelu")),
            learning_rate=float(training.get("learning_rate", 3e-4)),
            batch_size=int(training.get("batch_size", 4)),
            epochs=int(training.get("epochs", 5)),
            max_steps=int(training.get("max_steps", 100)),
            gradient_clip=float(training.get("gradient_clip", 1.0)),
            validation_frequency=int(training.get("validation_frequency", 1)),
            checkpoint_dir=str(training.get("checkpoint_dir", "checkpoints")),
            seed=int(system.get("seed", 42)),
            device=str(system.get("device", "auto")),
            raw_dir=str(data.get("raw_dir", "data/raw")),
            processed_dir=str(data.get("processed_dir", "data/processed")),
            train_ratio=float(data.get("train_ratio", 0.9)),
            validation_ratio=float(data.get("validation_ratio", 0.05)),
        )
