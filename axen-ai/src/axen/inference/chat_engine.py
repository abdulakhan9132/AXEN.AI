from __future__ import annotations

from pathlib import Path
from threading import Event
from typing import Any

from axen.inference.generator import generate_text
from axen.training.overfit import load_checkpoint
from axen.utils.device import detect_device


class AXENChatEngine:
    """Small local adapter between the desktop UI and AXEN inference."""

    def __init__(self, checkpoint_path: str | Path = "checkpoints/best.pt") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.model = None
        self.tokenizer = None
        self.metadata: dict[str, Any] = {}
        self.error: str | None = None

    def load_model(self) -> bool:
        if not self.checkpoint_path.exists():
            self.error = f"Checkpoint not found: {self.checkpoint_path}"
            return False
        try:
            self.model, self.tokenizer, self.metadata = load_checkpoint(
                self.checkpoint_path, device=detect_device()
            )
        except (OSError, RuntimeError, KeyError, ValueError, TypeError) as exc:
            self.model = None
            self.tokenizer = None
            self.error = str(exc)
            return False
        self.error = None
        return True

    def is_model_ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None and self.error is None

    def model_info(self) -> dict[str, Any]:
        config = self.metadata.get("config", {})
        parameter_count = (
            sum(parameter.numel() for parameter in self.model.parameters())
            if self.model is not None
            else 0
        )
        return {
            "model": "AXEN Tiny",
            "parameters": parameter_count,
            "vocabulary": len(self.tokenizer.vocab) if self.tokenizer is not None else config.get("vocab_size", 0),
            "context": config.get("context_length", 0),
            "device": str(next(self.model.parameters()).device) if self.model is not None else str(detect_device()),
            "status": "READY" if self.is_model_ready() else "NOT TRAINED",
            "error": self.error,
        }

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 48,
        temperature: float = 0.8,
        stop_event: Event | None = None,
    ) -> str:
        if not prompt.strip():
            raise ValueError("Message cannot be empty.")
        if not self.is_model_ready():
            raise RuntimeError("AXEN model is not ready yet. Complete training before chatting.")
        if stop_event is not None and stop_event.is_set():
            return ""
        return generate_text(
            self.model,
            self.tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop_event=stop_event,
        )
