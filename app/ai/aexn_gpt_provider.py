"""
app/ai/aexn_gpt_provider.py
===========================
Model Provider Adapter for AEXN GPT 1.0 — Custom Trained BPE Neural Model.

Integrates with AEXN AI's Unified Model Management System, Think Engine,
and Agent Loop without external API dependencies.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import torch

from .base_model import AIModel
from app.context.models import ModelRequest
from axen.model.bpe_tokenizer import AEXNTokenizer
from axen.model.foundation import AXENFoundationModel, FoundationModelConfig

logger = logging.getLogger("axen.ai.aexn_gpt")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class AEXNGPTProvider(AIModel):
    """
    Native Provider for AEXN GPT 1.0.
    Direct in-process PyTorch inference with BPE tokenizer.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        tokenizer_path: str = "data/shakespeare/aexn_bpe.json",
        device: str = "auto",
    ) -> None:
        self.device_str = device
        self.device = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else "cpu" if device == "auto" else device)
        self.tokenizer_path = PROJECT_ROOT / tokenizer_path

        # Resolve checkpoint preference: instruct -> best -> latest
        candidates = [
            PROJECT_ROOT / (checkpoint_path or "checkpoints/aexn_gpt/aexn_gpt_instruct.pt"),
            PROJECT_ROOT / "checkpoints/aexn_gpt/best_model.pt",
            PROJECT_ROOT / "checkpoints/aexn_gpt/checkpoint_latest.pt",
        ]
        self.resolved_checkpoint: Optional[Path] = None
        for cand in candidates:
            if cand.exists():
                self.resolved_checkpoint = cand
                break

        self.tokenizer: Optional[AEXNTokenizer] = None
        self.model: Optional[AXENFoundationModel] = None
        self._initialized = False

    def _lazy_init(self) -> None:
        """Initialize model and tokenizer on first generation call."""
        if self._initialized:
            return

        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"AEXN BPE Tokenizer not found at {self.tokenizer_path}")

        self.tokenizer = AEXNTokenizer.load(self.tokenizer_path)

        if self.resolved_checkpoint and self.resolved_checkpoint.exists():
            ckpt = torch.load(self.resolved_checkpoint, map_location=self.device, weights_only=False)
            cfg = FoundationModelConfig.from_dict(ckpt.get("config", {}))
            cfg.device = str(self.device)
            self.model = AXENFoundationModel(config=cfg, tokenizer=self.tokenizer)
            self.model.load_state_dict(ckpt["model_state"])
        else:
            raise FileNotFoundError("No trained AEXN GPT checkpoint is available")

        self.model.to(self.device)
        self.model.eval()
        self._initialized = True
        logger.info("Initialized AEXN GPT 1.0 Provider on %s", self.device)

    def health_check(self) -> bool:
        """Return True if tokenizer exists and PyTorch environment is functional."""
        return self.tokenizer_path.exists() and self.resolved_checkpoint is not None

    def generate(self, request: ModelRequest) -> str:
        self._lazy_init()
        messages = request.context.messages_for_model if request.context else []
        if not messages:
            raise ValueError("A user message is required")

        temperature = request.generation_settings.get("temperature", 0.7)
        max_tokens = request.generation_settings.get("max_new_tokens", request.generation_settings.get("max_tokens", 80))

        # Extract latest user message
        user_query = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_query = m.get("content", "")
                break
        if not user_query and messages:
            user_query = messages[-1].get("content", "")

        prompt = f"USER: {user_query.strip()}\nASSISTANT: "
        completion = self.model.generate(
            prompt_or_tokens=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=30,
            top_p=0.9,
        )
        return completion.strip()

    def stream(self, request: ModelRequest) -> Iterator[str]:
        full_text = self.generate(request)
        words = full_text.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")

    def structured_output(self, request: ModelRequest, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        text = self.generate(request)
        try:
            return json.loads(text)
        except Exception:
            return {"response": text, "model": "aexn_gpt_1.0"}
