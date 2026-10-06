"""
axen/model/foundation.py
========================
Canonical AXEN Foundation Model Implementation, Architecture, Tokenizer, and Checkpoint Manager.

This is the genuine, native neural brain of AXEN:
- Decoder-only Transformer with causal multi-head attention
- Versioned deterministic ASCII + JSON + Symbol Tokenizer (Tokenizer v2)
- Structured supervised training support with causal next-token prediction
- Full support for native structured tool calls, conversation, and reasoning
- Seamless integration with BrainDecisionAdapter and AgentLoop
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

logger = logging.getLogger("axen.model.foundation")


# ---------------------------------------------------------------------------
# Tokenizer V2 (Versioned, deterministic ASCII + JSON + Tool Call Tokenizer)
# ---------------------------------------------------------------------------

class FoundationTokenizer:
    """
    Deterministic Tokenizer for AXEN Foundation Model (v2).
    
    Guarantees:
    - Index 0: <PAD>
    - Index 1: <UNK>
    - Index 2: <BOS>
    - Index 3: <EOS>
    - Indices 4-5: \\n, \\t
    - Indices 6-100: All 95 printable ASCII characters (space to ~), including
      all JSON delimiters ({}, []:" ,), math operators (+-*/=), numbers (0-9),
      code symbols, and punctuation.
    - Total canonical vocabulary: 101 characters (well within vocab_size=256).
    - Unknown characters gracefully mapped to <UNK> (1).
    - Supports loading legacy v1 vocabulary for backward compatibility.
    """

    SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    VERSION = "v2"

    def __init__(self, vocab: Optional[Dict[str, int]] = None, version: str = "v2") -> None:
        self.version = version
        if vocab is not None:
            self.vocab = dict(vocab)
        else:
            self.vocab = self._build_canonical_v2_vocab()

        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.pad_token_id = self.vocab.get(self.pad_token, 0)
        self.unk_token_id = self.vocab.get(self.unk_token, 1)
        self.bos_token_id = self.vocab.get(self.bos_token, 2)
        self.eos_token_id = self.vocab.get(self.eos_token, 3)

    @classmethod
    def _build_canonical_v2_vocab(cls) -> Dict[str, int]:
        mapping: Dict[str, int] = {}
        for idx, tok in enumerate(cls.SPECIAL_TOKENS):
            mapping[tok] = idx
        mapping["\n"] = len(mapping)
        mapping["\t"] = len(mapping)
        # Printable ASCII: ord 32 (space) through ord 126 (tilde)
        for code in range(32, 127):
            ch = chr(code)
            if ch not in mapping:
                mapping[ch] = len(mapping)
        return mapping

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        tokens: List[int] = []
        if add_special_tokens:
            tokens.append(self.bos_token_id)
        for ch in text:
            tokens.append(self.vocab.get(ch, self.unk_token_id))
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        """Decode token IDs back to string."""
        chars: List[str] = []
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        for tid in token_ids:
            tid_int = int(tid)
            if skip_special_tokens and tid_int in special_ids:
                continue
            if tid_int == self.unk_token_id:
                if not skip_special_tokens:
                    chars.append(self.unk_token)
                continue
            token_str = self.inverse_vocab.get(tid_int, "")
            if skip_special_tokens and token_str in self.SPECIAL_TOKENS:
                continue
            chars.append(token_str)
        return "".join(chars)

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({
                "version": self.version,
                "vocab": self.vocab,
            }, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FoundationTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "vocab" in data:
            return cls(vocab=data["vocab"], version=data.get("version", "v2"))
        elif isinstance(data, dict):
            return cls(vocab=data, version="v1")
        raise ValueError(f"Invalid tokenizer file format at {path}")


# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------

@dataclass
class FoundationModelConfig:
    """Hyperparameters and architectural config for AXEN Foundation Model."""
    model_name: str = "axen_foundation"
    vocab_size: int = 256
    context_length: int = 256
    embedding_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    ff_dim: int = 256
    dropout: float = 0.1
    activation: str = "gelu"
    learning_rate: float = 0.0003
    device: str = "auto"
    tokenizer_version: str = "v2"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FoundationModelConfig:
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Transformer Architecture
# ---------------------------------------------------------------------------

class CausalMultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})")
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self._causal_mask(seq_len, x.device)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        result = torch.matmul(weights, v)
        result = result.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dim)
        return self.out_proj(result)


class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1, activation: str = "gelu") -> None:
        super().__init__()
        self.self_attention = CausalMultiHeadAttention(embedding_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.GELU() if activation.lower() == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embedding_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class TransformerCore(nn.Module):
    def __init__(self, config: FoundationModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.positional_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        self.layers = nn.ModuleList([
            DecoderBlock(
                embedding_dim=config.embedding_dim,
                num_heads=config.num_heads,
                ff_dim=config.ff_dim,
                dropout=config.dropout,
                activation=config.activation,
            )
            for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.embedding_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        seq_len = token_ids.size(1)
        if seq_len > self.config.context_length:
            raise ValueError(f"Sequence length {seq_len} exceeds context length {self.config.context_length}")
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(token_ids.size(0), -1)
        x = self.token_embedding(token_ids) + self.positional_embedding(positions)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# AXEN Foundation Model
# ---------------------------------------------------------------------------

class AXENFoundationModel(nn.Module):
    """
    Authoritative AXEN Foundation Model.
    
    Provides:
    - forward(token_ids) -> logits
    - generate(prompt_or_tokens, ...) -> generated string or tokens
    - chat(messages, ...) -> direct text output for runtime compatibility
    - is_loaded -> True when initialized
    """

    def __init__(self, config: Optional[FoundationModelConfig] = None, tokenizer: Optional[FoundationTokenizer] = None) -> None:
        super().__init__()
        self.config = config or FoundationModelConfig()
        self.tokenizer = tokenizer or FoundationTokenizer(version=self.config.tokenizer_version)
        self.transformer = TransformerCore(self.config)
        self.lm_head = nn.Linear(self.config.embedding_dim, self.config.vocab_size)
        self._device = self._resolve_device(self.config.device)
        self.to(self._device)
        self._loaded: bool = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> torch.device:
        return self._device

    def _resolve_device(self, dev: str) -> torch.device:
        if dev == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(dev)

    def count_parameters(self) -> Tuple[int, int]:
        """Returns (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.transformer(token_ids)
        return self.lm_head(hidden)

    @staticmethod
    def format_prompt(user_query: str) -> str:
        """Standard prompt template format for AXEN foundation model."""
        return f"USER: {user_query.strip()}\nASSISTANT: "

    def generate(
        self,
        prompt_or_tokens: Union[str, torch.Tensor],
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Autoregressive text generation."""
        self.eval()
        if isinstance(prompt_or_tokens, str):
            encoded = self.tokenizer.encode(prompt_or_tokens, add_special_tokens=True)
            # Remove EOS from initial prompt so generation continues
            if encoded and encoded[-1] == self.tokenizer.eos_token_id:
                encoded = encoded[:-1]
            token_ids = torch.tensor([encoded], dtype=torch.long, device=self.device)
        else:
            token_ids = prompt_or_tokens.to(self.device)

        context_length = self.config.context_length
        generated = token_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                model_input = generated[:, -context_length:]
                logits = self(model_input)[:, -1, :] / max(temperature, 1e-5)

                if temperature <= 0.05:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    if top_p is not None and 0.0 < top_p < 1.0:
                        probs = torch.softmax(logits, dim=-1)
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = False
                        indices_to_remove = torch.zeros_like(sorted_indices_to_remove).scatter_(1, sorted_indices, sorted_indices_to_remove)
                        logits = logits.masked_fill(indices_to_remove, -1e9)

                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                generated = torch.cat([generated, next_token], dim=1)

        # Decode generated slice (excluding initial prompt)
        new_token_ids = generated[0, token_ids.size(1):].tolist()
        return self.tokenizer.decode(new_token_ids, skip_special_tokens=True)

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 150,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 0.9,
        **kwargs: Any,
    ) -> str:
        """
        Chat generation interface compatible with BrainDecisionAdapter and AgentLoop.
        Extracts user query, builds prompt, and generates reply.
        """
        user_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break
        if not user_content and messages:
            user_content = messages[-1].get("content", "")

        eff_tokens = max_new_tokens if max_new_tokens is not None else max_tokens
        prompt = self.format_prompt(user_content)
        raw_out = self.generate(prompt, max_new_tokens=eff_tokens, temperature=temperature, top_p=top_p)
        return raw_out.strip()

    def save_checkpoint(
        self,
        path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        step: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save model checkpoint with config, tokenizer, and metadata."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state": self.state_dict(),
            "config": self.config.to_dict(),
            "tokenizer_vocab": self.tokenizer.vocab,
            "tokenizer_version": self.tokenizer.version,
            "step": step,
            "metadata": metadata or {},
        }
        if optimizer is not None:
            payload["optimizer_state"] = optimizer.state_dict()
        torch.save(payload, p)
        logger.info("Saved AXEN Foundation Model checkpoint to %s", p)

    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tuple["AXENFoundationModel", Dict[str, Any]]:
        """Load model checkpoint and return (model, metadata)."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {p}")
        target_dev = "cpu" if device is None else str(device)
        payload = torch.load(p, map_location=target_dev, weights_only=False)

        # Support both new format and legacy axen-ai format
        if "config" in payload:
            cfg_dict = payload["config"]
            cfg = FoundationModelConfig.from_dict(cfg_dict)
        else:
            cfg = FoundationModelConfig()

        if "tokenizer_vocab" in payload:
            tok = FoundationTokenizer(
                vocab=payload["tokenizer_vocab"],
                version=payload.get("tokenizer_version", "v2"),
            )
        elif "tokenizer_state" in payload:
            tok = FoundationTokenizer(vocab=payload["tokenizer_state"], version="v1")
        else:
            tok = FoundationTokenizer()

        model = cls(config=cfg, tokenizer=tok)

        state_dict = payload.get("model_state") or payload.get("model")
        if state_dict:
            model.load_state_dict(state_dict)

        metadata = payload.get("metadata", {})
        return model, metadata
