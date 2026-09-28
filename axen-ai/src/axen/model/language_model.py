from __future__ import annotations

import torch
from torch import nn
from threading import Event

from .transformer import Transformer


class AXENLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, embedding_dim: int, num_layers: int, num_heads: int, ff_dim: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.transformer = Transformer(vocab_size, context_length, embedding_dim, num_layers, num_heads, ff_dim, dropout, activation)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        vocab_size = self.transformer.token_embedding.num_embeddings
        if token_ids.numel() and (token_ids.min() < 0 or token_ids.max() >= vocab_size):
            raise ValueError(
                f"Token ID out of range: min={int(token_ids.min())}, "
                f"max={int(token_ids.max())}, vocabulary size={vocab_size}"
            )
        hidden = self.transformer(token_ids)
        return self.lm_head(hidden)

    def generate(self, token_ids: torch.Tensor, max_new_tokens: int = 8, temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None, stop_event: Event | None = None) -> torch.Tensor:
        generated = token_ids.clone()
        context_length = self.transformer.positional_embedding.num_embeddings
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if stop_event is not None and stop_event.is_set():
                    break
                model_input = generated[:, -context_length:]
                logits = self(model_input)[:, -1, :] / max(temperature, 1e-6)
                if top_k is not None:
                    topk = min(top_k, logits.size(-1))
                    values, indices = torch.topk(logits, k=topk)
                    masked = torch.full_like(logits, -1e9)
                    masked.scatter_(1, indices, values)
                    logits = masked
                if top_p is not None:
                    probs = torch.softmax(logits, dim=-1)
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cumulative = torch.cumsum(sorted_probs, dim=-1)
                    remove = cumulative > top_p
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = False
                    sorted_idx_to_remove = sorted_idx[remove]
                    logits = logits.clone()
                    logits.scatter_(1, sorted_idx_to_remove.unsqueeze(1), -1e9)
                next_token = torch.softmax(logits, dim=-1).multinomial(num_samples=1)
                if next_token.min() < 0 or next_token.max() >= logits.size(-1):
                    raise RuntimeError("Generation produced a token outside the model vocabulary.")
                generated = torch.cat([generated, next_token], dim=1)
                if next_token.eq(3).all():
                    break
        return generated
