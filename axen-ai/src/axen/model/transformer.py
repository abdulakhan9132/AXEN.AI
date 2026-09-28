from __future__ import annotations

import torch
from torch import nn

from .attention import MultiHeadAttention


class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.self_attention = MultiHeadAttention(embedding_dim, num_heads, dropout=dropout)
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
        residual = x
        x = self.norm1(x)
        x = self.self_attention(x)
        x = residual + x
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        return residual + x


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, embedding_dim: int, num_layers: int, num_heads: int, ff_dim: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = nn.Embedding(context_length, embedding_dim)
        self.layers = nn.ModuleList([
            DecoderBlock(embedding_dim, num_heads, ff_dim, dropout=dropout, activation=activation)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(token_ids.size(1), device=token_ids.device).unsqueeze(0).expand(token_ids.size(0), -1)
        x = self.token_embedding(token_ids) + self.positional_embedding(positions)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
