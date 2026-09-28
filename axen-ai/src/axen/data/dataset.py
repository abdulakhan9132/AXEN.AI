from __future__ import annotations

from pathlib import Path
import json
from typing import Iterable, List, Sequence

import torch


class TextDataset(torch.utils.data.Dataset):
    """Create next-token prediction pairs from a token sequence."""

    def __init__(self, tokens: Sequence[int], context_length: int):
        self.tokens = list(tokens)
        self.context_length = context_length

    def __len__(self) -> int:
        return max(1, len(self.tokens) - self.context_length)

    def __getitem__(self, index: int):
        start = index
        end = start + self.context_length + 1
        window = self.tokens[start:end]
        if len(window) < self.context_length + 1:
            window = window + [0] * (self.context_length + 1 - len(window))
        x = torch.tensor(window[:-1], dtype=torch.long)
        y = torch.tensor(window[1:], dtype=torch.long)
        return x, y

    @classmethod
    def from_file(cls, path: str | Path, context_length: int) -> "TextDataset":
        source = Path(path)
        text = source.read_text(encoding="utf-8")
        if source.suffix.lower() == ".json":
            values = [int(value) for value in json.loads(text)]
        else:
            values = [int(v) for v in text.split()]
        return cls(values, context_length=context_length)
