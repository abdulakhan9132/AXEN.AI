from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


class Tokenizer:
    """Simple character-level tokenizer for the first AXEN model iteration."""

    SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

    def __init__(self, vocab: Dict[str, int] | None = None):
        if vocab is None:
            vocab = {token: idx for idx, token in enumerate(self.SPECIAL_TOKENS)}
        self.vocab = vocab
        self.inverse_vocab = {idx: token for token, idx in vocab.items()}
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def fit(self, texts: Iterable[str]) -> "Tokenizer":
        vocab = {token: idx for idx, token in enumerate(self.SPECIAL_TOKENS)}
        for text in texts:
            for ch in text:
                if ch not in vocab:
                    vocab[ch] = len(vocab)
        self.vocab = vocab
        self.inverse_vocab = {idx: token for token, idx in vocab.items()}
        return self

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens: List[int] = []
        if add_special_tokens:
            tokens.append(self.vocab.get(self.bos_token, 0))
        for ch in text:
            tokens.append(self.vocab.get(ch, self.vocab.get(self.unk_token, 0)))
        if add_special_tokens:
            tokens.append(self.vocab.get(self.eos_token, 0))
        return tokens

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(self.inverse_vocab.get(int(token), self.unk_token) for token in token_ids)

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, sort_keys=True)

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab=vocab)
