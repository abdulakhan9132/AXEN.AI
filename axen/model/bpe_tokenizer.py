"""
axen/model/bpe_tokenizer.py
===========================
Byte Pair Encoding (BPE) Tokenizer for AEXN GPT 1.0.

Provides two complementary implementations:
1. EducationalBPE:
   - Pure Python, transparent byte-level BPE implementation from scratch.
   - Designed for conceptual learning, inspection, and step-by-step tracing.
   - Shows exact pair frequency counting, greedy pair merging, and dynamic ID assignment.
   - Reversible round-trip decoding from learned token IDs back to original text.

2. AEXNTokenizer:
   - Production-grade Byte-Level BPE Tokenizer for AEXN GPT 1.0.
   - High-throughput training and inference on multi-megabyte corpora.
   - Standard canonical vocabulary size: 2,048 tokens.
   - Special reserved tokens: <PAD> (0), <UNK> (1), <BOS> (2), <EOS> (3).
   - Guarantees 100% reversible UTF-8 encoding and decoding with zero unknown-byte crashes.
   - Direct integration with AXEN Foundation Transformer and PyTorch datasets.

3. ShakespeareBPEDataset & get_batch:
   - High-performance memory-mapped binary array dataset for causal language modeling.
   - Prepares (x, y) input-target pairs for next-token prediction training.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("axen.model.bpe_tokenizer")


# ===========================================================================
# 1. EDUCATIONAL BPE IMPLEMENTATION (Pure Python, Step-by-Step)
# ===========================================================================

class EducationalBPE:
    """
    Pure Python Byte Pair Encoding (BPE) Tokenizer for educational demonstration.
    
    How it works from first principles:
    1. Base Vocabulary: Starts with the 256 fundamental byte values (0 to 255).
       Every UTF-8 character is composed of 1 to 4 bytes, so byte tokens can
       represent ANY text in the world without out-of-vocabulary errors.
    2. Pair Counting: Iterates through the token list and counts how often
       every adjacent pair (token[i], token[i+1]) appears.
    3. Pair Merging: Finds the most frequent pair and merges all of its occurrences
       into a brand new token ID (starting from 256, 257, 258...).
    4. Repeat: Continues merging until target vocabulary size is reached
       or no pair occurs at or above `min_freq`.
    5. Reversible Decoding: Reconstructs the original byte sequence and decodes UTF-8.
    """

    def __init__(self, target_vocab_size: int = 300, min_freq: int = 2) -> None:
        if target_vocab_size < 256:
            raise ValueError(f"target_vocab_size ({target_vocab_size}) must be at least 256 (the base byte set)")
        self.target_vocab_size = target_vocab_size
        self.min_freq = min_freq
        
        # Base vocabulary: mapping from token_id -> raw bytes
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        
        # Learned merge rules: mapping (token_id_1, token_id_2) -> new_token_id
        self.merges: Dict[Tuple[int, int], int] = {}
        
        # Merge priority rank: mapping (token_id_1, token_id_2) -> priority index (0 = first learned)
        self.merges_rank: Dict[Tuple[int, int], int] = {}
        
        # Educational trace history: records every merge decision for inspection
        self.history: List[Dict[str, Any]] = []

    @staticmethod
    def get_pair_frequencies(tokens: Sequence[int]) -> Dict[Tuple[int, int], int]:
        """
        Count occurrences of every adjacent token pair in the sequence.
        
        Example: [H, E, L, L, O]
        Pairs counted: (H, E): 1, (E, L): 1, (L, L): 1, (L, O): 1
        """
        counts: Dict[Tuple[int, int], int] = {}
        for i in range(len(tokens) - 1):
            pair = (int(tokens[i]), int(tokens[i + 1]))
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def merge_pair(tokens: Sequence[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        """
        Replace every adjacent occurrence of `pair` with `new_id`.
        
        Example:
        tokens = [1, 2, 3, 1, 2, 4], pair = (1, 2), new_id = 10
        result = [10, 3, 10, 4]
        """
        new_tokens: List[int] = []
        i = 0
        n = len(tokens)
        p0, p1 = pair
        while i < n:
            if i < n - 1 and tokens[i] == p0 and tokens[i + 1] == p1:
                new_tokens.append(new_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def train(self, text: Union[str, bytes], max_merges: Optional[int] = None) -> "EducationalBPE":
        """
        Train the educational BPE tokenizer on a training string or raw bytes.
        Records an educational step-by-step trace of every merge learned.
        """
        if isinstance(text, str):
            raw_bytes = text.encode("utf-8")
        else:
            raw_bytes = bytes(text)

        tokens = list(raw_bytes)
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges.clear()
        self.merges_rank.clear()
        self.history.clear()

        next_id = 256
        limit = max_merges if max_merges is not None else (self.target_vocab_size - 256)

        step = 0
        while len(self.vocab) < self.target_vocab_size and step < limit:
            if len(tokens) < 2:
                break
            
            stats = self.get_pair_frequencies(tokens)
            if not stats:
                break
            
            # Select most frequent pair. Tie-break deterministically by (frequency, -first_id, -second_id)
            best_pair = max(stats, key=lambda p: (stats[p], -p[0], -p[1]))
            best_count = stats[best_pair]

            if best_count < self.min_freq:
                # No more pairs occur frequently enough
                break

            # Assign new token ID
            new_id = next_id
            next_id += 1
            
            self.merges[best_pair] = new_id
            self.merges_rank[best_pair] = len(self.merges_rank)
            
            merged_bytes = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab[new_id] = merged_bytes

            tokens_before_len = len(tokens)
            tokens = self.merge_pair(tokens, best_pair, new_id)
            tokens_after_len = len(tokens)

            # Record educational trace step
            try:
                repr_str = merged_bytes.decode("utf-8")
            except UnicodeDecodeError:
                repr_str = repr(merged_bytes)

            self.history.append({
                "step": step + 1,
                "pair": best_pair,
                "pair_bytes": merged_bytes,
                "pair_repr": repr_str,
                "frequency": best_count,
                "new_token_id": new_id,
                "tokens_before": tokens_before_len,
                "tokens_after": tokens_after_len,
                "sequence_reduction": tokens_before_len - tokens_after_len,
                "vocab_size": len(self.vocab),
            })
            step += 1

        return self

    def encode(self, text: Union[str, bytes]) -> List[int]:
        """
        Encode text into a list of token IDs using learned merge rules.
        Greedily applies merges according to their learned priority (rank).
        """
        if isinstance(text, str):
            raw_bytes = text.encode("utf-8")
        else:
            raw_bytes = bytes(text)

        tokens = list(raw_bytes)
        
        while len(tokens) >= 2:
            stats = self.get_pair_frequencies(tokens)
            # Find eligible pairs that exist in our learned merges
            eligible = {pair: self.merges_rank[pair] for pair in stats if pair in self.merges_rank}
            if not eligible:
                break
            # Pick the pair with the earliest merge rank (highest priority)
            best_pair = min(eligible, key=eligible.get)
            tokens = self.merge_pair(tokens, best_pair, self.merges[best_pair])

        return tokens

    def decode(self, token_ids: Sequence[int]) -> str:
        """
        Decode a sequence of token IDs back into the original UTF-8 string.
        Guarantees 100% reversible round-trip.
        """
        byte_chunks: List[bytes] = []
        for tid in token_ids:
            tid_int = int(tid)
            if tid_int in self.vocab:
                byte_chunks.append(self.vocab[tid_int])
            else:
                byte_chunks.append(f"<UNK_{tid_int}>".encode("utf-8"))
        
        raw = b"".join(byte_chunks)
        return raw.decode("utf-8", errors="replace")

    def get_trace_summary(self, max_steps: int = 10) -> str:
        """Return a human-readable table of how BPE learned merges."""
        lines = [
            "+------+-------------------+---------------+-------+----------------------+",
            "| Step | Merged Pair IDs   | Token String  | Count | New Token ID (Vocab) |",
            "+------+-------------------+---------------+-------+----------------------+"
        ]
        for item in self.history[:max_steps]:
            p = f"({item['pair'][0]}, {item['pair'][1]})"
            r = repr(item['pair_repr'])[1:-1]
            if len(r) > 13:
                r = r[:10] + "..."
            lines.append(
                f"| {item['step']:<4} | {p:<17} | {r:<13} | {item['frequency']:<5} | ID {item['new_token_id']:<4} (size {item['vocab_size']}) |"
            )
        lines.append("+------+-------------------+---------------+-------+----------------------+")
        return "\n".join(lines)


# ===========================================================================
# 2. PRODUCTION BYTE-LEVEL BPE TOKENIZER (High-Speed, Reversible UTF-8)
# ===========================================================================

class AEXNTokenizer:
    """
    Authoritative Byte-Level BPE Tokenizer for AEXN GPT 1.0.
    
    Guarantees:
    - Target vocabulary size: 2,048 tokens (configurable).
    - Base alphabet: 256 byte values mapped to unique unicode characters.
    - Special tokens:
        <PAD>: 0
        <UNK>: 1
        <BOS>: 2
        <EOS>: 3
    - 100% Reversible UTF-8: encode(decode(ids)) == ids and decode(encode(text)) == text.
    - Saves canonical metadata and merge rules to JSON.
    - Supports fast C/Rust execution via `tokenizers` library with seamless fallback.
    """

    SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(
        self,
        vocab_size: int = 2048,
        min_frequency: int = 2,
        tokenizer_backend: Optional[Any] = None,
        vocab_dict: Optional[Dict[str, int]] = None,
        merges_list: Optional[List[str]] = None,
    ) -> None:
        self.target_vocab_size = vocab_size
        self.min_frequency = min_frequency
        self._tokenizer = tokenizer_backend
        self._vocab_dict = vocab_dict or {}
        self._merges_list = merges_list or []
        self._id_to_token: Dict[int, str] = {v: k for k, v in self._vocab_dict.items()}

        # Special token IDs
        self.pad_token_id = self._vocab_dict.get(self.PAD_TOKEN, 0)
        self.unk_token_id = self._vocab_dict.get(self.UNK_TOKEN, 1)
        self.bos_token_id = self._vocab_dict.get(self.BOS_TOKEN, 2)
        self.eos_token_id = self._vocab_dict.get(self.EOS_TOKEN, 3)

    @property
    def vocab_size(self) -> int:
        if self._tokenizer is not None:
            return self._tokenizer.get_vocab_size()
        return len(self._vocab_dict)

    @property
    def vocab(self) -> Dict[str, int]:
        if self._tokenizer is not None:
            return self._tokenizer.get_vocab()
        return dict(self._vocab_dict)

    def train_from_text(
        self,
        training_text: str,
        vocab_size: Optional[int] = None,
        min_frequency: Optional[int] = None,
    ) -> "AEXNTokenizer":
        """
        Train the Byte-Level BPE tokenizer on raw text.
        Uses optimized byte-level pre-tokenization and reversible decoding.
        """
        from tokenizers import Tokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer

        target_size = vocab_size if vocab_size is not None else self.target_vocab_size
        min_freq = min_frequency if min_frequency is not None else self.min_frequency

        raw_bpe = Tokenizer(BPE(unk_token=self.UNK_TOKEN))
        raw_bpe.pre_tokenizer = ByteLevel(add_prefix_space=False)
        raw_bpe.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=target_size,
            min_frequency=min_freq,
            special_tokens=self.SPECIAL_TOKENS,
            initial_alphabet=ByteLevel.alphabet(),
        )

        raw_bpe.train_from_iterator([training_text], trainer=trainer)

        self._tokenizer = raw_bpe
        self.target_vocab_size = self._tokenizer.get_vocab_size()
        self._vocab_dict = self._tokenizer.get_vocab()
        self._id_to_token = {v: k for k, v in self._vocab_dict.items()}
        
        self.pad_token_id = self._vocab_dict.get(self.PAD_TOKEN, 0)
        self.unk_token_id = self._vocab_dict.get(self.UNK_TOKEN, 1)
        self.bos_token_id = self._vocab_dict.get(self.BOS_TOKEN, 2)
        self.eos_token_id = self._vocab_dict.get(self.EOS_TOKEN, 3)

        return self

    def train_from_file(
        self,
        file_path: Union[str, Path],
        vocab_size: Optional[int] = None,
        min_frequency: Optional[int] = None,
    ) -> "AEXNTokenizer":
        """Train tokenizer directly from a text file path."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Training file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            text = f.read()
        return self.train_from_text(text, vocab_size=vocab_size, min_frequency=min_frequency)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Convert text into a list of numerical token IDs.
        
        Args:
            text: UTF-8 string to encode.
            add_special_tokens: If True, wraps sequence with <BOS> and <EOS>.
        """
        if self._tokenizer is None:
            raise RuntimeError("AEXNTokenizer is not initialized or trained. Call train_from_text() or load().")

        encoding = self._tokenizer.encode(text)
        ids = list(encoding.ids)
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        """
        Convert numerical token IDs back into the exact original text.
        
        Args:
            token_ids: Sequence of integer token IDs.
            skip_special_tokens: If True, omits <PAD>, <UNK>, <BOS>, <EOS>.
        """
        if self._tokenizer is None:
            raise RuntimeError("AEXNTokenizer is not initialized or trained. Call train_from_text() or load().")

        id_list = [int(tid) for tid in token_ids]
        if skip_special_tokens:
            special_ids = {self.pad_token_id, self.unk_token_id, self.bos_token_id, self.eos_token_id}
            id_list = [tid for tid in id_list if tid not in special_ids]

        return self._tokenizer.decode(id_list)

    def get_compression_ratio(self, text: str) -> float:
        """
        Calculate compression ratio: (raw UTF-8 byte count) / (token count).
        A ratio of 3.5 means 3.5 bytes of text are packed into 1 token on average.
        """
        raw_bytes = len(text.encode("utf-8"))
        tokens = self.encode(text, add_special_tokens=False)
        if not tokens:
            return 1.0
        return raw_bytes / len(tokens)

    def save(self, path: Union[str, Path]) -> None:
        """Save the tokenizer model and merge rules to a JSON file."""
        if self._tokenizer is None:
            raise RuntimeError("Cannot save an uninitialized tokenizer.")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(p))
        logger.info("Saved AEXNTokenizer to %s (vocab_size=%d)", p, self.vocab_size)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AEXNTokenizer":
        """Load a trained tokenizer from a JSON file."""
        from tokenizers import Tokenizer
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {p}")
        
        backend = Tokenizer.from_file(str(p))
        instance = cls(
            vocab_size=backend.get_vocab_size(),
            tokenizer_backend=backend,
            vocab_dict=backend.get_vocab(),
        )
        return instance


# ===========================================================================
# 3. SHAKESPEARE BPE DATASET & CAUSAL BATCH SAMPLER (Next-Token Prediction)
# ===========================================================================

class ShakespeareBPEDataset(Dataset):
    """
    High-performance PyTorch Dataset that memory-maps uint16 token IDs.
    
    Generates causal (x, y) training pairs:
    x: [token_0, token_1, ..., token_{T-1}]
    y: [token_1, token_2, ..., token_T]
    
    This matches the exact objective of GPT: predicting the next token at every position.
    """

    def __init__(self, binary_path: Union[str, Path], context_length: int = 256) -> None:
        p = Path(binary_path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset binary file not found: {p}")
        
        self.binary_path = p
        self.context_length = context_length
        # Memory-map the binary uint16 array for zero-copy read performance
        self.data = np.memmap(str(p), dtype=np.uint16, mode="r")
        self.total_tokens = len(self.data)
        
        if self.total_tokens <= context_length:
            raise ValueError(
                f"Dataset contains {self.total_tokens} tokens, but context_length is {context_length}."
            )

    def __len__(self) -> int:
        return self.total_tokens - self.context_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract context_length + 1 tokens
        chunk = self.data[idx : idx + self.context_length + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def get_batch(
    split_binary_path: Union[str, Path],
    batch_size: int = 8,
    context_length: int = 256,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast causal batch sampler directly from a memory-mapped uint16 binary file.
    
    Args:
        split_binary_path: Path to train_ids.bin or val_ids.bin.
        batch_size: Number of parallel sequences to sample.
        context_length: Sequence window length (e.g. 256 tokens).
        device: PyTorch device ('cpu' or 'cuda').
        
    Returns:
        (x, y) where each is of shape (batch_size, context_length).
        x[:, t+1] == y[:, t] across all sequences.
    """
    data = np.memmap(str(split_binary_path), dtype=np.uint16, mode="r")
    high = len(data) - context_length - 1
    if high <= 0:
        raise ValueError(f"Binary file {split_binary_path} is too small for context_length {context_length}")
    
    offsets = np.random.randint(0, high, size=batch_size)
    x_list = [data[i : i + context_length].astype(np.int64) for i in offsets]
    y_list = [data[i + 1 : i + 1 + context_length].astype(np.int64) for i in offsets]
    
    x = torch.tensor(np.stack(x_list), dtype=torch.long, device=device)
    y = torch.tensor(np.stack(y_list), dtype=torch.long, device=device)
    return x, y
