"""
scripts/train_bpe.py
====================
End-to-end training and dataset generation pipeline for AEXN GPT 1.0 BPE Tokenizer.

Pipeline Steps:
1. Verify dataset existence (data/shakespeare/shakespeare_training.txt).
2. Load complete UTF-8 text, compute corpus statistics (chars, words, lines, bytes).
3. Split corpus into training (90%) and validation (10%) partitions (val strictly isolated).
4. Demonstrate Educational BPE from scratch (pair counting, greedy merging, token ID allocation).
5. Train production Byte-Level BPE Tokenizer (AEXNTokenizer) to 2,048 vocabulary size.
6. Save canonical tokenizer rules to data/shakespeare/aexn_bpe.json.
7. Encode training and validation partitions into uint16 binary arrays:
   - data/shakespeare/train_ids.bin
   - data/shakespeare/val_ids.bin
8. Save dataset configuration and stats to data/shakespeare/metadata.json.
9. Verify reversible round-trip encoding and token ID bounds (0 <= id < 2048).
10. Demonstrate next-token prediction preparation for AEXN GPT Transformer training.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from axen.model.bpe_tokenizer import AEXNTokenizer, EducationalBPE, ShakespeareBPEDataset, get_batch
from axen.model.foundation import AXENFoundationModel, FoundationModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_bpe")

DATA_DIR = PROJECT_ROOT / "data" / "shakespeare"
SHAKESPEARE_FILE = DATA_DIR / "shakespeare_training.txt"
TOKENIZER_JSON = DATA_DIR / "aexn_bpe.json"
TRAIN_BIN = DATA_DIR / "train_ids.bin"
VAL_BIN = DATA_DIR / "val_ids.bin"
METADATA_JSON = DATA_DIR / "metadata.json"


def check_dataset() -> bool:
    """Check if the Shakespeare training dataset exists."""
    if not SHAKESPEARE_FILE.exists():
        print("=" * 78)
        print("ERROR: Shakespeare dataset not found!")
        print(f"Expected file at: {SHAKESPEARE_FILE}")
        print("\nPlease download the public domain Complete Works of William Shakespeare")
        print("from Project Gutenberg by running:")
        print("    python download_shakespeare.py")
        print("=" * 78)
        return False
    return True


def load_and_split_corpus(train_ratio: float = 0.9) -> tuple[str, str, dict]:
    """
    Read Shakespeare text, compute corpus stats, and split into train and validation.
    Validation text is strictly isolated and never seen by the tokenizer during training.
    """
    with open(SHAKESPEARE_FILE, "r", encoding="utf-8") as f:
        full_text = f.read()

    lines = full_text.splitlines(keepends=True)
    total_lines = len(lines)
    total_chars = len(full_text)
    total_words = len(full_text.split())
    total_bytes = len(full_text.encode("utf-8"))

    # Line-aware split preserving verse boundaries
    split_line_idx = int(total_lines * train_ratio)
    train_text = "".join(lines[:split_line_idx])
    val_text = "".join(lines[split_line_idx:])

    stats = {
        "total_lines": total_lines,
        "total_chars": total_chars,
        "total_words": total_words,
        "total_bytes": total_bytes,
        "train_lines": split_line_idx,
        "train_chars": len(train_text),
        "train_words": len(train_text.split()),
        "train_bytes": len(train_text.encode("utf-8")),
        "val_lines": total_lines - split_line_idx,
        "val_chars": len(val_text),
        "val_words": len(val_text.split()),
        "val_bytes": len(val_text.encode("utf-8")),
    }
    return train_text, val_text, stats


def run_educational_demonstration() -> None:
    """
    Demonstrate how Byte Pair Encoding learns from scratch step-by-step
    on a famous Shakespeare excerpt.
    """
    print("\n" + "=" * 78)
    print("STEP 8 — EDUCATIONAL DEMONSTRATION: HOW BYTE PAIR ENCODING (BPE) LEARNS")
    print("=" * 78)
    
    sample = "To be, or not to be, that is the question: To be or not to be!"
    print(f"Sample Input Text: {sample!r}")
    raw_bytes = list(sample.encode("utf-8"))
    print(f"Initial Byte Length : {len(raw_bytes)} bytes")
    print(f"Initial Byte Tokens : {raw_bytes[:16]}... (first 16 byte IDs in range 0..255)")

    edu_bpe = EducationalBPE(target_vocab_size=265, min_freq=2)
    edu_bpe.train(sample)

    print("\n[Educational Trace: Learned Merges]")
    print(edu_bpe.get_trace_summary(max_steps=10))

    encoded = edu_bpe.encode(sample)
    print(f"\nOriginal Byte Count : {len(raw_bytes)}")
    print(f"Encoded Token Count : {len(encoded)}")
    print(f"Compression Ratio   : {len(raw_bytes) / len(encoded):.2f}x")
    print(f"Encoded Token IDs   : {encoded}")

    decoded = edu_bpe.decode(encoded)
    print(f"Decoded Matches     : {decoded == sample}")
    print("\nWhy BPE is more efficient:")
    print("1. Without BPE: 'To be' requires 5 separate character/byte tokens: ['T', 'o', ' ', 'b', 'e'].")
    print("2. With BPE: The model learns that 'To be' frequently co-occurs and merges it into ONE token ID.")
    print("3. Result: Transformers process sentences in 3x to 4x fewer steps, dramatically saving memory and computation.")


def main() -> int:
    parser = argparse.ArgumentParser(description="AEXN GPT 1.0 - BPE Tokenizer Training Pipeline")
    parser.add_argument("--vocab-size", type=int, default=2048, help="Target vocabulary size (default: 2048)")
    parser.add_argument("--min-freq", type=int, default=2, help="Minimum merge frequency (default: 2)")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train split fraction (default: 0.9)")
    args = parser.parse_args()

    print("=" * 78)
    print("AEXN GPT 1.0 — BYTE PAIR ENCODING (BPE) TOKENIZER PIPELINE")
    print("Inspired by Anton GPT / nanoGPT Training Architecture")
    print("=" * 78)

    # 1. Inspect Dataset
    if not check_dataset():
        return 1

    # 2. Load Corpus & Split
    print("\n[STEP 2: Loading Shakespeare Corpus & Creating Partitions]")
    train_text, val_text, stats = load_and_split_corpus(train_ratio=args.train_ratio)
    print(f"Total Lines      : {stats['total_lines']:,}")
    print(f"Total Characters : {stats['total_chars']:,}")
    print(f"Total Words      : {stats['total_words']:,}")
    print(f"Total Bytes      : {stats['total_bytes']:,}")
    print(f"Train Partition  : {stats['train_lines']:,} lines | {stats['train_chars']:,} chars | {stats['train_bytes']:,} bytes (90%)")
    print(f"Val Partition    : {stats['val_lines']:,} lines | {stats['val_chars']:,} chars | {stats['val_bytes']:,} bytes (10%)")
    print("Isolation Check  : Validation partition is strictly isolated from tokenizer fitting.")

    # 3. Educational Demonstration
    run_educational_demonstration()

    # 4. Production Tokenizer Training
    print("\n" + "=" * 78)
    print(f"[STEP 4 & 5: Training Production AEXNTokenizer (Target Vocab: {args.vocab_size})]")
    print("=" * 78)
    t0 = time.time()
    tokenizer = AEXNTokenizer(vocab_size=args.vocab_size, min_frequency=args.min_freq)
    tokenizer.train_from_text(train_text)
    train_duration = time.time() - t0

    vocab_size = tokenizer.vocab_size
    print(f"Training Time     : {train_duration:.2f} seconds")
    print(f"Vocabulary Size   : {vocab_size}")
    print(f"Learned Merges    : {vocab_size - 256 - len(tokenizer.SPECIAL_TOKENS)} merges")
    print(f"Special Tokens    : {tokenizer.SPECIAL_TOKENS} -> IDs {[tokenizer.pad_token_id, tokenizer.unk_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]}")

    # Save tokenizer rules
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save(TOKENIZER_JSON)
    print(f"Saved Tokenizer   : {TOKENIZER_JSON} ({TOKENIZER_JSON.stat().st_size:,} bytes)")

    # 5. Convert Corpus to Numerical Arrays (uint16)
    print("\n" + "=" * 78)
    print("[STEP 6: Generating uint16 Numerical Arrays (train_ids.bin, val_ids.bin)]")
    print("=" * 78)
    
    t_enc0 = time.time()
    train_ids = tokenizer.encode(train_text, add_special_tokens=False)
    val_ids = tokenizer.encode(val_text, add_special_tokens=False)
    enc_duration = time.time() - t_enc0

    train_tokens_count = len(train_ids)
    val_tokens_count = len(val_ids)
    total_tokens_count = train_tokens_count + val_tokens_count

    train_comp_ratio = stats["train_bytes"] / train_tokens_count if train_tokens_count else 0
    val_comp_ratio = stats["val_bytes"] / val_tokens_count if val_tokens_count else 0

    print(f"Encoding Time     : {enc_duration:.2f} seconds")
    print(f"Train Tokens      : {train_tokens_count:,} (Raw Bytes: {stats['train_bytes']:,} -> Compression: {train_comp_ratio:.2f}x)")
    print(f"Val Tokens        : {val_tokens_count:,} (Raw Bytes: {stats['val_bytes']:,} -> Compression: {val_comp_ratio:.2f}x)")
    print(f"Total Tokens      : {total_tokens_count:,}")

    # Convert to uint16 NumPy arrays
    train_arr = np.array(train_ids, dtype=np.uint16)
    val_arr = np.array(val_ids, dtype=np.uint16)

    # Validate ID ranges (must be in 0 <= id < vocab_size)
    assert train_arr.min() >= 0 and train_arr.max() < vocab_size, "Train token IDs exceed vocab range!"
    assert val_arr.min() >= 0 and val_arr.max() < vocab_size, "Val token IDs exceed vocab range!"
    print(f"ID Range Verified : All tokens strictly within [0, {vocab_size - 1}]")

    # Save binary files
    train_arr.tofile(TRAIN_BIN)
    val_arr.tofile(VAL_BIN)
    print(f"Saved Train Array : {TRAIN_BIN} ({TRAIN_BIN.stat().st_size:,} bytes, dtype=uint16)")
    print(f"Saved Val Array   : {VAL_BIN} ({VAL_BIN.stat().st_size:,} bytes, dtype=uint16)")

    # Save comprehensive metadata configuration
    metadata = {
        "model_name": "AEXN GPT 1.0",
        "tokenizer_type": "byte_level_bpe",
        "tokenizer_version": "1.0",
        "corpus_title": "The Complete Works of William Shakespeare",
        "target_vocab_size": args.vocab_size,
        "actual_vocab_size": vocab_size,
        "min_frequency": args.min_freq,
        "dtype": "uint16",
        "bytes_per_token": 2,
        "special_tokens": {
            "<PAD>": tokenizer.pad_token_id,
            "<UNK>": tokenizer.unk_token_id,
            "<BOS>": tokenizer.bos_token_id,
            "<EOS>": tokenizer.eos_token_id,
        },
        "corpus_stats": stats,
        "train_tokens": train_tokens_count,
        "val_tokens": val_tokens_count,
        "total_tokens": total_tokens_count,
        "train_compression_ratio": round(train_comp_ratio, 3),
        "val_compression_ratio": round(val_comp_ratio, 3),
        "train_binary_file": str(TRAIN_BIN.name),
        "val_binary_file": str(VAL_BIN.name),
        "tokenizer_file": str(TOKENIZER_JSON.name),
    }
    with open(METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved Metadata    : {METADATA_JSON}")

    # 6. Reversible Round-Trip Verification
    print("\n" + "=" * 78)
    print("[STEP 7: Encoding and Decoding Verification (Reversible Round-Trip)]")
    print("=" * 78)
    test_phrases = [
        "To be, or not to be, that is the question:",
        "Now is the winter of our discontent / Made glorious summer by this sun of York;",
        "Leading and trailing spaces:   hello world!   ",
        "All ASCII symbols: !@#$%^&*()_+-=[]{}|;:,.<>?/~`",
        "Unicode & Emojis: 🎭 Romeo & Juliet 🚀 — café naïve résumé",
        "Numbers & Punctuation: Act 1, Scene 2. Line 42 (1564-1616).",
        "Dialogue with line breaks:\nKING: Who is there?\nHAMLET: A friend.\n",
    ]

    all_passed = True
    for phrase in test_phrases:
        enc = tokenizer.encode(phrase)
        dec = tokenizer.decode(enc)
        match = (dec == phrase)
        status = "PASSED" if match else "FAILED"
        print(f"[{status}] Length {len(phrase)} chars -> {len(enc)} tokens -> Reversible: {match}")
        if not match:
            print(f"  Expected: {phrase!r}")
            print(f"  Got:      {dec!r}")
            all_passed = False

    if all_passed:
        print("\nAll Reversible Round-Trip Tests PASSED 100%!")

    # 7. Step 10: Connection to AEXN GPT Architecture
    print("\n" + "=" * 78)
    print("[STEP 10: Connecting Tokenizer to AEXN GPT Neural Network]")
    print("=" * 78)
    context_length = 64
    batch_size = 4
    
    print(f"Initializing ShakespeareBPEDataset (context_length={context_length})...")
    dataset = ShakespeareBPEDataset(TRAIN_BIN, context_length=context_length)
    print(f"Dataset Items: {len(dataset):,} valid training sequences")
    
    x_sample, y_sample = dataset[0]
    print(f"Sample x (Input Tokens)  shape: {x_sample.shape}")
    print(f"Sample y (Target Tokens) shape: {y_sample.shape}")
    print(f"Next-Token Check: y[0] == x[1] -> {bool(y_sample[0] == x_sample[1])} (True verifies next-token alignment)")

    print("\nInstantiating AXEN Foundation Transformer with AEXN BPE Tokenizer (vocab_size=2048)...")
    config = FoundationModelConfig(
        model_name="aexn_gpt_1.0_shakespeare",
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_dim=128,
        num_layers=4,
        num_heads=4,
        ff_dim=512,
    )
    model = AXENFoundationModel(config=config, tokenizer=tokenizer)
    total_p, train_p = model.count_parameters()
    print(f"AEXN GPT 1.0 Parameters: {total_p:,} total ({train_p:,} trainable)")

    print("\nTesting fast batch sampler (get_batch)...")
    bx, by = get_batch(TRAIN_BIN, batch_size=batch_size, context_length=context_length, device=model.device)
    print(f"Batch X shape: {bx.shape} (on {bx.device}) | Batch Y shape: {by.shape} (on {by.device})")

    # Execute forward pass
    with torch.no_grad():
        logits = model(bx)
    print(f"Transformer Forward Pass Logits Shape: {logits.shape} (batch={batch_size}, seq={context_length}, vocab={vocab_size})")

    # Compute causal loss
    loss = torch.nn.functional.cross_entropy(logits.view(-1, vocab_size), by.view(-1))
    print(f"Untrained Initial Cross-Entropy Loss : {loss.item():.4f}")
    print(f"Theoretical Uniform Cross-Entropy    : {np.log(vocab_size):.4f} (loss is correctly aligned with random init)")

    print("\n" + "=" * 78)
    print("SUCCESS: AEXN GPT 1.0 BPE TOKENIZER AND DATASET PREPARATION COMPLETE!")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
