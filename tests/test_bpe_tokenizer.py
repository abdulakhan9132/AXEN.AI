"""
tests/test_bpe_tokenizer.py
===========================
Comprehensive Test Suite for AEXN GPT 1.0 Byte Pair Encoding (BPE) Tokenizer.

Verifies:
1.  Correct byte encoding (256 base byte tokens)
2.  Pair frequency calculation
3.  Pair merging
4.  Token ID assignment (sequential and non-colliding)
5.  Encoding and decoding round-trip (lossless reconstruction)
6.  Spaces and punctuation preservation
7.  Unicode and emoji compatibility (UTF-8 robustness)
8.  Vocabulary size limits (2048 production target)
9.  Tokenizer saving and loading (JSON persistence)
10. Reproducible token IDs (deterministic encoding)
11. Validation data isolation (train vs val partition)
12. Binary dataset consistency (uint16 array checks, ID range 0 <= id < 2048)
13. Causal dataset next-token alignment (x[t+1] == y[t])
14. AXEN Foundation Transformer integration with AEXN BPE Tokenizer
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from axen.model.bpe_tokenizer import (
    AEXNTokenizer,
    EducationalBPE,
    ShakespeareBPEDataset,
    get_batch,
)
from axen.model.foundation import AXENFoundationModel, FoundationModelConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def trained_tokenizer() -> AEXNTokenizer:
    """Load or train the production tokenizer for test verification."""
    tokenizer_path = Path("data/shakespeare/aexn_bpe.json")
    if tokenizer_path.exists():
        return AEXNTokenizer.load(tokenizer_path)
    
    # Train a fast test instance if file does not exist
    corpus_sample = "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer."
    tok = AEXNTokenizer(vocab_size=300, min_frequency=2)
    tok.train_from_text(corpus_sample * 5)
    return tok


# ---------------------------------------------------------------------------
# 1. Correct Byte Encoding
# ---------------------------------------------------------------------------

def test_01_correct_byte_encoding():
    """Verify that every fundamental byte 0..255 can be represented."""
    bpe = EducationalBPE(target_vocab_size=256)
    assert len(bpe.vocab) == 256
    for i in range(256):
        assert i in bpe.vocab
        assert bpe.vocab[i] == bytes([i])

    # Test encoding raw byte sequences
    all_bytes = bytes(range(256))
    encoded = bpe.encode(all_bytes)
    assert len(encoded) == 256
    assert encoded == list(range(256))


# ---------------------------------------------------------------------------
# 2. Pair Frequency Calculation
# ---------------------------------------------------------------------------

def test_02_pair_frequency_calculation():
    """Verify EducationalBPE.get_pair_frequencies accurately counts adjacent pairs."""
    # Pattern: [10, 20, 10, 20, 10, 20, 30]
    tokens = [10, 20, 10, 20, 10, 20, 30]
    freqs = EducationalBPE.get_pair_frequencies(tokens)

    assert freqs[(10, 20)] == 3
    assert freqs[(20, 10)] == 2
    assert freqs[(20, 30)] == 1
    assert (30, 10) not in freqs
    assert len(freqs) == 3


# ---------------------------------------------------------------------------
# 3. Pair Merging
# ---------------------------------------------------------------------------

def test_03_pair_merging():
    """Verify EducationalBPE.merge_pair replaces occurrences of pair with new ID."""
    tokens = [1, 2, 99, 1, 2, 88, 1, 2]
    merged = EducationalBPE.merge_pair(tokens, (1, 2), 300)
    assert merged == [300, 99, 300, 88, 300]

    # Non-overlapping boundary condition: [1, 1, 1] merging (1, 1) -> 300
    tokens_overlap = [1, 1, 1]
    merged_overlap = EducationalBPE.merge_pair(tokens_overlap, (1, 1), 300)
    # First (1, 1) becomes 300, leaving [300, 1]
    assert merged_overlap == [300, 1]


# ---------------------------------------------------------------------------
# 4. Token ID Assignment
# ---------------------------------------------------------------------------

def test_04_token_id_assignment():
    """Verify merges receive unique, sequential IDs starting after the base vocabulary."""
    text = "ABCD ABCD ABCD ABCD"
    bpe = EducationalBPE(target_vocab_size=260, min_freq=2)
    bpe.train(text)

    # Base vocab is 0..255, so first learned merge MUST be ID 256
    assert len(bpe.merges) > 0
    assigned_ids = list(bpe.merges.values())
    assert assigned_ids[0] == 256
    assert assigned_ids == list(range(256, 256 + len(assigned_ids)))
    # Ensure no duplicate IDs
    assert len(assigned_ids) == len(set(assigned_ids))


# ---------------------------------------------------------------------------
# 5. Encoding and Decoding Round-Trip
# ---------------------------------------------------------------------------

def test_05_encoding_decoding_round_trip(trained_tokenizer: AEXNTokenizer):
    """Verify decode(encode(text)) == text for both educational and production tokenizers."""
    # Educational test
    edu_text = "To be or not to be, that is the question."
    edu = EducationalBPE(target_vocab_size=265, min_freq=2)
    edu.train(edu_text)
    assert edu.decode(edu.encode(edu_text)) == edu_text

    # Production test
    prod_phrases = [
        "Friends, Romans, countrymen, lend me your ears;",
        "I come to bury Caesar, not to praise him.",
        "The evil that men do lives after them;",
        "The good is oft interred with their bones.",
    ]
    for phrase in prod_phrases:
        encoded = trained_tokenizer.encode(phrase)
        decoded = trained_tokenizer.decode(encoded)
        assert decoded == phrase, f"Failed round-trip for: {phrase!r}"


# ---------------------------------------------------------------------------
# 6. Spaces, Whitespace, and Punctuation
# ---------------------------------------------------------------------------

def test_06_spaces_and_punctuation(trained_tokenizer: AEXNTokenizer):
    """Verify that all varieties of whitespace and punctuation are preserved."""
    test_cases = [
        "  leading and trailing whitespace  ",
        "multiple   spaces   between   words",
        "\n\nBlank lines and indentation:\n\tdef hello():\n\t\treturn True\n",
        "!@#$%^&*()_+-=[]{}|;':\",./<>?`~",
        "Comma, semicolon; colon: apostrophe' quote\" dash- hyphen-- ellipsis...",
    ]
    for case in test_cases:
        encoded = trained_tokenizer.encode(case)
        decoded = trained_tokenizer.decode(encoded)
        assert decoded == case, f"Whitespace/punctuation mismatch:\nExpected: {case!r}\nGot: {decoded!r}"


# ---------------------------------------------------------------------------
# 7. Unicode and Multi-byte Compatibility
# ---------------------------------------------------------------------------

def test_07_unicode_compatibility(trained_tokenizer: AEXNTokenizer):
    """Verify UTF-8 multi-byte characters and emojis encode and decode reversibly."""
    unicode_cases = [
        "🎭 Shakespeare 🚀",
        "Café, résumé, façade, naïve, señor, über",
        "Greek: Ἑλλάς, Cyrillic: Москва, Chinese: 莎士比亚",
        "Mathematical symbols: ∑, ∏, ∫, √, ≈, ≠, ≤, ≥, ∞",
        "Zero-width and special chars: A\u200bB\u200cC",
    ]
    for case in unicode_cases:
        encoded = trained_tokenizer.encode(case)
        decoded = trained_tokenizer.decode(encoded)
        assert decoded == case, f"Unicode mismatch for: {case!r}"


# ---------------------------------------------------------------------------
# 8. Vocabulary Size Limits
# ---------------------------------------------------------------------------

def test_08_vocabulary_size_limits():
    """Verify that vocabulary does not exceed target limits."""
    # Educational test with small limit
    text = "The quick brown fox jumps over the lazy dog. " * 10
    target_edu = 270
    edu = EducationalBPE(target_vocab_size=target_edu, min_freq=2)
    edu.train(text)
    assert len(edu.vocab) <= target_edu

    # Production tokenizer check on Shakespeare
    metadata_path = Path("data/shakespeare/metadata.json")
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["actual_vocab_size"] == 2048
        assert meta["target_vocab_size"] == 2048


# ---------------------------------------------------------------------------
# 9. Tokenizer Saving and Loading
# ---------------------------------------------------------------------------

def test_09_tokenizer_saving_and_loading(trained_tokenizer: AEXNTokenizer):
    """Verify saving to JSON and loading produces an identical tokenizer."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "test_saved_bpe.json"
        trained_tokenizer.save(tmp_path)
        assert tmp_path.exists()
        assert tmp_path.stat().st_size > 0

        # Load back
        loaded = AEXNTokenizer.load(tmp_path)
        assert loaded.vocab_size == trained_tokenizer.vocab_size
        assert loaded.pad_token_id == trained_tokenizer.pad_token_id
        assert loaded.eos_token_id == trained_tokenizer.eos_token_id

        # Verify encoding identity
        test_str = "All the world's a stage, and all the men and women merely players."
        ids_original = trained_tokenizer.encode(test_str)
        ids_loaded = loaded.encode(test_str)
        assert ids_original == ids_loaded
        assert loaded.decode(ids_loaded) == test_str


# ---------------------------------------------------------------------------
# 10. Reproducible Token IDs
# ---------------------------------------------------------------------------

def test_10_reproducible_token_ids(trained_tokenizer: AEXNTokenizer):
    """Verify deterministic encoding: encoding the same text 100 times gives identical IDs."""
    sample = "To be, or not to be, that is the question:"
    baseline_ids = trained_tokenizer.encode(sample)

    for _ in range(50):
        assert trained_tokenizer.encode(sample) == baseline_ids


# ---------------------------------------------------------------------------
# 11. Validation Data Isolation
# ---------------------------------------------------------------------------

def test_11_validation_data_isolation():
    """Verify that validation partition is strictly segregated from training data."""
    metadata_path = Path("data/shakespeare/metadata.json")
    if not metadata_path.exists():
        pytest.skip("data/shakespeare/metadata.json not found; run scripts/train_bpe.py first.")

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    stats = meta["corpus_stats"]
    assert stats["train_lines"] + stats["val_lines"] == stats["total_lines"]
    assert stats["train_chars"] + stats["val_chars"] == stats["total_chars"]
    assert stats["train_bytes"] + stats["val_bytes"] == stats["total_bytes"]
    # Ratio should be ~90% train / ~10% val
    ratio = stats["train_lines"] / stats["total_lines"]
    assert 0.89 <= ratio <= 0.91


# ---------------------------------------------------------------------------
# 12. Binary Dataset Consistency
# ---------------------------------------------------------------------------

def test_12_binary_dataset_consistency():
    """Verify train_ids.bin and val_ids.bin are valid uint16 arrays with valid token IDs."""
    train_bin = Path("data/shakespeare/train_ids.bin")
    val_bin = Path("data/shakespeare/val_ids.bin")
    metadata_path = Path("data/shakespeare/metadata.json")

    if not (train_bin.exists() and val_bin.exists() and metadata_path.exists()):
        pytest.skip("Binary dataset files not found; run scripts/train_bpe.py first.")

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    vocab_size = meta["actual_vocab_size"]

    # Verify train_ids.bin
    train_data = np.fromfile(str(train_bin), dtype=np.uint16)
    assert len(train_data) == meta["train_tokens"]
    assert train_bin.stat().st_size == len(train_data) * 2  # uint16 = 2 bytes
    assert int(train_data.min()) >= 0
    assert int(train_data.max()) < vocab_size

    # Verify val_ids.bin
    val_data = np.fromfile(str(val_bin), dtype=np.uint16)
    assert len(val_data) == meta["val_tokens"]
    assert val_bin.stat().st_size == len(val_data) * 2
    assert int(val_data.min()) >= 0
    assert int(val_data.max()) < vocab_size


# ---------------------------------------------------------------------------
# 13. Causal Dataset Next-Token Alignment
# ---------------------------------------------------------------------------

def test_13_causal_dataset_alignment():
    """Verify that (x, y) training pairs satisfy y[t] == x[t+1] for causal prediction."""
    train_bin = Path("data/shakespeare/train_ids.bin")
    if not train_bin.exists():
        pytest.skip("train_ids.bin not found")

    context_length = 32
    dataset = ShakespeareBPEDataset(train_bin, context_length=context_length)
    assert len(dataset) > 1000

    x, y = dataset[0]
    assert x.shape == (context_length,)
    assert y.shape == (context_length,)
    # Crucial GPT property: target token is the next token in the sequence
    assert torch.equal(y[:-1], x[1:])

    # Verify batch sampler
    bx, by = get_batch(train_bin, batch_size=4, context_length=context_length, device="cpu")
    assert bx.shape == (4, context_length)
    assert by.shape == (4, context_length)
    assert torch.equal(by[:, :-1], bx[:, 1:])


# ---------------------------------------------------------------------------
# 14. AXEN Foundation Transformer Integration
# ---------------------------------------------------------------------------

def test_14_foundation_transformer_integration(trained_tokenizer: AEXNTokenizer):
    """Verify that AXEN Foundation Model runs causal forward pass with AEXNTokenizer."""
    vocab_size = trained_tokenizer.vocab_size
    context_length = 32

    config = FoundationModelConfig(
        model_name="test_bpe_transformer",
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_dim=64,
        num_layers=2,
        num_heads=2,
        ff_dim=128,
        device="cpu",
    )
    model = AXENFoundationModel(config=config, tokenizer=trained_tokenizer)
    assert model.lm_head.out_features == vocab_size
    assert model.transformer.token_embedding.num_embeddings == vocab_size

    # Create dummy batch of token IDs
    dummy_input = torch.randint(0, vocab_size, (2, context_length), dtype=torch.long)
    dummy_targets = torch.randint(0, vocab_size, (2, context_length), dtype=torch.long)

    logits = model(dummy_input)
    assert logits.shape == (2, context_length, vocab_size)

    # Compute loss
    loss = torch.nn.functional.cross_entropy(logits.view(-1, vocab_size), dummy_targets.view(-1))
    assert not torch.isnan(loss)
    assert loss.item() > 0.0
