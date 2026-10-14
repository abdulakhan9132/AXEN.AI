"""
scripts/finetune_aexn_gpt.py
============================
Instruction Fine-Tuning Pipeline for AEXN GPT 1.0.

Key Capabilities:
1. Loads pretrained AEXN GPT base checkpoint (checkpoints/aexn_gpt/best_model.pt).
2. Uses approved high-quality instruction-response conversational pairs.
3. Implements prompt/response loss masking (ignore_index=-100):
   - Computes cross-entropy loss exclusively on the ASSISTANT tokens.
   - Prevents the model from memorizing user queries and focuses gradients on helpful answers.
4. Uses conservative learning rate (5e-5) with AdamW to avoid catastrophic forgetting.
5. Evaluates conversational generation before and after instruction fine-tuning.
6. Saves instruction-tuned model to checkpoints/aexn_gpt/aexn_gpt_instruct.pt.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from axen.model.bpe_tokenizer import AEXNTokenizer
from axen.model.dataset import CURATED_CAPABILITY_SEEDS
from axen.model.foundation import AXENFoundationModel, FoundationModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finetune_aexn_gpt")

APPROVED_INSTRUCTION_EXAMPLES: List[Dict[str, str]] = [
    {"user": "Hello, who are you?", "assistant": "I am AEXN GPT 1.0, your personal local neural assistant."},
    {"user": "What can you do?", "assistant": "I can answer questions, reason through problems, write code, and assist your workflows."},
    {"user": "Who created you?", "assistant": "I was trained from scratch as the native brain of the AEXN AI project."},
    {"user": "What is 2 + 2?", "assistant": "2 + 2 equals 4."},
    {"user": "Can you explain BPE tokenization?", "assistant": "Byte Pair Encoding converts text into numbers by merging frequent adjacent byte pairs into single subword tokens."},
    {"user": "How does a Transformer learn?", "assistant": "It uses causal self-attention and feed-forward networks to predict the next token given previous context."},
    {"user": "Write a short poem about time.", "assistant": "Time flows on like a silent stream, turning tomorrow into a dream."},
    {"user": "What is Python?", "assistant": "Python is a versatile, high-level programming language widely used in AI, data science, and web applications."},
]


def prepare_masked_example(
    tokenizer: AEXNTokenizer,
    user_prompt: str,
    assistant_response: str,
    context_length: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Format and tokenize example with prompt loss masking.
    Format:
      USER: <prompt>\nASSISTANT: <response><EOS>
    
    Target labels:
      - Prompt tokens are marked with -100 (ignored by PyTorch cross_entropy)
      - Response tokens are preserved for next-token prediction
    """
    prompt_str = f"USER: {user_prompt.strip()}\nASSISTANT: "
    full_str = f"{prompt_str}{assistant_response.strip()}"

    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    full_ids = tokenizer.encode(full_str, add_special_tokens=False) + [tokenizer.eos_token_id]

    # Truncate or pad to context_length + 1
    if len(full_ids) > context_length + 1:
        full_ids = full_ids[: context_length + 1]

    # Input x and Target y
    x_ids = full_ids[:-1]
    y_ids = full_ids[1:]

    # Mask prompt positions in target y with -100
    prompt_len = min(len(prompt_ids) - 1, len(y_ids))
    targets = []
    for i, token in enumerate(y_ids):
        if i < prompt_len:
            targets.append(-100)  # Mask out user prompt
        else:
            targets.append(token)  # Supervise assistant output

    # Pad if shorter than context_length
    pad_len = context_length - len(x_ids)
    if pad_len > 0:
        x_ids = x_ids + [tokenizer.pad_token_id] * pad_len
        targets = targets + [-100] * pad_len

    x = torch.tensor(x_ids[:context_length], dtype=torch.long)
    y = torch.tensor(targets[:context_length], dtype=torch.long)
    return x, y


def finetune(
    base_checkpoint_path: Path,
    output_checkpoint_path: Path,
    tokenizer_path: Path,
    learning_rate: float = 0.00005,
    epochs: int = 15,
    device_str: str = "auto",
) -> Dict[str, Any]:
    """Run supervised instruction fine-tuning on approved conversational dataset."""
    if not base_checkpoint_path.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {base_checkpoint_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    dev = torch.device("cuda" if (device_str == "auto" and torch.cuda.is_available()) else device_str if device_str != "auto" else "cpu")
    tokenizer = AEXNTokenizer.load(tokenizer_path)

    # Load base model
    ckpt = torch.load(base_checkpoint_path, map_location=dev, weights_only=False)
    cfg = FoundationModelConfig.from_dict(ckpt.get("config", {}))
    cfg.device = str(dev)

    model = AXENFoundationModel(config=cfg, tokenizer=tokenizer)
    model.load_state_dict(ckpt["model_state"])
    model.to(dev)

    print("=" * 78)
    print("AEXN GPT 1.0 — INSTRUCTION FINE-TUNING")
    print(f"Base Checkpoint : {base_checkpoint_path}")
    print(f"Device          : {dev}")
    print(f"Learning Rate   : {learning_rate} (conservative to preserve foundation)")
    print("=" * 78)

    # Baseline evaluation before fine-tuning
    test_user_query = "Hello, who are you?"
    print("\n[Conversational Response Before Instruction Tuning]")
    before_resp = model.chat([{"role": "user", "content": test_user_query}], max_tokens=30, temperature=0.7)
    print(f"User: {test_user_query}")
    print(f"AEXN GPT (Pretrained Base): {before_resp!r}")

    # Build dataset
    all_examples = []
    for ex in APPROVED_INSTRUCTION_EXAMPLES:
        all_examples.append((ex["user"], ex["assistant"]))
    for seed in CURATED_CAPABILITY_SEEDS[:12]:
        all_examples.append((seed.get("input", ""), seed.get("target", "")))

    dataset_tensors = [
        prepare_masked_example(tokenizer, u, a, context_length=cfg.context_length)
        for u, a in all_examples if u and a
    ]
    print(f"\nPrepared {len(dataset_tensors)} instruction-response examples with prompt loss masking.")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    model.train()

    t0 = time.time()
    losses = []

    for epoch in range(epochs):
        epoch_losses = []
        # Shuffle dataset each epoch
        indices = list(range(len(dataset_tensors)))
        random.shuffle(indices)

        for idx in indices:
            x, y = dataset_tensors[idx]
            x = x.unsqueeze(0).to(dev)
            y = y.unsqueeze(0).to(dev)

            optimizer.zero_grad()
            logits = model(x)

            # Compute loss only on unmasked assistant tokens
            loss = torch.nn.functional.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1), ignore_index=-100)
            if not torch.isnan(loss) and loss.item() > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        losses.append(mean_loss)
        if (epoch + 1) % 3 == 0 or (epoch + 1) == epochs:
            print(f"Epoch {epoch+1:2d}/{epochs} | Response Loss: {mean_loss:.4f}")

    duration = time.time() - t0
    print(f"\nFine-tuning completed in {duration:.2f} seconds. Final Response Loss: {losses[-1]:.4f}")

    # Save fine-tuned checkpoint
    output_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": ckpt.get("step", 0) + epochs,
        "model_state": model.state_dict(),
        "config": cfg.to_dict(),
        "fine_tune_type": "instruction_sft",
        "final_loss": losses[-1],
    }
    torch.save(payload, output_checkpoint_path)
    print(f"Saved Instruction Checkpoint: {output_checkpoint_path}")

    # Conversational test after fine-tuning
    print("\n[Conversational Response After Instruction Tuning]")
    after_resp = model.chat([{"role": "user", "content": test_user_query}], max_tokens=30, temperature=0.7)
    print(f"User: {test_user_query}")
    print(f"AEXN GPT (Instruction Tuned): {after_resp!r}")

    return {
        "epochs": epochs,
        "duration_sec": duration,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "before_response": before_resp,
        "after_response": after_resp,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Instruction Fine-Tune AEXN GPT 1.0")
    parser.add_argument("--base-checkpoint", type=str, default="checkpoints/aexn_gpt/best_model.pt")
    parser.add_argument("--output-checkpoint", type=str, default="checkpoints/aexn_gpt/aexn_gpt_instruct.pt")
    parser.add_argument("--tokenizer", type=str, default="data/shakespeare/aexn_bpe.json")
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    base_ckpt = PROJECT_ROOT / args.base_checkpoint
    if not base_ckpt.exists() and (base_ckpt.parent / "checkpoint_latest.pt").exists():
        base_ckpt = base_ckpt.parent / "checkpoint_latest.pt"

    out_ckpt = PROJECT_ROOT / args.output_checkpoint
    tok_path = PROJECT_ROOT / args.tokenizer

    finetune(
        base_checkpoint_path=base_ckpt,
        output_checkpoint_path=out_ckpt,
        tokenizer_path=tok_path,
        learning_rate=args.lr,
        epochs=args.epochs,
        device_str=args.device,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
