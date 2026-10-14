"""
scripts/generate_aexn_text.py
=============================
Interactive and CLI Autoregressive Text Generation for AEXN GPT 1.0.

Features:
1. Loads trained model checkpoint (checkpoints/aexn_gpt/best_model.pt).
2. Loads BPE Tokenizer (data/shakespeare/aexn_bpe.json).
3. Accepts prompt from CLI argument or runs in interactive conversation/prompting mode.
4. Autoregressively generates next tokens using temperature, top-k, and top-p sampling.
5. Decodes token IDs back to human-readable text in real-time.
"""

from __future__ import annotations

import argparse
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from axen.model.bpe_tokenizer import AEXNTokenizer
from axen.model.foundation import AXENFoundationModel, FoundationModelConfig


def load_model_and_tokenizer(
    checkpoint_path: Path,
    tokenizer_path: Path,
    device: Optional[torch.device] = None,
) -> Tuple[AXENFoundationModel, AEXNTokenizer]:
    """Load trained AEXN GPT checkpoint and BPE tokenizer."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AEXNTokenizer.load(tokenizer_path)

    checkpoint = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    cfg_dict = checkpoint.get("config", {})
    cfg = FoundationModelConfig.from_dict(cfg_dict)
    cfg.device = str(dev)

    model = AXENFoundationModel(config=cfg, tokenizer=tokenizer)
    model.load_state_dict(checkpoint["model_state"])
    model.to(dev)
    model.eval()

    return model, tokenizer


def generate_text(
    model: AXENFoundationModel,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
) -> str:
    """Generate completion from prompt using trained model."""
    t0 = time.time()
    completion = model.generate(
        prompt_or_tokens=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    duration = max(time.time() - t0, 1e-5)
    return completion, duration


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate text using trained AEXN GPT 1.0")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/aexn_gpt/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--tokenizer", type=str, default="data/shakespeare/aexn_bpe.json", help="Path to BPE tokenizer JSON")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt text (if omitted, launches interactive mode)")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (>0.0)")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k filtering threshold")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling threshold")
    parser.add_argument("--device", type=str, default="auto", help="Inference device ('auto', 'cuda', 'cpu')")
    args = parser.parse_args()

    ckpt_path = PROJECT_ROOT / args.checkpoint
    tok_path = PROJECT_ROOT / args.tokenizer

    # Fallback to latest checkpoint if best_model doesn't exist
    if not ckpt_path.exists() and (ckpt_path.parent / "checkpoint_latest.pt").exists():
        ckpt_path = ckpt_path.parent / "checkpoint_latest.pt"

    print("=" * 78)
    print("AEXN GPT 1.0 — AUTOREGRESSIVE TEXT GENERATION")
    print(f"Loading checkpoint: {ckpt_path}")
    print(f"Loading tokenizer : {tok_path}")
    print("=" * 78)

    dev = None if args.device == "auto" else torch.device(args.device)
    model, tokenizer = load_model_and_tokenizer(ckpt_path, tok_path, device=dev)
    total_p, _ = model.count_parameters()
    print(f"Model loaded successfully on {model.device} ({total_p:,} parameters).")
    print(f"Sampling parameters: temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
    print("-" * 78)

    if args.prompt:
        # Single prompt execution
        completion, duration = generate_text(
            model,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print(f"\n[PROMPT]:\n{args.prompt}\n")
        print(f"[AEXN GPT GENERATED COMPLETION]:\n{completion}\n")
        print(f"(Generated in {duration:.2f}s on {model.device})")
    else:
        # Interactive mode
        print("\nEntering Interactive Generation Mode (type 'exit' or 'quit' to stop).\n")
        while True:
            try:
                user_prompt = input("AEXN GPT > ").strip()
                if not user_prompt:
                    continue
                if user_prompt.lower() in ("exit", "quit", "q"):
                    print("Exiting AEXN GPT Generation.")
                    break

                completion, duration = generate_text(
                    model,
                    user_prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                )
                print(f"\n[AEXN GPT]:\n{completion}\n")
                print(f"(Generated in {duration:.2f}s)\n" + "-" * 50)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break

    return 0


if __name__ == "__main__":
    sys.exit(main())
