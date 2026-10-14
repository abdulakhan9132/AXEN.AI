"""
scripts/tune_aexn_gpt.py
========================
Hyperparameter Tuning and Empirical Evaluation for AEXN GPT 1.0.

Evaluates configurations across:
1. Learning rates (e.g. 1e-4 vs 3e-4 vs 8e-4)
2. Architectural scale (e.g. 64 dim / 2 layers vs 128 dim / 4 layers)
3. Regularization (dropout 0.0 vs 0.1, weight decay 0.0 vs 0.01)

Measures:
- Final Train Loss
- Final Validation Loss
- Validation Perplexity (PPL)
- Peak GPU VRAM Usage (MB)
- Training Speed (tokens/sec)
- Qualitative Text Completion

Outputs:
- outputs/tuning_results.json
- Printed comparative summary table identifying the Pareto-optimal configuration.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

from axen.model.bpe_tokenizer import AEXNTokenizer, get_batch
from axen.model.foundation import AXENFoundationModel, FoundationModelConfig
from scripts.train_aexn_gpt import create_scheduler, evaluate_validation, set_seed, training_step

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tune_aexn_gpt")


def run_experiment_trial(
    trial_name: str,
    lr: float,
    embedding_dim: int,
    num_layers: int,
    num_heads: int,
    ff_dim: int,
    dropout: float,
    weight_decay: float,
    batch_size: int = 4,
    context_length: int = 64,
    max_steps: int = 100,
    eval_steps: int = 15,
    device: torch.device = torch.device("cpu"),
    train_bin: Path = Path("data/shakespeare/train_ids.bin"),
    val_bin: Path = Path("data/shakespeare/val_ids.bin"),
    tokenizer: Optional[AEXNTokenizer] = None,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    amsgrad: bool = False,
    grad_clip: float = 1.0,
    warmup_steps: int = 50,
    seed: int = 1337,
) -> Dict[str, Any]:
    """Run a single bounded training trial and report empirical metrics."""
    logger.info("Starting Trial [%s]: lr=%.5f, dim=%d, layers=%d, dropout=%.2f", trial_name, lr, embedding_dim, num_layers, dropout)
    set_seed(seed)
    if max_steps <= 0 or eval_steps <= 0:
        raise ValueError("Trial and validation steps must be positive")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    cfg = FoundationModelConfig(
        model_name=trial_name,
        vocab_size=tokenizer.vocab_size if tokenizer else 2048,
        context_length=context_length,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout,
        device=str(device),
    )
    model = AXENFoundationModel(config=cfg, tokenizer=tokenizer)
    optimizer = model.configure_optimizers(
        weight_decay=weight_decay,
        learning_rate=lr,
        betas=betas, eps=eps, amsgrad=amsgrad,
    )
    scheduler = create_scheduler(optimizer, warmup_steps, max_steps, lr, lr * 0.1)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    t_start = time.time()
    train_losses = []
    updates = []
    tokens_per_batch = batch_size * context_length

    for step in range(max_steps):
        x, y = get_batch(train_bin, batch_size=batch_size, context_length=context_length, device=device)
        metrics = training_step(model, optimizer, x, y, grad_clip=grad_clip, scaler=scaler, scheduler=scheduler)
        updates.append(metrics)
        train_losses.append(metrics["loss"])

    duration = max(time.time() - t_start, 1e-5)
    total_tokens = max_steps * tokens_per_batch
    tok_per_sec = total_tokens / duration

    # Fixed evaluation seed and batch size make validation comparable across trials.
    set_seed(seed + 1)
    mean_val_loss, perplexity = evaluate_validation(model, val_bin, eval_steps=eval_steps,
        batch_size=4, context_length=context_length, device=device, use_amp=use_amp)
    mean_train_loss = float(np.mean(train_losses[-20:]))

    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024**2))

    # Qualitative sample generation
    sample_text = model.generate("To be, or not to be", max_new_tokens=25, temperature=0.7, top_k=20)

    total_p, _ = model.count_parameters()
    norms = [m["grad_norm"] for m in updates if m["grad_norm"] is not None]

    return {
        "trial_name": trial_name,
        "parameters": total_p,
        "learning_rate": lr,
        "embedding_dim": embedding_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "betas": list(betas), "eps": eps, "amsgrad": amsgrad,
        "batch_size": batch_size, "warmup_steps": warmup_steps, "grad_clip": grad_clip,
        "steps": max_steps, "tokens_processed": total_tokens, "seed": seed,
        "device": str(device), "mixed_precision": use_amp,
        "gradient_norm_mean": float(np.mean(norms)) if norms else None,
        "gradient_norm_max": max(norms) if norms else None,
        "stable": all(m["update_applied"] and m["grad_norm"] is not None for m in updates),
        "skipped_updates": sum(not m["update_applied"] for m in updates),
        "updates": updates,
        "duration_sec": round(duration, 2),
        "tokens_per_sec": round(tok_per_sec, 1),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "final_train_loss": round(mean_train_loss, 4),
        "final_val_loss": round(mean_val_loss, 4),
        "perplexity": round(perplexity, 2),
        "sample_output": sample_text,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune hyperparameters for AEXN GPT 1.0")
    parser.add_argument("--steps-per-trial", type=int, default=80, help="Steps per trial run")
    parser.add_argument("--device", type=str, default="auto", help="Device ('auto', 'cuda', 'cpu')")
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=15)
    parser.add_argument("--output", type=str, default="outputs/tuning_results.json")
    args = parser.parse_args()

    train_bin = PROJECT_ROOT / "data" / "shakespeare" / "train_ids.bin"
    val_bin = PROJECT_ROOT / "data" / "shakespeare" / "val_ids.bin"
    tok_json = PROJECT_ROOT / "data" / "shakespeare" / "aexn_bpe.json"
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = PROJECT_ROOT / args.output
    results_file.parent.mkdir(parents=True, exist_ok=True)

    if not train_bin.exists() or not tok_json.exists():
        print("ERROR: Training dataset or tokenizer not found. Run scripts/train_bpe.py first.")
        return 1

    tokenizer = AEXNTokenizer.load(tok_json)

    dev_str = args.device
    if dev_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(dev_str)

    print("=" * 80)
    print("AEXN GPT 1.0 — HYPERPARAMETER TUNING & EMPIRICAL COMPARISON")
    print(f"Device: {device} | Steps per Trial: {args.steps_per_trial}")
    print("=" * 80)

    # Systematic trials exploring core learning parameters
    trials_configs = [
        # Trial 1: Baseline Architecture (128-dim, 4 layers, lr=3e-4)
        {
            "trial_name": "config_baseline_lr3e-4",
            "lr": 0.0003,
            "embedding_dim": 128,
            "num_layers": 4,
            "num_heads": 4,
            "ff_dim": 512,
            "dropout": 0.1,
            "weight_decay": 0.01,
            "betas": (0.9, 0.999), "batch_size": 4,
        },
        # Trial 2: Conservative Learning Rate (lr=1e-4)
        {
            "trial_name": "config_conservative_lr1e-4",
            "lr": 0.0001,
            "embedding_dim": 128,
            "num_layers": 4,
            "num_heads": 4,
            "ff_dim": 512,
            "dropout": 0.1,
            "weight_decay": 0.01,
            "betas": (0.9, 0.999), "batch_size": 4,
        },
        # Trial 3: Aggressive Learning Rate (lr=8e-4)
        {
            "trial_name": "config_aggressive_lr8e-4",
            "lr": 0.0008,
            "embedding_dim": 128,
            "num_layers": 4,
            "num_heads": 4,
            "ff_dim": 512,
            "dropout": 0.1,
            "weight_decay": 0.01,
            "betas": (0.9, 0.999), "batch_size": 4,
        },
        # Trial 4: Lightweight Fast Architecture (64-dim, 2 layers, lr=5e-4)
        {
            "trial_name": "config_lightweight_64dim",
            "lr": 0.0005,
            "embedding_dim": 64,
            "num_layers": 2,
            "num_heads": 2,
            "ff_dim": 256,
            "dropout": 0.05,
            "weight_decay": 0.01,
            "betas": (0.9, 0.999), "batch_size": 4,
        },
    ]
    # Retain the original architecture trials and vary optimizer factors individually.
    baseline = trials_configs[0]
    for name, overrides in (
        ("config_beta95_99", {"betas": (0.95, 0.99)}),
        ("config_no_decay", {"weight_decay": 0.0}),
        ("config_batch8", {"batch_size": 8}),
    ):
        trials_configs.append({**baseline, **overrides, "trial_name": name})

    results = []
    for tc in trials_configs:
        print(f"\n--> Running Trial: {tc['trial_name']} ...")
        res = run_experiment_trial(
            trial_name=tc["trial_name"],
            lr=tc["lr"],
            embedding_dim=tc["embedding_dim"],
            num_layers=tc["num_layers"],
            num_heads=tc["num_heads"],
            ff_dim=tc["ff_dim"],
            dropout=tc["dropout"],
            weight_decay=tc["weight_decay"],
            betas=tc["betas"], batch_size=tc["batch_size"],
            warmup_steps=args.warmup_steps, eval_steps=args.eval_steps,
            max_steps=args.steps_per_trial,
            device=device,
            train_bin=train_bin,
            val_bin=val_bin,
            tokenizer=tokenizer,
        )
        results.append(res)
        results_file.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
        print(f"    Train Loss: {res['final_train_loss']:.4f} | Val Loss: {res['final_val_loss']:.4f} | PPL: {res['perplexity']:.2f} | Speed: {res['tokens_per_sec']:.0f} tok/s")

    # Save results to JSON
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Determine best configuration by validation loss
    stable_results = [r for r in results if r["stable"] and math.isfinite(r["final_val_loss"])]
    if not stable_results:
        print("No stable trial available; results saved without promoting a best configuration.")
        return 1
    best_trial = min(stable_results, key=lambda r: r["final_val_loss"])
    results_file.with_name(results_file.stem + "_best.json").write_text(
        json.dumps(best_trial, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING RESULTS SUMMARY")
    print("=" * 80)
    header = f"{'Trial Name':<28} | {'Params':<9} | {'LR':<7} | {'TrainLoss':<9} | {'ValLoss':<7} | {'PPL':<6} | {'Tok/s':<6}"
    print(header)
    print("-" * len(header))
    for r in results:
        marker = " *" if r["trial_name"] == best_trial["trial_name"] else ""
        print(
            f"{r['trial_name']:<28} | "
            f"{r['parameters']:<9,d} | "
            f"{r['learning_rate']:<7.4f} | "
            f"{r['final_train_loss']:<9.4f} | "
            f"{r['final_val_loss']:<7.4f} | "
            f"{r['perplexity']:<6.2f} | "
            f"{r['tokens_per_sec']:<6.0f}{marker}"
        )

    print("-" * len(header))
    print(f"BEST CONFIGURATION: {best_trial['trial_name']} (Lowest Val Loss: {best_trial['final_val_loss']:.4f}, PPL: {best_trial['perplexity']:.2f})")
    print(f"Results saved to: {results_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
