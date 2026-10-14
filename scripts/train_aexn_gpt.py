"""
scripts/train_aexn_gpt.py
=========================
Complete Autoregressive GPT Neural Network Training Pipeline for AEXN GPT 1.0.

Key Capabilities:
1. Loads pre-tokenized BPE binary arrays (data/shakespeare/train_ids.bin, val_ids.bin).
2. Sets up causal input-target batches (x: token[t], y: token[t+1]).
3. Performs forward propagation through Token Embeddings, Positional Embeddings,
   Causal Multi-Head Attention, Pre-LN, and Feed-Forward Blocks.
4. Computes Cross-Entropy loss from raw logits against target tokens.
5. Performs backpropagation, calculates gradients, and applies gradient clipping.
6. Updates model weights via AdamW optimizer with decoupled weight decay.
7. Employs cosine learning-rate decay schedule with linear warmup.
8. Evaluates on unseen validation data periodically (loss & perplexity).
9. Manages model checkpoints (weights, optimizer, scheduler, RNG state) with resumption.
10. Supports Mixed-Precision Training (torch.amp.autocast) on NVIDIA GPUs.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
import yaml

from axen.model.bpe_tokenizer import AEXNTokenizer, get_batch
from axen.model.foundation import AXENFoundationModel, FoundationModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_aexn_gpt")


def set_seed(seed: int = 1337) -> None:
    """Set random seeds for reproducible initialization and training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_learning_rate(
    step: int,
    warmup_steps: int,
    max_steps: int,
    base_lr: float,
    min_lr: float,
) -> float:
    """
    Cosine learning-rate schedule with linear warmup.
    1. Linear warmup: 0 to base_lr over warmup_steps.
    2. Cosine decay: base_lr down to min_lr over remaining steps.
    3. Post max_steps: stays at min_lr.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)


def create_scheduler(optimizer, warmup_steps, max_steps, base_lr, min_lr):
    """Prepare the first update's LR; step AFTER successful optimizer updates."""
    if base_lr <= 0 or not 0 <= min_lr <= base_lr or warmup_steps < 0 or max_steps <= 0:
        raise ValueError("Invalid learning-rate schedule configuration")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: get_learning_rate(step, warmup_steps, max_steps, base_lr, min_lr) / base_lr
    )
    scheduler.schedule_config = dict(warmup_steps=warmup_steps, max_steps=max_steps, base_lr=base_lr, min_lr=min_lr)
    return scheduler


def training_step(model, optimizer, x, y, *, grad_clip=1.0, scaler=None, scheduler=None):
    """Shared training/tuning step using raw-logit CE and unscaled gradient clipping."""
    use_amp = scaler is not None and scaler.is_enabled()
    model.train()
    with torch.amp.autocast(device_type=x.device.type, enabled=use_amp):
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite training loss; no optimizer update applied")
    optimizer.zero_grad(set_to_none=True)
    if use_amp:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()
    if grad_clip <= 0:
        raise ValueError("grad_clip must be positive")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=not use_amp)
    lr = optimizer.param_groups[0]["lr"]
    if use_amp:
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        updated = scaler.get_scale() >= old_scale
    else:
        optimizer.step()
        updated = True
    if updated and scheduler is not None:
        scheduler.step()
    return {"loss": float(loss.detach()), "grad_norm": float(norm) if torch.isfinite(norm) else None,
            "clipped_grad_norm": min(float(norm), grad_clip) if torch.isfinite(norm) else None,
            "update_applied": updated, "lr": lr}


@torch.no_grad()
def evaluate_validation(
    model: AXENFoundationModel,
    val_bin_path: Path,
    eval_steps: int = 20,
    batch_size: int = 4,
    context_length: int = 64,
    device: torch.device = torch.device("cpu"),
    use_amp: bool = False,
) -> Tuple[float, float]:
    """
    Evaluate the model on unseen validation data.
    Returns: (val_loss, perplexity).
    Perplexity = exp(val_loss), measuring how confident the model's predictions are.
    """
    model.eval()
    losses = []
    device_type = "cuda" if device.type == "cuda" else "cpu"

    for _ in range(eval_steps):
        x, y = get_batch(val_bin_path, batch_size=batch_size, context_length=context_length, device=device)
        with torch.amp.autocast(device_type=device_type, enabled=use_amp and device_type == "cuda"):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())

    model.train()
    mean_loss = float(np.mean(losses))
    perplexity = float(np.exp(min(mean_loss, 20.0)))  # Cap to prevent numerical overflow
    return mean_loss, perplexity


def save_checkpoint(
    checkpoint_path: Path,
    model: AXENFoundationModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    val_loss: float,
    config_dict: Dict[str, Any],
    scheduler=None,
    scaler=None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    best_val_loss: Optional[float] = None,
) -> None:
    """Save training checkpoint with model weights, optimizer, RNG states, and config."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "val_loss": val_loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config_dict,
        "hyperparameters": hyperparameters or {},
        "best_val_loss": val_loss if best_val_loss is None else best_val_loss,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "schedule_config": getattr(scheduler, "schedule_config", None),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "rng_states": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    torch.save(payload, checkpoint_path)
    logger.info("Saved checkpoint at step %d to %s", step, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: AXENFoundationModel,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
    scheduler=None,
    scaler=None,
) -> Tuple[int, float]:
    """Load model weights and optimizer state from checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None:
        if checkpoint.get("scheduler_state") is not None:
            if checkpoint.get("schedule_config") != scheduler.schedule_config:
                raise ValueError("Resume scheduler configuration does not match checkpoint")
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        else:
            # Legacy checkpoints used a stateless schedule: position it at the next update.
            scheduler.last_epoch = int(checkpoint.get("step", 0))
            rates = [base * fn(scheduler.last_epoch) for base, fn in zip(scheduler.base_lrs, scheduler.lr_lambdas)]
            for group, rate in zip(optimizer.param_groups, rates):
                group["lr"] = rate
            scheduler._last_lr = rates
    if scaler is not None and checkpoint.get("scaler_state"):
        scaler.load_state_dict(checkpoint["scaler_state"])

    rng_states = checkpoint.get("rng_states", {})
    if rng_states:
        if "torch" in rng_states and rng_states["torch"] is not None:
            torch.set_rng_state(rng_states["torch"].cpu())
        if "cuda" in rng_states and rng_states["cuda"] is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state(rng_states["cuda"].cpu())
            except Exception:
                pass
        if "numpy" in rng_states and rng_states["numpy"] is not None:
            np.random.set_state(rng_states["numpy"])
        if "python" in rng_states and rng_states["python"] is not None:
            random.setstate(rng_states["python"])

    step = checkpoint.get("step", 0)
    val_loss = checkpoint.get("best_val_loss", checkpoint.get("val_loss", float("inf")))
    logger.info("Resumed checkpoint from %s at step %d (best val loss: %.4f)", checkpoint_path, step, val_loss)
    return step, val_loss


def train(
    config_path: str = "configs/aexn_gpt.yaml",
    resume_path: Optional[str] = None,
    override_max_steps: Optional[int] = None,
    output_dir: Optional[str] = None,
    stop_after_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Main training routine."""
    cfg_file = PROJECT_ROOT / config_path
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_file}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    resume_metadata = None
    if resume_path:
        resume_metadata = torch.load(resume_path, map_location="cpu", weights_only=False)
        cfg["model"].update(resume_metadata["config"])
        saved_training = resume_metadata.get("hyperparameters", {}).get("training", {})
        for key in ("optimizer", "amsgrad", "scheduler", "learning_rate", "min_lr", "weight_decay",
                    "beta1", "beta2", "eps", "grad_clip", "warmup_steps", "batch_size", "max_steps"):
            if key in saved_training:
                cfg["training"][key] = saved_training[key]

    m_cfg = cfg["model"]
    t_cfg = cfg["training"]
    d_cfg = cfg["data"]
    c_cfg = cfg["checkpoints"]

    set_seed(t_cfg.get("seed", 1337))

    # Resolve paths
    train_bin = PROJECT_ROOT / d_cfg["train_binary"]
    val_bin = PROJECT_ROOT / d_cfg["val_binary"]
    tok_json = PROJECT_ROOT / d_cfg["tokenizer_file"]
    meta_json = PROJECT_ROOT / d_cfg["metadata_file"]
    ckpt_dir = Path(output_dir) if output_dir else PROJECT_ROOT / c_cfg["checkpoint_dir"]
    cfg["checkpoints"]["checkpoint_dir"] = str(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not train_bin.exists() or not val_bin.exists() or not tok_json.exists():
        raise FileNotFoundError(
            "Dataset binary files or tokenizer not found. Run `scripts/train_bpe.py` first."
        )

    # Load tokenizer
    tokenizer = AEXNTokenizer.load(tok_json)
    vocab_size = tokenizer.vocab_size

    # Resolve device & mixed-precision
    req_dev = t_cfg.get("device", "auto")
    if req_dev == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(req_dev)

    use_amp = bool(t_cfg.get("mixed_precision", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif t_cfg.get("cpu_threads"):
        torch.set_num_threads(int(t_cfg["cpu_threads"]))

    print("=" * 78)
    print("AEXN GPT 1.0 — NEURAL NETWORK TRAINING PIPELINE")
    print(f"Device: {device} | Mixed-Precision (AMP): {use_amp}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} (Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB)")
    print("=" * 78)

    # Initialize model
    model_config = FoundationModelConfig(
        model_name=m_cfg.get("model_name", "aexn_gpt_1.0"),
        vocab_size=vocab_size,
        context_length=m_cfg.get("context_length", 64),
        embedding_dim=m_cfg.get("embedding_dim", 128),
        num_layers=m_cfg.get("num_layers", 4),
        num_heads=m_cfg.get("num_heads", 4),
        ff_dim=m_cfg.get("ff_dim", 512),
        dropout=m_cfg.get("dropout", 0.1),
        activation=m_cfg.get("activation", "gelu"),
        device=str(device),
    )
    model = AXENFoundationModel(config=model_config, tokenizer=tokenizer)
    total_params, trainable_params = model.count_parameters()
    print(f"Model Parameters: {total_params:,} total ({trainable_params:,} trainable)")

    # Training hyperparameters
    batch_size = t_cfg.get("batch_size", 4)
    context_length = model_config.context_length
    base_lr = float(t_cfg.get("learning_rate", 0.0003))
    min_lr = float(t_cfg.get("min_lr", 0.00003))
    weight_decay = float(t_cfg.get("weight_decay", 0.01))
    beta1 = float(t_cfg.get("beta1", 0.9))
    beta2 = float(t_cfg.get("beta2", 0.999))
    eps = float(t_cfg.get("eps", 1e-8))
    grad_clip = float(t_cfg.get("grad_clip", 1.0))
    warmup_steps = int(t_cfg.get("warmup_steps", 50))
    max_steps = int(t_cfg.get("max_steps", 500)) if override_max_steps is None else override_max_steps
    eval_interval = int(t_cfg.get("eval_interval", 50))
    eval_steps = int(t_cfg.get("eval_steps", 20))
    if t_cfg.get("optimizer", "AdamW").lower() != "adamw":
        raise ValueError("Only the existing PyTorch AdamW optimizer is supported")
    if t_cfg.get("scheduler", "cosine").lower() != "cosine":
        raise ValueError("Only cosine scheduling is supported")
    if min(batch_size, max_steps, eval_interval, eval_steps) <= 0 or grad_clip <= 0:
        raise ValueError("Training counts and gradient clipping must be positive")

    # Configure AdamW optimizer with decoupled weight decay
    optimizer = model.configure_optimizers(
        weight_decay=weight_decay,
        learning_rate=base_lr,
        betas=(beta1, beta2),
        eps=eps,
        amsgrad=bool(t_cfg.get("amsgrad", False)),
    )
    schedule_cfg = (resume_metadata or {}).get("schedule_config") or dict(
        warmup_steps=warmup_steps, max_steps=max_steps, base_lr=base_lr, min_lr=min_lr)
    scheduler = create_scheduler(optimizer, **schedule_cfg)
    cfg["training"]["max_steps"] = max_steps

    # Resume from checkpoint if requested
    start_step = 0
    best_val_loss = float("inf")
    if resume_path:
        r_p = Path(resume_path)
        if r_p.exists():
            start_step, best_val_loss = load_checkpoint(r_p, model, optimizer, device, scheduler, scaler)
        else:
            logger.warning("Resume checkpoint %s not found. Starting from step 0.", r_p)

    # Diagnostics must not consume the RNG restored for the next training batch.
    rng_torch = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    rng_numpy, rng_python = np.random.get_state(), random.getstate()
    # Initial generation before training
    test_prompt = "To be, or not to be"
    print("\n[Baseline Generation Before Training]")
    gen_before = model.generate(test_prompt, max_new_tokens=30, temperature=0.8, top_k=20)
    print(f"Prompt : {test_prompt!r}")
    print(f"Output : {gen_before!r}")

    # Initial evaluation
    init_val_loss, init_val_ppl = evaluate_validation(
        model, val_bin, eval_steps=eval_steps, batch_size=batch_size, context_length=context_length, device=device, use_amp=use_amp
    )
    print(f"\nInitial Validation Loss: {init_val_loss:.4f} | Initial Perplexity: {init_val_ppl:.2f}")
    init_train_loss, _ = evaluate_validation(
        model, train_bin, eval_steps=eval_steps, batch_size=batch_size,
        context_length=context_length, device=device, use_amp=use_amp)
    torch.set_rng_state(rng_torch)
    if rng_cuda:
        torch.cuda.set_rng_state_all(rng_cuda)
    np.random.set_state(rng_numpy)
    random.setstate(rng_python)

    print("\nStarting Training Loop...")
    print(f"Steps: {start_step} -> {max_steps} | Batch Size: {batch_size} | Sequence Length: {context_length}")
    print("-" * 78)

    model.train()
    t_start = time.time()
    total_tokens_processed = 0
    recent_losses = []

    history = {
        "step": [],
        "train_loss": [],
        "val_loss": [],
        "val_ppl": [],
        "lr": [],
        "tokens_per_sec": [],
        "updates": [],
    }

    tokens_per_batch = batch_size * context_length

    end_step = min(max_steps, start_step + stop_after_steps) if stop_after_steps is not None else max_steps
    if end_step <= start_step:
        raise ValueError("Training must perform at least one step")
    for step in range(start_step, end_step):
        t_step_start = time.time()

        # Sample causal input/target tokens; shared step handles AMP -> clipping -> AdamW -> LR.
        x, y = get_batch(train_bin, batch_size=batch_size, context_length=context_length, device=device)
        metrics = training_step(model, optimizer, x, y, grad_clip=grad_clip, scaler=scaler, scheduler=scheduler)
        lr = metrics["lr"]
        history["updates"].append({"step": step + 1, **metrics})

        step_duration = max(time.time() - t_step_start, 1e-5)
        total_tokens_processed += tokens_per_batch
        step_tokens_per_sec = tokens_per_batch / step_duration
        recent_losses.append(metrics["loss"])

        # 5. Periodic evaluation & checkpointing
        if (step + 1) % eval_interval == 0 or (step + 1) == end_step:
            val_loss, val_ppl = evaluate_validation(
                model, val_bin, eval_steps=eval_steps, batch_size=batch_size, context_length=context_length, device=device, use_amp=use_amp
            )
            mean_train_loss = float(np.mean(recent_losses[-eval_interval:]))

            vram_mb = 0.0
            if device.type == "cuda":
                vram_mb = torch.cuda.max_memory_allocated() / (1024**2)

            print(
                f"Step {step+1:4d}/{max_steps} | "
                f"Train Loss: {mean_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val PPL: {val_ppl:6.2f} | "
                f"LR: {lr:.6f} | "
                f"Speed: {step_tokens_per_sec:6.0f} tok/s | "
                f"VRAM: {vram_mb:5.0f} MB"
            )

            history["step"].append(step + 1)
            history["train_loss"].append(round(mean_train_loss, 4))
            history["val_loss"].append(round(val_loss, 4))
            history["val_ppl"].append(round(val_ppl, 2))
            history["lr"].append(lr)
            history["tokens_per_sec"].append(round(step_tokens_per_sec, 1))

            best_val_loss = min(best_val_loss, val_loss)
            checkpoint_options = dict(scheduler=scheduler, scaler=scaler, hyperparameters=cfg, best_val_loss=best_val_loss)
            # Save latest checkpoint
            if c_cfg.get("save_latest", True):
                save_checkpoint(ckpt_dir / "checkpoint_latest.pt", model, optimizer, step + 1, val_loss, model_config.to_dict(), **checkpoint_options)

            # Save best checkpoint
            if c_cfg.get("save_best", True) and val_loss <= best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(ckpt_dir / "best_model.pt", model, optimizer, step + 1, val_loss, model_config.to_dict(), **checkpoint_options)

            # Save periodic step checkpoint
            if (step + 1) % c_cfg.get("save_interval", 100) == 0:
                save_checkpoint(ckpt_dir / f"checkpoint_step_{step+1}.pt", model, optimizer, step + 1, val_loss, model_config.to_dict(), **checkpoint_options)

    total_duration = time.time() - t_start
    avg_throughput = total_tokens_processed / max(total_duration, 1e-5)

    print("-" * 78)
    print(f"Training Complete in {total_duration:.2f}s!")
    print(f"Total Tokens Processed: {total_tokens_processed:,} ({avg_throughput:.0f} tokens/sec)")
    print(f"Best Validation Loss  : {best_val_loss:.4f}")

    # Generate text after training
    print("\n[Generation After Training (Trained Weights)]")
    gen_after = model.generate(test_prompt, max_new_tokens=40, temperature=0.7, top_k=30)
    print(f"Prompt : {test_prompt!r}")
    print(f"Output : {gen_after!r}")

    # Save summary report
    summary = {
        "model_name": model_config.model_name,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "start_step": start_step,
        "completed_steps": end_step,
        "total_duration_sec": round(total_duration, 2),
        "avg_tokens_per_sec": round(avg_throughput, 1),
        "initial_val_loss": round(init_val_loss, 4),
        "initial_val_ppl": round(init_val_ppl, 2),
        "best_val_loss": round(best_val_loss, 4),
        "final_val_loss": round(history["val_loss"][-1] if history["val_loss"] else 0.0, 4),
        "final_val_ppl": round(history["val_ppl"][-1] if history["val_ppl"] else 0.0, 2),
        "sample_generation_before": gen_before,
        "sample_generation_after": gen_after,
        "history": history,
        "initial_train_loss": init_train_loss,
        "final_train_loss": recent_losses[-1],
        "hyperparameters": cfg,
        "schedule_config": schedule_cfg,
        "device": str(device),
        "mixed_precision": use_amp,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0,
        "skipped_updates": sum(not item["update_applied"] for item in history["updates"]),
    }

    report_path = ckpt_dir / "training_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved training report to: {report_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AEXN GPT 1.0 on Shakespeare with BPE")
    parser.add_argument("--config", type=str, default="configs/aexn_gpt.yaml", help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--output-dir", type=str, default=None, help="Separate experiment/checkpoint directory")
    parser.add_argument("--stop-after", type=int, default=None, help="Stop after this many additional steps, preserving schedule horizon")
    args = parser.parse_args()

    train(config_path=args.config, resume_path=args.resume, override_max_steps=args.max_steps,
          output_dir=args.output_dir, stop_after_steps=args.stop_after)
