from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

from axen.config.loader import ModelConfig
from axen.data.clean import clean_text_file
from axen.data.split import split_text_file
from axen.data.dataset import TextDataset
from axen.data.pipeline import SUPPORTED_SUFFIXES, inspect_raw_data, prepare_data, preview_data
from axen.tokenizer.tokenizer import Tokenizer
from axen.model.language_model import AXENLanguageModel
from axen.inference.generator import generate_text
from axen.training.overfit import load_checkpoint
from axen.training.trainer import train_prepared
from axen.ui.desktop import launch_desktop


def _ensure_project_layout() -> None:
    for folder in ["data", "data/samples", "configs", "checkpoints", "logs", "docs", "tests"]:
        Path(folder).mkdir(parents=True, exist_ok=True)


def cmd_setup(_args: argparse.Namespace) -> int:
    _ensure_project_layout()
    print("AXEN AI project initialized.")
    print("Use 'python main.py prepare-data' to build the teaching corpus.")
    return 0


def cmd_prepare_data(_args: argparse.Namespace) -> int:
    config = ModelConfig.from_yaml(_args.config)
    manifest = prepare_data(
        config.raw_dir,
        config.processed_dir,
        seed=config.seed,
        train_ratio=config.train_ratio,
        validation_ratio=config.validation_ratio,
    )
    print(f"Prepared {manifest['document_count']} records from {config.raw_dir}")
    print(f"Vocabulary size: {manifest['vocabulary_size']}")
    print(f"Split counts: {manifest['split_counts']}")
    return 0


def cmd_data_preview(_args: argparse.Namespace) -> int:
    config = ModelConfig.from_yaml(_args.config)
    preview = preview_data(config.raw_dir, seed=config.seed)
    print(f"Files: {preview['file_count']}")
    print(f"File names: {preview['files']}")
    print(f"File types: {preview['file_types']}")
    print(f"Characters: {preview['characters']}")
    print(f"Estimated tokens: {preview['estimated_tokens']}")
    print(f"Empty records: {preview['empty_files']}")
    print(f"Duplicate records: {preview['duplicate_count']}")
    print(f"Train/validation/test samples: {preview['split_counts']}")
    for item in preview["preview"]:
        print(f"Preview [{item['source']}]: {item['text'][:160]!r}")
    return 0


def cmd_data_stats(_args: argparse.Namespace) -> int:
    config = ModelConfig.from_yaml(_args.config)
    manifest_path = Path(config.processed_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Missing data manifest. Run 'python main.py prepare-data' first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_tokens = manifest.get("split_token_counts", {})
    train_tokens = int(split_tokens.get("train", 0))
    windows = max(1, train_tokens - config.context_length)
    steps_per_epoch = math.ceil(windows / config.batch_size)
    training_steps = steps_per_epoch * config.epochs
    model = AXENLanguageModel(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        ff_dim=config.ff_dim,
        dropout=config.dropout,
        activation=config.activation,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    estimated_memory_mb = math.ceil(
        (parameter_count * 16 + config.batch_size * config.context_length * config.embedding_dim * 8) / (1024**2)
    )
    estimated_time_seconds = training_steps * 0.05
    print(f"Documents: {manifest.get('document_count', 0)}")
    print(f"Characters: {manifest.get('character_count', 0)}")
    print(f"Estimated tokens: {manifest.get('estimated_token_count', manifest.get('token_count', 0))}")
    print(f"Vocabulary size: {manifest.get('vocabulary_size', 0)}")
    print(f"Train tokens: {split_tokens.get('train', 0)}")
    print(f"Validation tokens: {split_tokens.get('validation', 0)}")
    print(f"Test tokens: {split_tokens.get('test', 0)}")
    print(f"Duplicate count: {manifest.get('duplicate_count', 0)}")
    print(f"Empty files: {manifest.get('empty_files', [])}")
    print(f"Malformed files: {manifest.get('malformed_files', {})}")
    print(f"Average document length: {manifest.get('average_document_length', 0):.2f}")
    print(f"Longest document: {manifest.get('longest_document', 0)}")
    print(f"Shortest document: {manifest.get('shortest_document', 0)}")
    print(f"Estimated training steps: {training_steps}")
    print(f"Estimated training time: {estimated_time_seconds / 60:.1f} minutes on an RTX 3050 (heuristic)")
    print(f"Estimated GPU memory: {estimated_memory_mb} MiB (weights, optimizer, and activations heuristic)")
    for warning in manifest.get("split_warnings", []):
        print(f"Split warning: {warning}")
    return 0


def cmd_training_readiness(_args: argparse.Namespace) -> int:
    config = ModelConfig.from_yaml(_args.config)
    reasons: list[str] = []
    raw_dir = Path(config.raw_dir)
    processed_dir = Path(config.processed_dir)
    if not raw_dir.exists():
        reasons.append(f"raw data directory does not exist: {raw_dir}")
    else:
        ingestion = inspect_raw_data(raw_dir)
        if not ingestion.records:
            reasons.append("no usable documents were found in data/raw")
        if ingestion.malformed_files:
            reasons.append(f"malformed files: {', '.join(sorted(ingestion.malformed_files))}")

    manifest_path = processed_dir / "manifest.json"
    tokenizer_path = processed_dir / "tokenizer.json"
    manifest: dict[str, object] = {}
    tokenizer: Tokenizer | None = None
    if not tokenizer_path.exists():
        reasons.append("tokenizer does not exist; run prepare-data")
    else:
        try:
            tokenizer = Tokenizer.load(tokenizer_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            reasons.append(f"tokenizer cannot be loaded: {exc}")
    if not manifest_path.exists():
        reasons.append("dataset manifest does not exist; run prepare-data")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            reasons.append(f"dataset manifest cannot be loaded: {exc}")

    if manifest_path.exists() and tokenizer_path.exists() and tokenizer is not None:
        split_token_counts = manifest.get("split_token_counts", {})
        if not isinstance(split_token_counts, dict) or int(split_token_counts.get("train", 0)) < config.context_length + 1:
            reasons.append(f"not enough train tokens for context length {config.context_length}")
        if config.vocab_size < tokenizer.vocab_size:
            reasons.append(f"model vocabulary {config.vocab_size} is smaller than tokenizer vocabulary {tokenizer.vocab_size}")

    required_files = ["train.json", "validation.json", "test.json", "train.txt", "validation.txt", "test.txt"]
    missing_files = [name for name in required_files if not (processed_dir / name).exists()]
    if missing_files:
        reasons.append(f"prepared split files are missing: {', '.join(missing_files)}")
    if manifest.get("split_warnings"):
        reasons.extend(str(warning) for warning in manifest["split_warnings"])

    config_checks = (
        config.vocab_size > 0
        and config.context_length > 0
        and config.embedding_dim > 0
        and config.num_layers > 0
        and config.num_heads > 0
        and config.embedding_dim % config.num_heads == 0
        and config.ff_dim > 0
        and config.batch_size > 0
        and config.epochs > 0
        and config.max_steps > 0
        and 0 < config.train_ratio < 1
        and 0 <= config.validation_ratio < 1
        and config.train_ratio + config.validation_ratio < 1
    )
    if not config_checks:
        reasons.append("configuration values are invalid")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA availability: {'PASS' if cuda_available else 'INFO - unavailable; CPU fallback will be used'}")
    if cuda_available:
        memory_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024**2))
        print(f"GPU memory: PASS - {torch.cuda.get_device_name(0)}, {memory_mb} MiB")
    else:
        print("GPU memory: INFO - unavailable")
    checks = {
        "data exists": raw_dir.exists(),
        "enough tokens": bool(manifest) and int(manifest.get("split_token_counts", {}).get("train", 0)) >= config.context_length + 1,
        "tokenizer exists": tokenizer_path.exists() and tokenizer is not None,
        "dataset prepared": manifest_path.exists(),
        "train split exists": (processed_dir / "train.json").exists(),
        "validation split exists": (processed_dir / "validation.json").exists(),
        "test split exists": (processed_dir / "test.json").exists(),
        "model vocabulary matches tokenizer": tokenizer is not None and config.vocab_size >= tokenizer.vocab_size,
        "configuration valid": config_checks,
    }
    for label, passed in checks.items():
        print(f"{label}: {'PASS' if passed else 'FAIL'}")
    if reasons:
        print("NOT READY")
        print("Reasons:")
        for reason in dict.fromkeys(reasons):
            print(f"- {reason}")
        return 1
    print("READY FOR TRAINING")
    return 0


def cmd_model_info(_args: argparse.Namespace) -> int:
    config_path = Path(_args.config) if _args.config else Path("configs/tiny.yaml")
    config = ModelConfig.from_yaml(config_path)
    print(config)
    return 0


def cmd_train(_args: argparse.Namespace) -> int:
    config = ModelConfig.from_yaml(_args.config)
    def report(row: dict[str, float | int | str]) -> None:
        validation = row["validation_loss"]
        validation_text = f"{float(validation):.4f}" if isinstance(validation, (int, float)) and validation == validation else "pending"
        print(
            f"Epoch {row['epoch']}: train_loss={float(row['train_loss']):.4f}, "
            f"validation_loss={validation_text}, learning_rate={float(row['learning_rate']):.6f}, "
            f"elapsed={float(row['elapsed_seconds']):.1f}s, steps={row['steps']}, device={row['device']}"
        )

    result = train_prepared(config, on_epoch=report)
    print(f"Training complete: {result.steps} optimizer steps on {result.device}.")
    print(f"Best validation loss: {result.best_validation_loss:.4f}")
    print(f"Best checkpoint: {result.best_checkpoint}")
    print(f"Latest checkpoint: {result.latest_checkpoint}")
    return 0


def cmd_generate(_args: argparse.Namespace) -> int:
    config = ModelConfig.from_yaml(_args.config)
    checkpoint = Path(config.checkpoint_dir) / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing trained checkpoint: {checkpoint}. Run 'python main.py train' first.")
    model, tokenizer, _ = load_checkpoint(checkpoint)
    prompt = _args.prompt_option or _args.prompt or "AXEN"
    text = generate_text(model, tokenizer, prompt, max_new_tokens=_args.max_new_tokens, temperature=_args.temperature)
    print(text)
    return 0


def cmd_test(_args: argparse.Namespace) -> int:
    return subprocess.call([sys.executable, "-m", "pytest", "-q", "tests"], cwd=Path(__file__).resolve().parents[3])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AXEN AI command line")
    sub = parser.add_subparsers(dest="command", required=True)

    setup = sub.add_parser("setup", help="Create the base project folders")
    setup.set_defaults(func=cmd_setup)

    prepare = sub.add_parser("prepare-data", help="Prepare local files from data/raw")
    prepare.add_argument("--config", default="configs/my_data_training.yaml")
    prepare.set_defaults(func=cmd_prepare_data)

    preview = sub.add_parser("data-preview", help="Preview local files without training")
    preview.add_argument("--config", default="configs/my_data_training.yaml")
    preview.set_defaults(func=cmd_data_preview)

    stats = sub.add_parser("data-stats", help="Display the processed data manifest")
    stats.add_argument("--config", default="configs/my_data_training.yaml")
    stats.set_defaults(func=cmd_data_stats)

    readiness = sub.add_parser("training-readiness", help="Check whether prepared data is ready for training")
    readiness.add_argument("--config", default="configs/my_data_training.yaml")
    readiness.set_defaults(func=cmd_training_readiness)

    info = sub.add_parser("model-info", help="Display a YAML config summary")
    info.add_argument("--config", default="configs/tiny.yaml")
    info.set_defaults(func=cmd_model_info)

    train = sub.add_parser("train", help="Prepare a training run stub")
    train.add_argument("--config", default="configs/tiny.yaml")
    train.set_defaults(func=cmd_train)

    gen = sub.add_parser("generate", help="Generate text with a small model stub")
    gen.add_argument("prompt", nargs="?", default=None)
    gen.add_argument("--prompt", dest="prompt_option", default=None)
    gen.add_argument("--config", default="configs/tiny.yaml")
    gen.add_argument("--max-new-tokens", type=int, default=12)
    gen.add_argument("--temperature", type=float, default=1.0)
    gen.set_defaults(func=cmd_generate)

    test = sub.add_parser("test", help="Run the project test suite")
    test.set_defaults(func=cmd_test)

    desktop = sub.add_parser("desktop", help="Launch the local Tkinter desktop chat")
    desktop.set_defaults(func=lambda _args: launch_desktop() or 0)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
