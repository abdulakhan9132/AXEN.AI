import json
from pathlib import Path

from axen.config.loader import ModelConfig
from axen.tokenizer.tokenizer import Tokenizer
from axen.training.overfit import load_checkpoint
from axen.training.trainer import train_prepared


def test_train_prepared_performs_steps_and_writes_reloadable_checkpoints(tmp_path: Path):
    processed = tmp_path / "processed"
    checkpoints = tmp_path / "checkpoints"
    processed.mkdir()
    tokenizer = Tokenizer().fit(["abcdefghij" * 8])
    tokenizer.save(processed / "tokenizer.json")
    tokenizer.encode("abcdefghij" * 8, add_special_tokens=False)
    train_tokens = tokenizer.encode("abcdefghij" * 8, add_special_tokens=False)
    validation_tokens = tokenizer.encode("jihgfedcba" * 8, add_special_tokens=False)
    (processed / "train.json").write_text(json.dumps(train_tokens), encoding="utf-8")
    (processed / "validation.json").write_text(json.dumps(validation_tokens), encoding="utf-8")

    config = ModelConfig(
        vocab_size=32,
        context_length=8,
        embedding_dim=16,
        num_layers=1,
        num_heads=2,
        ff_dim=32,
        dropout=0.0,
        learning_rate=0.01,
        batch_size=4,
        epochs=2,
        max_steps=0,
        gradient_clip=1.0,
        validation_frequency=1,
        checkpoint_dir=str(checkpoints),
        seed=7,
        device="cpu",
        processed_dir=str(processed),
    )

    result = train_prepared(config)

    assert result.steps > 0
    assert len(result.rows) == 2
    assert (checkpoints / "best.pt").exists()
    assert (checkpoints / "latest.pt").exists()
    model, loaded_tokenizer, payload = load_checkpoint(checkpoints / "best.pt", device="cpu")
    assert next(model.parameters()).device.type == "cpu"
    assert loaded_tokenizer.vocab == tokenizer.vocab
    assert 0 < payload["training_step"] <= result.steps
    assert payload["metadata"]["epoch"] in {1, 2}
