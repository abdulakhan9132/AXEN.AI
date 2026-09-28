from pathlib import Path

import pytest

from axen.inference.chat_engine import AXENChatEngine


def test_model_missing_is_not_ready(tmp_path: Path):
    engine = AXENChatEngine(tmp_path / "missing.pt")

    assert engine.load_model() is False
    assert engine.is_model_ready() is False
    assert engine.model_info()["status"] == "NOT TRAINED"
    with pytest.raises(RuntimeError, match="not ready"):
        engine.generate("hello")


def test_trained_checkpoint_is_available():
    engine = AXENChatEngine("checkpoints/overfit.pt")

    assert engine.load_model() is True
    info = engine.model_info()
    assert engine.is_model_ready() is True
    assert info["parameters"] > 0
    assert info["vocabulary"] > 0
