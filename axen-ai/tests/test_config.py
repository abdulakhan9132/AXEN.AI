from pathlib import Path

from axen.config.loader import ModelConfig


def test_model_config_from_yaml():
    config = ModelConfig.from_yaml(Path("configs/tiny.yaml"))

    assert config.embedding_dim == 24
    assert config.context_length == 16
    assert config.num_heads == 2
    assert config.device == "auto"
