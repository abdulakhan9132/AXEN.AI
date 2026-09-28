"""AI model abstraction for the local agent."""

from .base_model import AIModel
from .local_model import LocalModel
from .model_manager import ModelManager
from .config import LocalAIConfig, load_local_config

__all__ = ["AIModel", "LocalModel", "ModelManager", "LocalAIConfig", "load_local_config"]
