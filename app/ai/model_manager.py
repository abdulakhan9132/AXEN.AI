from __future__ import annotations

import os
from typing import Dict, Optional

from .base_model import AIModel
from .config import LocalAIConfig, load_local_config
from .local_model import LocalModel


class ModelManager:
    """Manage model selection, availability checks, and fallback behavior."""

    def __init__(self, preferred_model: Optional[str] = None, config_path: str = "configs/axen_local.yaml"):
        self.config: LocalAIConfig = load_local_config(config_path)
        self.preferred_model = preferred_model or os.getenv("LOCAL_MODEL_NAME", self.config.model)
        self._models: Dict[str, AIModel] = {"local": LocalModel(model_name=self.preferred_model, config=self.config)}

    def get_model(self, model_name: Optional[str] = None) -> AIModel:
        name = model_name or self.preferred_model
        if name not in self._models:
            self._models[name] = LocalModel(model_name=name, config=self.config)
        return self._models[name]

    def health_check(self, model_name: Optional[str] = None) -> bool:
        model = self.get_model(model_name)
        return model.health_check()

    def available_models(self) -> Dict[str, str]:
        return {name: repr(model) for name, model in self._models.items()}

    def unavailable_message(self) -> str:
        return (
            "Local AI model is unavailable. "
            f"Install Ollama, then run 'ollama pull {self.preferred_model}' or configure another local model and restart the agent."
        )
