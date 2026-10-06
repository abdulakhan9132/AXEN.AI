from __future__ import annotations

import os
from typing import Dict, Optional

from .base_model import AIModel
from .config import LocalAIConfig, load_local_config
from .local_model import LocalModel as OllamaCodingProvider
from .own_brain import AxenOwnBrainProvider
from .router import ModelRouter

class ModelManager:
    """Manage model selection, availability checks, and fallback behavior."""

    def __init__(self, preferred_model: Optional[str] = None, config_path: str = "configs/axen_local.yaml"):
        self.config: LocalAIConfig = load_local_config(config_path)
        
        # Default is ModelRouter
        self.preferred_model = preferred_model or os.getenv("LOCAL_MODEL_NAME", "router")
        
        self._models: Dict[str, AIModel] = {}
        if self.preferred_model == "router":
            self._models[self.preferred_model] = ModelRouter()
        elif self.preferred_model == "own_brain":
            self._models[self.preferred_model] = AxenOwnBrainProvider()
        else:
            self._models[self.preferred_model] = OllamaCodingProvider(model_name=self.preferred_model, config=self.config)

    def get_model(self, model_name: Optional[str] = None) -> AIModel:
        name = model_name or self.preferred_model
        if name not in self._models:
            if name == "router":
                self._models[name] = ModelRouter()
            elif name == "own_brain":
                self._models[name] = AxenOwnBrainProvider()
            else:
                self._models[name] = OllamaCodingProvider(model_name=name, config=self.config)
        return self._models[name]

    def health_check(self, model_name: Optional[str] = None) -> bool:
        model = self.get_model(model_name)
        return model.health_check()

    def available_models(self) -> Dict[str, str]:
        return {name: repr(model) for name, model in self._models.items()}

    def unavailable_message(self) -> str:
        if self.preferred_model in ("own_brain", "router"):
            return "AXEN Own Brain is unavailable. Start it with `node server.js`."
        return (
            "Local AI model is unavailable. "
            f"Install Ollama, then run 'ollama pull {self.preferred_model}' or configure another local model and restart the agent."
        )
