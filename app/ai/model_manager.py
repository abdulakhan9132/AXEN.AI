from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from .base_model import AIModel
from .config import LocalAIConfig, load_local_config
from .local_model import LocalModel as OllamaCodingProvider
from .own_brain import AxenOwnBrainProvider
from .router import ModelRouter
from .cloud_provider import CloudModelProvider

import time

logger = logging.getLogger("axen.ai.manager")


class ModelManager:
    """
    Unified AI Model Management System for AEXN AI.
    Supports local open-weight runtimes (Ollama, PyTorch HF, Node Brain)
    and optional local-first cloud API providers.
    """

    def __init__(self, preferred_model: Optional[str] = None, config_path: str = "configs/axen_local.yaml"):
        self.config: LocalAIConfig = load_local_config(config_path)
        self.preferred_model = preferred_model or os.getenv("LOCAL_MODEL_NAME", "router")
        self.active_model_name = self.preferred_model
        self._models: Dict[str, AIModel] = {}
        self._health_cache: Dict[str, tuple[float, bool]] = {}

    def get_model(self, model_name: Optional[str] = None) -> AIModel:
        name = (model_name or self.active_model_name).lower()
        if name not in self._models:
            if name == "router":
                self._models[name] = ModelRouter()
            elif name in ("own_brain", "axen_own_brain"):
                self._models[name] = AxenOwnBrainProvider()
            elif name.startswith("cloud_") or name in ("openai", "openrouter", "anthropic"):
                provider_type = name.replace("cloud_", "")
                self._models[name] = CloudModelProvider(provider=provider_type)
            elif name.startswith("hf_") or name in ("tinyllama", "pytorch"):
                # Use local Ollama or HF provider
                self._models[name] = OllamaCodingProvider(model_name=self.config.model, config=self.config)
            else:
                self._models[name] = OllamaCodingProvider(model_name=name, config=self.config)
        return self._models[name]

    def switch_model(self, model_name: str) -> bool:
        """Switch the active generation model."""
        clean_name = model_name.strip().lower()
        self.active_model_name = clean_name
        self.preferred_model = clean_name
        # Pre-instantiate to warm up adapter
        _ = self.get_model(clean_name)
        return True

    def health_check(self, model_name: Optional[str] = None) -> bool:
        target = (model_name or self.active_model_name).lower()
        now = time.monotonic()
        if target in self._health_cache:
            ts, val = self._health_cache[target]
            if now - ts < 15.0:
                return val
        model = self.get_model(target)
        val = model.health_check()
        self._health_cache[target] = (now, val)
        return val

    def list_models(self) -> List[Dict[str, Any]]:
        """Return full catalog of detected and available models."""
        models = [
            {
                "id": "router",
                "name": "Auto Router (Intelligent)",
                "provider": "hybrid",
                "type": "local",
                "context_length": 4096,
                "status": "active" if self.active_model_name == "router" else "ready",
                "available": True,
                "description": "Routes greetings to Node Brain, coding to Ollama/Qwen, and math to verified solvers.",
            },
            {
                "id": "own_brain",
                "name": "AXEN Local Brain (Neural)",
                "provider": "node_transformer",
                "type": "local",
                "context_length": 1024,
                "status": "active" if self.active_model_name == "own_brain" else "ready",
                "available": self.health_check("own_brain"),
                "description": "Ultra-lightweight zero-GPU local neural network running on Node.js.",
            },
            {
                "id": "qwen2.5:1.5b",
                "name": "Qwen 2.5 Coder (1.5B Q4)",
                "provider": "ollama",
                "type": "local",
                "context_length": self.config.context_length,
                "status": "active" if self.active_model_name in ("qwen2.5:1.5b", "ollama") else "ready",
                "available": self.health_check("qwen2.5:1.5b"),
                "description": "Specialized code generation & technical reasoning model optimized for 4GB VRAM.",
            },
            {
                "id": "tinyllama",
                "name": "TinyLlama (1.1B PyTorch CUDA)",
                "provider": "pytorch_hf",
                "type": "local",
                "context_length": 2048,
                "status": "active" if self.active_model_name in ("tinyllama", "pytorch") else "ready",
                "available": True,
                "description": "Direct local PyTorch/Transformers inference with CUDA acceleration.",
            },
            {
                "id": "cloud_openrouter",
                "name": "OpenRouter Cloud API (Optional)",
                "provider": "openrouter",
                "type": "cloud",
                "context_length": 128000,
                "status": "active" if self.active_model_name == "cloud_openrouter" else "ready",
                "available": self.health_check("cloud_openrouter"),
                "description": "Optional cloud provider for frontier models (Claude 3.5 Sonnet, GPT-4o, DeepSeek).",
            },
        ]
        return models

    def available_models(self) -> Dict[str, str]:
        catalog = self.list_models()
        return {m["id"]: m["name"] for m in catalog if m.get("available")}

    def unavailable_message(self) -> str:
        if self.active_model_name in ("own_brain", "router"):
            return "AXEN Own Brain is unavailable. Start it with `node server.js` or switch to local Ollama."
        return (
            f"Model '{self.active_model_name}' is currently unavailable. "
            "Please ensure Ollama is running (`ollama serve`), or choose another local model."
        )
