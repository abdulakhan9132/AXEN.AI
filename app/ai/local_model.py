from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterator, Optional

import requests

from .base_model import AIModel
from .config import LocalAIConfig, load_local_config


class LocalModel(AIModel):
    """Adapter for a local Ollama-compatible runtime."""

    def __init__(self, model_name: Optional[str] = None, base_url: Optional[str] = None, config: LocalAIConfig | None = None):
        settings = config or load_local_config()
        self.model_name = model_name or os.getenv("LOCAL_MODEL_NAME", settings.model)
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", settings.base_url)).rstrip("/")
        self.temperature = settings.temperature
        self.context_length = settings.context_length
        self.max_output_tokens = settings.max_output_tokens
        self.timeout = settings.timeout

    def health_check(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=min(self.timeout, 2.5))
            return response.status_code == 200
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        response_format: str = "text",
    ) -> str:
        if not self.health_check():
            raise RuntimeError(
                "Local AI model is unavailable. Install Ollama and pull a model, or use the framework fallback."
            )

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "system": system or "You are a helpful local AI agent.",
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.context_length,
                "num_predict": self.max_output_tokens,
            },
        }
        if response_format == "json":
            payload["format"] = "json"
        if context:
            payload["context"] = json.dumps(context)

        response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return str(data.get("response", ""))

    def stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        response_format: str = "text",
    ) -> Iterator[str]:
        if not self.health_check():
            raise RuntimeError("Local AI model is unavailable.")

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "system": system or "You are a helpful local AI agent.",
        }
        if context:
            payload["context"] = json.dumps(context)

        response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout, stream=True)
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if "response" in chunk:
                yield chunk["response"]

    def structured_output(
        self,
        prompt: str,
        *,
        schema: Optional[Dict[str, Any]] = None,
        system: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        text = self.generate(prompt, system=system, context=context, response_format="json")
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`\n ")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return {"raw": text}

    def __repr__(self) -> str:
        return f"LocalModel(model_name={self.model_name!r}, base_url={self.base_url!r})"
