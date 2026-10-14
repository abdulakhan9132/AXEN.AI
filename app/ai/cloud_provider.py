"""
app/ai/cloud_provider.py
========================
Adapter for optional cloud model providers (OpenAI, OpenRouter, Anthropic).

Key Principles:
- Local-First Privacy: Cloud calls are strictly disabled unless an API key is explicitly configured.
- Minimal Dependencies: Implemented using standard requests/urllib.
- Full Streaming & Structured Output support.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Iterator, Optional

import requests

from .base_model import AIModel
from app.context.models import ModelRequest

logger = logging.getLogger("axen.ai.cloud")

DEFAULT_ENDPOINTS = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "anthropic": "https://api.anthropic.com/v1",
}


class CloudModelProvider(AIModel):
    """Adapter for remote OpenAI/OpenRouter/Anthropic API providers."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        provider: str = "openrouter",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        temperature: float = 0.3,
        max_output_tokens: int = 1024,
    ):
        self.provider = provider.lower()
        self.model_name = model_name
        self.timeout = timeout
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        # Resolve API key from environment if not passed explicitly
        if api_key:
            self.api_key = api_key
        elif self.provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY", "")
        elif self.provider == "anthropic":
            self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        else:
            self.api_key = os.getenv("OPENROUTER_API_KEY", "") or os.getenv("CLOUD_API_KEY", "")

        self.base_url = (base_url or DEFAULT_ENDPOINTS.get(self.provider, DEFAULT_ENDPOINTS["openrouter"])).rstrip("/")

    def health_check(self) -> bool:
        """Check if provider has a configured API key."""
        if not self.api_key:
            return False
        # Lightweight check: verify key format isn't empty dummy
        return len(self.api_key.strip()) > 8

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "http://localhost:8000"
            headers["X-Title"] = "AXEN AI Local Assistant"
        elif self.provider == "anthropic":
            headers["x-api-key"] = self.api_key
            headers["anthropic-version"] = "2023-06-01"
            headers.pop("Authorization", None)
        return headers

    def generate(self, request: ModelRequest) -> str:
        if not self.health_check():
            raise RuntimeError(
                f"Cloud provider '{self.provider}' is unavailable. Please configure an API key."
            )

        messages = request.context.messages_for_model if hasattr(request, "context") and request.context else []
        if not messages and hasattr(request, "current_message"):
            messages = [{"role": "user", "content": str(request.current_message)}]

        headers = self._build_headers()
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": request.generation_settings.get("temperature", self.temperature),
            "max_tokens": request.generation_settings.get("max_tokens", self.max_output_tokens),
            "stream": False,
        }

        if request.generation_settings.get("format") == "json":
            payload["response_format"] = {"type": "json_object"}

        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if choices:
            return str(choices[0].get("message", {}).get("content", ""))
        return ""

    def stream(self, request: ModelRequest) -> Iterator[str]:
        if not self.health_check():
            raise RuntimeError(
                f"Cloud provider '{self.provider}' is unavailable. Please configure an API key."
            )

        messages = request.context.messages_for_model if hasattr(request, "context") and request.context else []
        headers = self._build_headers()
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": request.generation_settings.get("temperature", self.temperature),
            "max_tokens": request.generation_settings.get("max_tokens", self.max_output_tokens),
            "stream": True,
        }

        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout, stream=True)
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8") if isinstance(line, bytes) else line
            if line_str.startswith("data: "):
                data_part = line_str[6:].strip()
                if data_part == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_part)
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                except Exception:
                    continue

    def structured_output(self, request: ModelRequest, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        req_json = ModelRequest(
            context=request.context,
            generation_settings={**request.generation_settings, "format": "json"},
            session_metadata=request.session_metadata,
            cancel_event=request.cancel_event,
        )
        text = self.generate(req_json)
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`\n ")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {"raw": text}

    def __repr__(self) -> str:
        key_masked = f"{self.api_key[:4]}...{self.api_key[-3:]}" if len(self.api_key) > 7 else "unset"
        return f"CloudModelProvider(provider={self.provider!r}, model={self.model_name!r}, key={key_masked})"
