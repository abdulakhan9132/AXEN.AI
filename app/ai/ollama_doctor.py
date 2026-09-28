from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from typing import Any

import requests

from .config import LocalAIConfig, load_local_config


@dataclass
class DoctorReport:
    ollama_installed: bool
    service_running: bool
    configured_model: str
    model_available: bool
    model_responding: bool
    structured_output: bool
    selected_model: str | None = None
    model_details: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "OLLAMA": "PASS" if self.ollama_installed else "FAIL",
            "MODEL": "PASS" if self.model_available else "FAIL",
            "CONNECTION": "PASS" if self.service_running and self.model_responding else "FAIL",
            "STRUCTURED OUTPUT": "PASS" if self.structured_output else "FAIL",
            "ollama_installed": self.ollama_installed,
            "service_running": self.service_running,
            "configured_model": self.configured_model,
            "model_available": self.model_available,
            "model_responding": self.model_responding,
            "structured_output": self.structured_output,
            "selected_model": self.selected_model,
            "model_details": self.model_details,
            "message": self.message,
        }


def _model_name_matches(configured: str, name: str) -> bool:
    return name == configured or name.split(":", 1)[0] == configured.split(":", 1)[0]


def inspect_ollama(config: LocalAIConfig | None = None, *, probe: bool = True) -> DoctorReport:
    settings = config or load_local_config()
    installed = shutil.which("ollama") is not None
    if not installed:
        return DoctorReport(False, False, settings.model, False, False, False, message="Ollama executable not found. Install Ollama and pull a small local model such as qwen2.5:1.5b.")
    try:
        response = requests.get(f"{settings.base_url}/api/tags", timeout=min(settings.timeout, 5.0))
        response.raise_for_status()
        tags = response.json().get("models", [])
    except (OSError, requests.RequestException, ValueError) as exc:
        return DoctorReport(True, False, settings.model, False, False, False, message=f"Ollama is installed but the service is unavailable: {exc}")
    names = [str(item.get("name", "")) for item in tags if isinstance(item, dict)]
    matching = next((item for item in tags if isinstance(item, dict) and _model_name_matches(settings.model, str(item.get("name", "")))), None)
    if matching is None:
        selected = names[0] if names else None
        return DoctorReport(True, True, settings.model, False, False, False, selected_model=selected, message=f"No configured model {settings.model!r} is installed. Available local models: {names or 'none'}.")
    details = matching.get("details", {}) if isinstance(matching, dict) else {}
    if not probe:
        return DoctorReport(True, True, settings.model, True, False, False, selected_model=str(matching.get("name")), model_details=details, message="Model is installed; probes were disabled.")
    responding = False
    structured = False
    try:
        probe_response = requests.post(
            f"{settings.base_url}/api/generate",
            json={"model": matching.get("name", settings.model), "prompt": "Reply with OK.", "stream": False, "options": {"num_predict": 4, "temperature": 0}},
            timeout=settings.timeout,
        )
        probe_response.raise_for_status()
        responding = bool(probe_response.json().get("response"))
    except (OSError, requests.RequestException, ValueError):
        responding = False
    try:
        structured_response = requests.post(
            f"{settings.base_url}/api/generate",
            json={"model": matching.get("name", settings.model), "prompt": "Return JSON only: {\"ok\": true}.", "format": "json", "stream": False, "options": {"num_predict": 16, "temperature": 0}},
            timeout=settings.timeout,
        )
        structured_response.raise_for_status()
        response_text = structured_response.json().get("response", "")
        parsed = json.loads(response_text) if isinstance(response_text, str) else None
        structured = isinstance(parsed, dict)
    except (OSError, requests.RequestException, ValueError):
        structured = False
    return DoctorReport(True, True, settings.model, True, responding, structured, selected_model=str(matching.get("name")), model_details=details, message="Ollama and the configured local model passed the available probes." if responding else "The configured model did not respond to the probe.")
