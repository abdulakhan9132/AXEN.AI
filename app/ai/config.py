from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LocalAIConfig:
    backend: str = "ollama"
    model: str = "qwen2.5:1.5b"
    temperature: float = 0.2
    context_length: int = 4096
    max_output_tokens: int = 512
    timeout: float = 60.0
    base_url: str = "http://localhost:11434"
    max_strategy_attempts: int = 5
    verification_required: bool = True
    max_subproblems: int = 5
    max_formalization_attempts: int = 3


def load_local_config(path: str | Path = "configs/axen_local.yaml") -> LocalAIConfig:
    config_path = Path(path)
    if not config_path.exists():
        return LocalAIConfig()
    data: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    llm = data.get("llm", {})
    reasoning = data.get("reasoning", {})
    if not isinstance(llm, dict) or not isinstance(reasoning, dict):
        raise ValueError("AXEN local configuration sections must be mappings.")
    return LocalAIConfig(
        backend=str(llm.get("backend", LocalAIConfig.backend)),
        model=str(llm.get("model", LocalAIConfig.model)),
        temperature=float(llm.get("temperature", LocalAIConfig.temperature)),
        context_length=int(llm.get("context_length", LocalAIConfig.context_length)),
        max_output_tokens=int(llm.get("max_output_tokens", LocalAIConfig.max_output_tokens)),
        timeout=float(llm.get("timeout", LocalAIConfig.timeout)),
        base_url=str(llm.get("base_url", LocalAIConfig.base_url)),
        max_strategy_attempts=int(reasoning.get("max_strategy_attempts", LocalAIConfig.max_strategy_attempts)),
        verification_required=bool(reasoning.get("verification_required", LocalAIConfig.verification_required)),
        max_subproblems=int(reasoning.get("max_subproblems", LocalAIConfig.max_subproblems)),
        max_formalization_attempts=int(reasoning.get("max_formalization_attempts", LocalAIConfig.max_formalization_attempts)),
    )
