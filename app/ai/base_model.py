from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional
from app.context.models import ModelRequest


class AIModel(ABC):
    """Shared interface for all model adapters."""

    @abstractmethod
    def generate(self, request: ModelRequest) -> str:
        """Generate a text response."""

    @abstractmethod
    def stream(self, request: ModelRequest) -> Iterator[str]:
        """Stream model output incrementally."""

    @abstractmethod
    def structured_output(self, request: ModelRequest, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return validated structured JSON-like output."""

    @abstractmethod
    def health_check(self) -> bool:
        """Return whether the model runtime is available."""
