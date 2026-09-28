from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional


class AIModel(ABC):
    """Shared interface for all model adapters."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        response_format: str = "text",
    ) -> str:
        """Generate a text response."""

    @abstractmethod
    def stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        response_format: str = "text",
    ) -> Iterator[str]:
        """Stream model output incrementally."""

    @abstractmethod
    def structured_output(
        self,
        prompt: str,
        *,
        schema: Optional[Dict[str, Any]] = None,
        system: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return validated structured JSON-like output."""

    @abstractmethod
    def health_check(self) -> bool:
        """Return whether the model runtime is available."""
