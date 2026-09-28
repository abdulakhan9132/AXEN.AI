from __future__ import annotations

import re
from typing import Any

from app.ai.model_manager import ModelManager
from reasoning.math.reasoning_controller import ReasoningController


_MATH_MARKERS = re.compile(r"\b(equation|calculate|compute|derivative|integral|probability|gcd|prime|determinant|matrix|mean|variance|solve|factor|distance|triangle|circle|fibonacci|sequence|inequality)\b|[=^]|\d+\s*[+*/-]\s*\d+")
_UNSUPPORTED_MARKERS = re.compile(r"\b(file|folder|shell|command|browser|website|scrape|install|delete)\b", re.IGNORECASE)


class TaskRouter:
    def __init__(self, model_manager: ModelManager | None = None, controller: ReasoningController | None = None) -> None:
        self.model_manager = model_manager or ModelManager()
        settings = self.model_manager.config
        self.controller = controller or ReasoningController(max_strategy_attempts=settings.max_strategy_attempts, max_formalization_attempts=settings.max_formalization_attempts)

    def classify(self, text: str) -> str:
        if _UNSUPPORTED_MARKERS.search(text):
            return "unsupported"
        if _MATH_MARKERS.search(text):
            return "mathematics"
        return "general"

    def route(self, text: str) -> dict[str, Any]:
        task_type = self.classify(text)
        if task_type == "mathematics":
            result = self.controller.solve(text, self.model_manager)
            payload = result.to_dict()
            payload["task_type"] = task_type
            return payload
        if task_type == "unsupported":
            return {"status": "unsupported_task", "task_type": task_type, "reason": "Only general conversation and mathematics are implemented by the Phase 4 router."}
        if not self.model_manager.health_check():
            return {"status": "llm_unavailable", "task_type": task_type, "reason": self.model_manager.unavailable_message()}
        try:
            response = self.model_manager.get_model().generate(text, system="Answer as a local AXEN assistant. Do not execute tools or claim unavailable capabilities.")
        except Exception as exc:
            return {"status": "llm_unavailable", "task_type": task_type, "reason": str(exc)}
        return {"status": "answered", "task_type": task_type, "answer": response}
