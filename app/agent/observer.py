from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Observation:
    facts: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.errors:
            return "; ".join(self.errors)
        if self.facts:
            return "; ".join(self.facts)
        return "Observation recorded without strong evidence."


class Observer:
    """Separate facts from assumptions to support safer decisions."""

    def observe(self, result: Dict[str, Any], *, context: Optional[str] = None) -> Observation:
        observation = Observation()
        if not isinstance(result, dict):
            observation.assumptions.append("Tool returned an unexpected format.")
            return observation

        if result.get("success") is True:
            observation.facts.append("Tool execution success observed.")
        elif result.get("success") is False:
            observation.errors.append(str(result.get("error") or "Tool execution failed."))

        payload = result.get("result") if isinstance(result.get("result"), dict) else {}
        if payload:
            evidence = payload.get("success")
            if evidence is True:
                observation.facts.append("Observed tool data confirms completion.")
            elif evidence is False:
                observation.assumptions.append("Tool output may be incomplete or stale.")

        if result.get("result") is not None and isinstance(result.get("result"), dict):
            if "files" in result["result"]:
                observation.facts.append(f"Found {len(result['result']['files'])} matching items.")
            if "error" in result["result"]:
                observation.errors.append(str(result["result"]["error"]))

        if context:
            observation.notes.append(context)
        return observation
