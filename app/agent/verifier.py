from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class VerificationResult:
    passed_: bool
    details: List[str]
    evidence: List[str]


class Verifier:
    """Validate that important actions match expected output."""

    def verify(self, step: Dict[str, Any], result: Dict[str, Any]) -> VerificationResult:
        tool_name = step.get("tool", "")
        details: List[str] = []
        evidence: List[str] = []

        if result.get("success") is False:
            return VerificationResult(False, [f"{tool_name} failed."], [str(result.get("error") or "error")])

        if tool_name == "filesystem.write":
            path = step.get("arguments", {}).get("path")
            if path and os.path.exists(path):
                details.append(f"File exists at {path}.")
                evidence.append(path)
            else:
                return VerificationResult(False, [f"No file was created at {path}."] , [path or "missing path"])
        elif tool_name == "filesystem.search":
            files = result.get("result", {}).get("files", [])
            if files:
                details.append(f"Found {len(files)} matching files.")
                evidence.extend(files[:5])
            else:
                details.append("Search completed successfully, but no files matched the filter.")
                evidence.append("no files matched")
        elif tool_name == "calculator.evaluate":
            value = result.get("result", {}).get("result")
            if value is not None:
                details.append(f"Calculator produced {value}.")
                evidence.append(str(value))
            else:
                return VerificationResult(False, ["Calculator returned no numeric output."], [])
        else:
            details.append(f"Tool {tool_name} executed successfully.")
            evidence.append(str(result.get("result")))

        return VerificationResult(True, details, evidence)
