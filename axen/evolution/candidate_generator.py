"""
axen/evolution/candidate_generator.py
======================================
Generates alternative code implementations for identified weaknesses.

Uses the existing AIModel interface to produce candidate improvements.
Each candidate preserves the original code and proposes modifications
along with a rationale.
"""

from __future__ import annotations

import difflib
import logging
import uuid
from typing import Any, Dict, List, Optional

from .models import Candidate, Experiment
from .weakness_detector import Weakness

logger = logging.getLogger("axen.evolution.candidate_generator")

# Default prompt template for code improvement generation
_IMPROVEMENT_PROMPT = """You are an expert Python engineer improving a specific module.

## Weakness Identified
{weakness_description}

## Evidence
{evidence}

## Target File
{target_file}

## Current Code
```python
{original_code}
```

## Instructions
1. Analyze the weakness and evidence carefully.
2. Propose a specific, targeted improvement to fix the identified issue.
3. Preserve all existing interfaces, function signatures, and public APIs.
4. Do not rename classes, functions, or modules.
5. Include clear comments explaining what changed and why.
6. The improvement must be deterministically testable.

## Output
Return ONLY the complete improved Python code. No explanations outside the code.
"""

_RATIONALE_PROMPT = """Given this code change, provide a brief 2-3 sentence rationale:

## Original Code (excerpt)
```python
{original_excerpt}
```

## Modified Code (excerpt)
```python
{modified_excerpt}
```

## Weakness Being Addressed
{weakness_description}

Rationale:"""


class CandidateGenerator:
    """
    Generates code improvement candidates using an AI model.

    Uses the AIModel interface (generate method) to produce alternative
    implementations targeting identified weaknesses.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        max_candidates: int = 3,
        temperature_range: tuple[float, float] = (0.3, 0.8),
    ) -> None:
        """
        Args:
            model: Any object with a generate(prompt: str) -> str method.
                   Compatible with AIModel, Ollama adapters, or the
                   AXENFoundationModel. If None, rule-based fallback mutations are used.
            max_candidates: Maximum number of candidates to generate per weakness.
            temperature_range: Range of temperatures for diversity.
        """
        self.model = model
        self.max_candidates = max_candidates
        self.temperature_range = temperature_range

    def generate_candidates(
        self,
        experiment: Optional[Experiment] = None,
        weakness: Optional[Weakness] = None,
        original_code: str = "",
        parent_id: Optional[str] = None,
        num_candidates: Optional[int] = None,
        experiment_id: Optional[str] = None,
        count: Optional[int] = None,
    ) -> List[Candidate]:
        """
        Generate multiple candidate improvements for a weakness.
        """
        n = num_candidates or count or self.max_candidates
        exp_id = experiment.experiment_id if experiment else (experiment_id or f"exp_{uuid.uuid4().hex[:8]}")
        desc = weakness.description if weakness else "General performance improvement"
        t_file = weakness.target_file if weakness else ""
        w_id = weakness.weakness_id if weakness else "weak_0"
        w_cat = weakness.category if weakness else "general"

        candidates: List[Candidate] = []

        evidence_str = "\n".join(f"- {e}" for e in weakness.evidence) if weakness and weakness.evidence else "No specific evidence provided."

        prompt = _IMPROVEMENT_PROMPT.format(
            weakness_description=desc,
            evidence=evidence_str,
            target_file=t_file,
            original_code=original_code,
        )

        for i in range(n):
            try:
                # Generate candidate code using the model (or fallback mutation)
                proposed_code = self._generate_code(prompt, candidate_index=i, original_code=original_code)
                if not proposed_code:
                    continue

                # Compute diff summary
                diff_summary = self._compute_diff(original_code, proposed_code)

                # Generate rationale
                rationale = self._generate_rationale(
                    original=original_code,
                    proposed=proposed_code,
                    weakness_desc=desc,
                )

                candidate = Candidate(
                    experiment_id=exp_id,
                    parent_id=parent_id,
                    target_file=t_file,
                    original_code=original_code,
                    proposed_code=proposed_code,
                    code=proposed_code,
                    diff_summary=diff_summary,
                    diff=diff_summary,
                    rationale=rationale,
                    metadata={
                        "weakness_id": w_id,
                        "candidate_index": i,
                        "weakness_category": w_cat,
                    },
                )
                candidates.append(candidate)
                logger.info(
                    "Generated candidate %s for experiment %s (index %d)",
                    candidate.candidate_id, exp_id, i,
                )

            except Exception as exc:
                logger.warning(
                    "Failed to generate candidate %d for experiment %s: %s",
                    i, exp_id, exc,
                )
                continue

        return candidates

    def _generate_code(self, prompt: str, candidate_index: int = 0, original_code: str = "") -> str:
        """Generate code using the model, extracting clean Python from the response."""
        if not self.model or not hasattr(self.model, "generate"):
            # Clean fallback mutation for testing/offline mode
            return f"# Variant {candidate_index + 1}\n{original_code.rstrip()}\n"

        try:
            # Use generate method compatible with AIModel interface
            if hasattr(self.model, "generate"):
                # Check if model accepts a string prompt directly
                from inspect import signature
                sig = signature(self.model.generate)
                params = list(sig.parameters.keys())

                if params and params[0] in ("request", "prompt"):
                    try:
                        from app.context.models import ModelRequest, AssembledContext
                        request = ModelRequest(context=AssembledContext(messages_for_model=[{"role": "user", "content": prompt}]),
                                               generation_settings={"temperature": 0.2}, session_metadata={"purpose": "candidate_proposal"})
                        raw = self.model.generate(request)
                    except (ImportError, TypeError):
                        raw = self.model.generate(prompt)
                else:
                    raw = self.model.generate(prompt)

                return self._extract_python_code(str(raw))
            return f"# Variant {candidate_index + 1}\n{original_code.rstrip()}\n"

        except Exception as exc:
            logger.warning("Code generation failed (index %d): %s", candidate_index, exc)
            return f"# Fallback Variant {candidate_index + 1}\n{original_code.rstrip()}\n"

    def _extract_python_code(self, response: str) -> str:
        """Extract Python code from model response, handling markdown fences."""
        # Try to extract from ```python ... ``` blocks
        import re
        pattern = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
        matches = pattern.findall(response)
        if matches:
            # Return the longest code block (most likely the full implementation)
            return max(matches, key=len).strip()

        # Try generic code blocks
        pattern = re.compile(r"```\s*\n(.*?)```", re.DOTALL)
        matches = pattern.findall(response)
        if matches:
            return max(matches, key=len).strip()

        # If no code blocks, check if the response looks like valid Python
        stripped = response.strip()
        if stripped.startswith(("import ", "from ", "def ", "class ", '"""', "'''")):
            return stripped

        return stripped

    def _compute_diff(self, original: str, proposed: str) -> str:
        """Compute a unified diff summary between original and proposed code."""
        orig_lines = original.splitlines(keepends=True)
        prop_lines = proposed.splitlines(keepends=True)
        diff = difflib.unified_diff(
            orig_lines, prop_lines,
            fromfile="original", tofile="proposed",
            lineterm="",
        )
        diff_text = "\n".join(list(diff)[:100])  # Limit diff size
        return diff_text

    def _generate_rationale(
        self,
        original: str = "",
        proposed: str = "",
        weakness_desc: str = "",
        original_code: str = "",
        modified_code: str = "",
        weakness_description: str = "",
    ) -> str:
        """Generate a brief rationale for the code change."""
        orig = original or original_code
        prop = proposed or modified_code
        desc = weakness_desc or weakness_description

        try:
            # Take first 500 chars of each for the rationale prompt
            orig_excerpt = orig[:500]
            mod_excerpt = prop[:500]

            prompt = _RATIONALE_PROMPT.format(
                original_excerpt=orig_excerpt,
                modified_excerpt=mod_excerpt,
                weakness_description=desc,
            )

            if hasattr(self.model, "generate"):
                try:
                    from app.context.models import ModelRequest, AssembledContext
                    request = ModelRequest(context=AssembledContext(messages_for_model=[{"role": "user", "content": prompt}]),
                                           generation_settings={"temperature": 0.2}, session_metadata={"purpose": "candidate_rationale"})
                    raw = self.model.generate(request)
                except (ImportError, TypeError):
                    raw = self.model.generate(prompt)
                return str(raw).strip()[:500]
            return "Improvement candidate generated to address identified weakness."
        except Exception:
            return "Improvement candidate generated to address identified weakness."
