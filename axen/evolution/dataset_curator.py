"""
axen/evolution/dataset_curator.py
=================================
Quality gating and curation for neural training datasets.

Enforces strict safety invariants:
1. Rejects unverified AI-generated outputs
2. Sanitizes input/output length and structure
3. Deduplicates samples using content hashing
4. Supports ingesting verified tool execution traces as instruction pairs
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("axen.evolution.dataset_curator")


@dataclass
class CuratedSample:
    """A verified, sanitized training sample."""

    instruction: str
    output: str
    source: str  # e.g., "human_verified", "tool_trace_verified", "benchmark_success"
    quality_score: float = 1.0
    content_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content_hash:
            raw = f"{self.instruction.strip()}###{self.output.strip()}"
            self.content_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_record(self) -> Dict[str, Any]:
        return {
            "instruction": self.instruction,
            "output": self.output,
            "source": self.source,
            "quality_score": self.quality_score,
            "hash": self.content_hash,
            "metadata": self.metadata,
            "verified": True,
        }


class DatasetCurator:
    """
    Validates and sanitizes training data for the AXEN foundation model.
    """

    def __init__(
        self,
        min_instruction_len: int = 5,
        max_instruction_len: int = 2000,
        min_output_len: int = 2,
        max_output_len: int = 4000,
    ) -> None:
        self.min_instruction_len = min_instruction_len
        self.max_instruction_len = max_instruction_len
        self.min_output_len = min_output_len
        self.max_output_len = max_output_len
        self._seen_hashes: Set[str] = set()

    def validate_sample(
        self,
        instruction: str,
        output: str,
        is_verified: bool = True,
        source: str = "curated",
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a sample satisfies curation standards.

        Returns:
            (valid, reason_if_invalid)
        """
        # Hard Rule: Unverified model outputs are strictly rejected
        if not is_verified:
            return False, "Rejected: Sample is unverified by test or human."

        inst = instruction.strip()
        out = output.strip()

        if len(inst) < self.min_instruction_len:
            return False, f"Instruction too short (<{self.min_instruction_len} chars)"
        if len(inst) > self.max_instruction_len:
            return False, f"Instruction exceeds maximum length ({self.max_instruction_len})"
        if len(out) < self.min_output_len:
            return False, f"Output too short (<{self.min_output_len} chars)"
        if len(out) > self.max_output_len:
            return False, f"Output exceeds maximum length ({self.max_output_len})"

        # Deduplication check
        raw = f"{inst}###{out}"
        h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if h in self._seen_hashes:
            return False, "Duplicate sample detected."

        return True, None

    def curate_samples(
        self,
        raw_samples: List[Dict[str, Any]],
        strict_verified: bool = True,
    ) -> List[CuratedSample]:
        """
        Process a list of raw sample dicts and return only verified, sanitized samples.
        """
        curated: List[CuratedSample] = []
        for s in raw_samples:
            instruction = s.get("instruction", s.get("prompt", ""))
            output = s.get("output", s.get("response", ""))
            is_verified = s.get("verified", not strict_verified)
            source = s.get("source", "batch_curation")

            is_valid, reason = self.validate_sample(
                instruction=instruction,
                output=output,
                is_verified=is_verified,
                source=source,
            )
            if is_valid:
                sample = CuratedSample(
                    instruction=instruction.strip(),
                    output=output.strip(),
                    source=source,
                    quality_score=float(s.get("quality_score", 1.0)),
                    metadata=s.get("metadata", {}),
                )
                self._seen_hashes.add(sample.content_hash)
                curated.append(sample)
            else:
                logger.debug("Discarding sample: %s", reason)

        logger.info(
            "Curation complete: %d accepted out of %d candidates.",
            len(curated),
            len(raw_samples),
        )
        return curated

    def convert_tool_trace(
        self,
        user_prompt: str,
        tool_name: str,
        arguments: Dict[str, Any],
        observation: str,
        final_answer: str,
        verifier_passed: bool,
    ) -> Optional[CuratedSample]:
        """
        Transform a successful tool execution trace into a high-quality training sample.
        """
        if not verifier_passed:
            return None

        inst = f"User Request: {user_prompt.strip()}"
        out = (
            f"Action: Call tool `{tool_name}` with args {json.dumps(arguments)}\n"
            f"Observation: {observation.strip()}\n"
            f"Final Answer: {final_answer.strip()}"
        )

        is_valid, _ = self.validate_sample(inst, out, is_verified=True, source="tool_trace")
        if not is_valid:
            return None

        sample = CuratedSample(
            instruction=inst,
            output=out,
            source="tool_trace_verified",
            metadata={"tool_name": tool_name},
        )
        self._seen_hashes.add(sample.content_hash)
        return sample
