"""
axen/knowledge/grounding_verifier.py
====================================
Deterministic, local grounding verification for AXEN RAG.

Evaluates whether the generated answer is grounded in retrieved source passages.
Identifies:
- Supported claims
- Unsupported claims
- Grounding ratio (0.0 to 1.0)
- Grounding status: GROUNDED, PARTIALLY_GROUNDED, UNSUPPORTED, NO_SOURCE
"""

from __future__ import annotations

import re
from typing import List, Sequence

from .document_models import DocumentChunk, GroundingResult, GroundingStatus, ScoredChunk


class GroundingVerifier:
    """Verifies that generated answers do not hallucinate facts absent from sources."""

    _STOPWORDS = frozenset({
        "a", "an", "the", "is", "are", "was", "were", "and", "or", "in", "on", "at",
        "to", "for", "with", "by", "of", "from", "that", "this", "it", "as", "be",
        "has", "have", "had", "can", "will", "would", "should", "not", "but",
    })

    def verify_grounding(
        self,
        answer: str,
        sources: Sequence[DocumentChunk | ScoredChunk],
        strictness: float = 0.5,
    ) -> GroundingResult:
        """
        Verify claims in `answer` against `sources`.
        Returns GroundingResult with status and claim breakdown.
        """
        if not sources:
            return GroundingResult(
                status=GroundingStatus.NO_SOURCE,
                grounded_ratio=0.0,
                detail="No source passages available to verify against.",
            )

        # Collect source text
        source_texts = []
        source_names = []
        for s in sources:
            if isinstance(s, ScoredChunk):
                source_texts.append(s.chunk.text.lower())
                source_names.append(s.citation)
            else:
                source_texts.append(s.text.lower())
                source_names.append(s.source_location or "source")

        combined_source = " ".join(source_texts)

        # Split answer into candidate sentences/claims
        sentences = [s.strip() for s in re.split(r"[.\n;]+", answer) if len(s.strip()) > 15]
        if not sentences:
            return GroundingResult(
                status=GroundingStatus.GROUNDED,
                grounded_ratio=1.0,
                sources_used=list(set(source_names)),
                detail="Answer contains no verifiable claims.",
            )

        supported: list[str] = []
        unsupported: list[str] = []

        for sent in sentences:
            words = [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", sent)]
            content_words = [w for w in words if w not in self._STOPWORDS and len(w) > 2]
            if not content_words:
                continue

            # Check overlap of content words in source
            matches = sum(1 for w in content_words if w in combined_source)
            ratio = matches / len(content_words)

            if ratio >= strictness:
                supported.append(sent)
            else:
                unsupported.append(sent)

        total = len(supported) + len(unsupported)
        grounded_ratio = len(supported) / total if total > 0 else 1.0

        if grounded_ratio >= 0.75:
            status = GroundingStatus.GROUNDED
        elif grounded_ratio >= 0.40:
            status = GroundingStatus.PARTIALLY_GROUNDED
        else:
            status = GroundingStatus.UNSUPPORTED

        return GroundingResult(
            status=status,
            grounded_ratio=grounded_ratio,
            supported_claims=supported,
            unsupported_claims=unsupported,
            sources_used=list(set(source_names)),
            detail=f"Grounding ratio: {grounded_ratio:.2f} ({len(supported)}/{total} claims supported)",
        )
