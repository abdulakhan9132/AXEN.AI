"""
axen/security/memory.py
=======================
Persistent Cybersecurity Memory & Knowledge Storage for AXEN.

Implements Section 6 & 9 of Ethical Hacking Charter:
- Records vulnerability discoveries, root causes, remediations, and lessons learned.
- Stores findings in SQLiteMemoryStore for long-term cross-session recall.
- Allows the agent to query previous security solutions before analyzing new targets.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from axen.memory.memory_models import ImportanceLevel, MemoryItem, MemoryType
from axen.memory.memory_store import SQLiteMemoryStore
from .models import VulnerabilityFinding

logger = logging.getLogger("axen.security.memory")


class CybersecurityMemory:
    """Manages long-term cybersecurity findings and remediation knowledge."""

    def __init__(self, store: Optional[SQLiteMemoryStore] = None) -> None:
        self.store = store or SQLiteMemoryStore()

    def record_vulnerability(self, finding: VulnerabilityFinding) -> int:
        """
        Record a vulnerability discovery with root-cause analysis and lessons learned.
        """
        key = f"sec_vuln_{finding.category.value.lower()}_{finding.finding_id[:8]}"
        payload = {
            "finding_id": finding.finding_id,
            "title": finding.title,
            "category": finding.category.value,
            "severity": finding.severity.value,
            "root_cause": finding.root_cause,
            "conditions_required": finding.conditions_required,
            "security_impact": finding.security_impact,
            "detection_method": finding.detection_method,
            "remediation_guidance": finding.remediation_guidance,
            "lessons_learned": finding.lessons_learned or "Always validate and sanitize before execution.",
            "status": finding.status.value,
        }

        item = MemoryItem(
            user_id="axen_security_agent",
            memory_type=MemoryType.TECHNICAL_FACT,
            key=key,
            value=json.dumps(payload),
            importance=ImportanceLevel.HIGH if finding.severity.value in ("CRITICAL", "HIGH") else ImportanceLevel.MEDIUM,
            confidence=1.0,
            source="cybersecurity_assessment",
            scope="security",
        )

        row_id = self.store.add(item)
        logger.info("Recorded cybersecurity memory [%s] -> row id %d", key, row_id)
        return row_id

    def recall_security_knowledge(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve previous security discoveries and remediations matching query.
        """
        memories = self.store.search_text(query, user_id="axen_security_agent", scope="security")
        results = []
        for m in memories[:limit]:
            try:
                data = json.loads(m.value)
                results.append(data)
            except Exception:
                results.append({"raw": m.value, "key": m.key})
        return results
