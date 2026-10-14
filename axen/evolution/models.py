"""
axen/evolution/models.py
========================
Core data models for the AXEN Self-Evolution Engine.

Defines the structured representations for:
- Experiments: A single improvement attempt targeting a specific weakness
- Candidates: Alternative code implementations generated during an experiment
- BenchmarkResults: Deterministic evaluation metrics for a candidate
- EvolutionRecords: Parent-child version lineage for tracking improvement history
- ApprovalRequests: Human-in-the-loop approval gate records

All models are dataclass-based with JSON serialization, matching the patterns
used throughout the AXEN codebase (AgentState, ToolResult, MemoryItem).
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ExperimentStatus(str, Enum):
    """Lifecycle states for an evolution experiment."""
    CREATED = "created"
    DETECTING = "detecting"
    GENERATING = "generating"
    EVALUATING = "evaluating"
    SELECTING = "selecting"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    MERGED = "merged"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CandidateStatus(str, Enum):
    """Lifecycle states for a candidate implementation."""
    PENDING = "pending"
    EVALUATING = "evaluating"
    EVALUATED = "evaluated"
    PASSED = "passed"
    FAILED = "failed"
    SELECTED = "selected"
    APPROVED = "approved"
    REJECTED = "rejected"
    MERGED = "merged"


class ApprovalStatus(str, Enum):
    """Human approval decision states."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

@dataclass
class Experiment:
    """
    A single self-improvement experiment.

    Targets a specific weakness (identified via test failures or performance
    regressions) and tracks the full lifecycle from detection through
    candidate generation, evaluation, selection, and approval.
    """
    experiment_id: str = field(default_factory=lambda: f"exp_{uuid.uuid4().hex[:12]}")
    title: str = ""
    description: str = ""
    target_module: str = ""
    target_file: str = ""
    weakness_description: str = ""
    weakness_evidence: List[str] = field(default_factory=list)
    status: ExperimentStatus = ExperimentStatus.CREATED
    candidate_ids: List[str] = field(default_factory=list)
    selected_candidate_id: Optional[str] = None
    parent_experiment_id: Optional[str] = None
    generation: int = 0
    config: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())
    completed_at: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def transition_to(self, new_status: ExperimentStatus) -> None:
        """Safely transition experiment to a new status."""
        self.status = new_status
        self.updated_at = dt.datetime.now(dt.timezone.utc).isoformat()
        if new_status in (
            ExperimentStatus.APPROVED,
            ExperimentStatus.REJECTED,
            ExperimentStatus.MERGED,
            ExperimentStatus.FAILED,
            ExperimentStatus.CANCELLED,
        ):
            if not self.completed_at:
                self.completed_at = dt.datetime.now(dt.timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dictionary representation."""
        d = asdict(self)
        d["status"] = self.status.value if isinstance(self.status, ExperimentStatus) else str(self.status)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experiment":
        """Reconstruct an Experiment from a dictionary."""
        data = dict(data)
        if "status" in data and isinstance(data["status"], str):
            try:
                data["status"] = ExperimentStatus(data["status"])
            except ValueError:
                data["status"] = ExperimentStatus.CREATED
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Candidate
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """
    An alternative code implementation generated during an experiment.

    Each candidate has a parent_id forming a version lineage tree.
    Code diffs, benchmark results, and approval status are tracked.
    """
    candidate_id: str = field(default_factory=lambda: f"cand_{uuid.uuid4().hex[:12]}")
    experiment_id: str = ""
    parent_id: Optional[str] = None
    generation: int = 0
    target_file: str = ""
    original_code: str = ""
    proposed_code: str = ""
    code: str = ""
    diff_summary: str = ""
    diff: str = ""
    rationale: str = ""
    status: CandidateStatus = CandidateStatus.PENDING
    benchmark_ids: List[str] = field(default_factory=list)
    benchmark_result: Optional[Any] = None
    error: Optional[str] = None
    correctness_score: float = 0.0
    performance_score: float = 0.0
    resource_score: float = 0.0
    composite_score: float = 0.0
    held_out_score: float = 0.0
    created_at: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.proposed_code and self.code:
            self.proposed_code = self.code
        elif not self.code and self.proposed_code:
            self.code = self.proposed_code
        if not self.diff_summary and self.diff:
            self.diff_summary = self.diff
        elif not self.diff and self.diff_summary:
            self.diff = self.diff_summary

    def transition_to(self, new_status: CandidateStatus) -> None:
        """Safely transition candidate to a new status."""
        self.status = new_status
        self.updated_at = dt.datetime.now(dt.timezone.utc).isoformat()

    def compute_composite_score(
        self,
        correctness_weight: float = 0.6,
        performance_weight: float = 0.25,
        resource_weight: float = 0.15,
    ) -> float:
        """
        Compute weighted composite score from individual metrics.

        Default weights prioritize correctness (60%) over performance (25%)
        and resource efficiency (15%).
        """
        self.composite_score = round(
            self.correctness_score * correctness_weight
            + self.performance_score * performance_weight
            + self.resource_score * resource_weight,
            4,
        )
        return self.composite_score

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dictionary representation."""
        d = asdict(self)
        d["status"] = self.status.value if isinstance(self.status, CandidateStatus) else str(self.status)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Candidate":
        """Reconstruct a Candidate from a dictionary."""
        data = dict(data)
        if "status" in data and isinstance(data["status"], str):
            try:
                data["status"] = CandidateStatus(data["status"])
            except ValueError:
                data["status"] = CandidateStatus.PENDING
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Benchmark Result
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """
    Deterministic evaluation result for a candidate.

    Stores correctness, performance, and resource metrics.
    Separates in-distribution (train) scores from held-out scores
    for anti-overfitting detection.
    """
    result_id: str = field(default_factory=lambda: f"bench_{uuid.uuid4().hex[:12]}")
    candidate_id: str = ""
    experiment_id: str = ""
    suite_name: str = "default"
    # Correctness metrics
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    pass_rate: float = 0.0
    # Held-out anti-overfitting metrics
    held_out_total: int = 0
    held_out_passed: int = 0
    held_out_pass_rate: float = 0.0
    # Performance metrics
    avg_execution_time_ms: float = 0.0
    max_execution_time_ms: float = 0.0
    p95_execution_time_ms: float = 0.0
    # Resource metrics
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    # Comparison against baseline
    baseline_pass_rate: float = 0.0
    baseline_held_out_pass_rate: Optional[float] = None
    improvement_delta: float = 0.0
    is_improvement: bool = False
    is_overfitting: bool = False
    # Execution details
    failed_test_names: List[str] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compute_pass_rate(self) -> None:
        """Compute pass rates from test counts."""
        if self.total_tests > 0:
            self.pass_rate = round(self.passed_tests / self.total_tests, 4)
        if self.held_out_total > 0:
            self.held_out_pass_rate = round(self.held_out_passed / self.held_out_total, 4)

    def detect_overfitting(self, threshold: float = 0.5) -> bool:
        """
        Detect if candidate is overfitting to in-distribution tests.

        Overfitting is flagged when:
        - Held-out improvement < threshold * in-distribution improvement
        - Or held-out pass rate actually decreased vs baseline
        """
        if self.held_out_total == 0:
            return False

        in_dist_improvement = self.pass_rate - self.baseline_pass_rate
        baseline_held_out = (self.baseline_pass_rate if self.baseline_held_out_pass_rate is None
                             else self.baseline_held_out_pass_rate)
        held_out_improvement = self.held_out_pass_rate - baseline_held_out

        if held_out_improvement < 0:
            self.is_overfitting = True
            return True

        if in_dist_improvement > 0 and held_out_improvement < threshold * in_dist_improvement:
            self.is_overfitting = True
            return True

        self.is_overfitting = False
        return False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkResult":
        data = dict(data)
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Evolution Record (Version Lineage)
# ---------------------------------------------------------------------------

@dataclass
class EvolutionRecord:
    """
    Tracks parent-child version relationships for improvement history.

    Forms a directed tree where each node is a code version and edges
    represent evolutionary improvements from parent to child.
    """
    record_id: str = field(default_factory=lambda: f"evo_{uuid.uuid4().hex[:12]}")
    parent_id: Optional[str] = None
    child_id: str = ""
    experiment_id: str = ""
    parent_version: str = ""
    child_version: str = ""
    target_module: str = ""
    target_file: str = ""
    improvement_type: str = ""  # "code", "strategy", "model_weights", "tool", "prompt"
    improvement_summary: str = ""
    correctness_delta: float = 0.0
    performance_delta: float = 0.0
    generation: int = 0
    was_accepted: bool = False
    accepted_by: str = ""
    created_at: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvolutionRecord":
        data = dict(data)
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Approval Request
# ---------------------------------------------------------------------------

@dataclass
class ApprovalRequest:
    """
    Human-in-the-loop approval gate record.

    Every code merge or weight activation must pass through an approval
    request that a human explicitly approves or rejects.
    """
    request_id: str = field(default_factory=lambda: f"appr_{uuid.uuid4().hex[:12]}")
    experiment_id: str = ""
    candidate_id: str = ""
    target_file: str = ""
    request_type: str = "code_merge"  # "code_merge", "weight_activation", "strategy_change"
    title: str = ""
    description: str = ""
    diff_summary: str = ""
    diff: str = ""
    risk_level: str = "medium"
    notes: str = ""
    reviewer: str = ""
    benchmark_summary: Dict[str, Any] = field(default_factory=dict)
    status: ApprovalStatus = ApprovalStatus.PENDING
    requested_at: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())
    decided_at: Optional[str] = None
    decided_by: str = ""
    rejection_reason: str = ""
    rollback_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.diff_summary and self.diff:
            self.diff_summary = self.diff
        elif not self.diff and self.diff_summary:
            self.diff = self.diff_summary
        if not self.title and self.target_file:
            self.title = f"Approve candidate for {self.target_file}"

    def approve(self, reviewer: str = "human", notes: str = "", decided_by: Optional[str] = None) -> None:
        """Mark this request as approved."""
        self.status = ApprovalStatus.APPROVED
        self.decided_at = dt.datetime.now(dt.timezone.utc).isoformat()
        self.decided_by = decided_by or reviewer
        self.reviewer = self.decided_by
        self.notes = notes

    def reject(self, reason: str = "", reviewer: str = "human", decided_by: Optional[str] = None) -> None:
        """Mark this request as rejected."""
        self.status = ApprovalStatus.REJECTED
        self.decided_at = dt.datetime.now(dt.timezone.utc).isoformat()
        self.decided_by = decided_by or reviewer
        self.reviewer = self.decided_by
        self.rejection_reason = reason

    def rollback(self, rollback_info: Dict[str, Any] | None = None) -> None:
        """Mark a previously approved request as rolled back."""
        self.status = ApprovalStatus.ROLLED_BACK
        self.decided_at = dt.datetime.now(dt.timezone.utc).isoformat()
        if rollback_info:
            self.rollback_info = rollback_info

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value if isinstance(self.status, ApprovalStatus) else str(self.status)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApprovalRequest":
        data = dict(data)
        if "status" in data and isinstance(data["status"], str):
            try:
                data["status"] = ApprovalStatus(data["status"])
            except ValueError:
                data["status"] = ApprovalStatus.PENDING
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
