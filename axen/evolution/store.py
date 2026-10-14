"""
axen/evolution/store.py
=======================
SQLite-backed persistent storage for the AXEN Self-Evolution Engine.

Follows the same thread-safe WAL-mode pattern established by
axen.memory.memory_store.SQLiteMemoryStore.

Tables:
- experiments: Evolution experiment lifecycle records
- candidates: Alternative code implementations with version lineage
- benchmark_results: Deterministic evaluation metrics
- evolution_history: Parent-child version relationship graph
- approvals: Human-in-the-loop approval gate records
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import (
    ApprovalRequest,
    ApprovalStatus,
    BenchmarkResult,
    Candidate,
    CandidateStatus,
    EvolutionRecord,
    Experiment,
    ExperimentStatus,
)

logger = logging.getLogger("axen.evolution.store")

DEFAULT_DB_PATH = Path("data/axen_evolution.db")


class EvolutionStore:
    """Thread-safe SQLite storage engine for evolution data."""

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    class _ConnectionContext:
        def __init__(self, store: "EvolutionStore") -> None:
            self.store = store
            self.conn: Optional[sqlite3.Connection] = None

        def __enter__(self) -> sqlite3.Connection:
            self.conn = sqlite3.connect(
                str(self.store.db_path),
                timeout=15.0,
                check_same_thread=False,
            )
            self.conn.row_factory = sqlite3.Row
            if str(self.store.db_path) != ":memory:":
                try:
                    self.conn.execute("PRAGMA journal_mode=WAL;")
                    self.conn.execute("PRAGMA synchronous=NORMAL;")
                except Exception as e:
                    logger.debug("Could not set WAL pragma: %s", e)
            self.conn.__enter__()
            return self.conn

        def __exit__(self, exc_type, exc_val, exc_tb) -> None:
            if self.conn is not None:
                try:
                    self.conn.__exit__(exc_type, exc_val, exc_tb)
                finally:
                    self.conn.close()

    def _get_connection(self):
        return self._ConnectionContext(self)

    def close(self) -> None:
        """Close any lingering database resources."""
        pass

    def _init_db(self) -> None:
        """Initialize all evolution tables."""
        with self._lock, self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT '',
                    description TEXT NOT NULL DEFAULT '',
                    target_module TEXT NOT NULL DEFAULT '',
                    target_file TEXT NOT NULL DEFAULT '',
                    weakness_description TEXT NOT NULL DEFAULT '',
                    weakness_evidence TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL DEFAULT 'created',
                    candidate_ids TEXT NOT NULL DEFAULT '[]',
                    selected_candidate_id TEXT,
                    parent_experiment_id TEXT,
                    generation INTEGER NOT NULL DEFAULT 0,
                    config TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    error TEXT,
                    metadata TEXT NOT NULL DEFAULT '{}'
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS candidates (
                    candidate_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    parent_id TEXT,
                    generation INTEGER NOT NULL DEFAULT 0,
                    target_file TEXT NOT NULL DEFAULT '',
                    original_code TEXT NOT NULL DEFAULT '',
                    proposed_code TEXT NOT NULL DEFAULT '',
                    diff_summary TEXT NOT NULL DEFAULT '',
                    rationale TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'pending',
                    benchmark_ids TEXT NOT NULL DEFAULT '[]',
                    correctness_score REAL NOT NULL DEFAULT 0.0,
                    performance_score REAL NOT NULL DEFAULT 0.0,
                    resource_score REAL NOT NULL DEFAULT 0.0,
                    composite_score REAL NOT NULL DEFAULT 0.0,
                    held_out_score REAL NOT NULL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_results (
                    result_id TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    suite_name TEXT NOT NULL DEFAULT 'default',
                    total_tests INTEGER NOT NULL DEFAULT 0,
                    passed_tests INTEGER NOT NULL DEFAULT 0,
                    failed_tests INTEGER NOT NULL DEFAULT 0,
                    pass_rate REAL NOT NULL DEFAULT 0.0,
                    held_out_total INTEGER NOT NULL DEFAULT 0,
                    held_out_passed INTEGER NOT NULL DEFAULT 0,
                    held_out_pass_rate REAL NOT NULL DEFAULT 0.0,
                    avg_execution_time_ms REAL NOT NULL DEFAULT 0.0,
                    max_execution_time_ms REAL NOT NULL DEFAULT 0.0,
                    p95_execution_time_ms REAL NOT NULL DEFAULT 0.0,
                    peak_memory_mb REAL NOT NULL DEFAULT 0.0,
                    avg_memory_mb REAL NOT NULL DEFAULT 0.0,
                    baseline_pass_rate REAL NOT NULL DEFAULT 0.0,
                    improvement_delta REAL NOT NULL DEFAULT 0.0,
                    is_improvement INTEGER NOT NULL DEFAULT 0,
                    is_overfitting INTEGER NOT NULL DEFAULT 0,
                    failed_test_names TEXT NOT NULL DEFAULT '[]',
                    error_messages TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS evolution_history (
                    record_id TEXT PRIMARY KEY,
                    parent_id TEXT,
                    child_id TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    parent_version TEXT NOT NULL DEFAULT '',
                    child_version TEXT NOT NULL DEFAULT '',
                    target_module TEXT NOT NULL DEFAULT '',
                    target_file TEXT NOT NULL DEFAULT '',
                    improvement_type TEXT NOT NULL DEFAULT '',
                    improvement_summary TEXT NOT NULL DEFAULT '',
                    correctness_delta REAL NOT NULL DEFAULT 0.0,
                    performance_delta REAL NOT NULL DEFAULT 0.0,
                    generation INTEGER NOT NULL DEFAULT 0,
                    was_accepted INTEGER NOT NULL DEFAULT 0,
                    accepted_by TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS approvals (
                    request_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    request_type TEXT NOT NULL DEFAULT 'code_merge',
                    title TEXT NOT NULL DEFAULT '',
                    description TEXT NOT NULL DEFAULT '',
                    diff_summary TEXT NOT NULL DEFAULT '',
                    benchmark_summary TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'pending',
                    requested_at TEXT NOT NULL,
                    decided_at TEXT,
                    decided_by TEXT NOT NULL DEFAULT '',
                    rejection_reason TEXT NOT NULL DEFAULT '',
                    rollback_info TEXT NOT NULL DEFAULT '{}',
                    metadata TEXT NOT NULL DEFAULT '{}'
                );
            """)

            # Indexes for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_candidates_experiment ON candidates(experiment_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_candidates_parent ON candidates(parent_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_benchmarks_candidate ON benchmark_results(candidate_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_history_parent ON evolution_history(parent_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_history_child ON evolution_history(child_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_approvals_status ON approvals(status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_approvals_experiment ON approvals(experiment_id);")

    # -----------------------------------------------------------------------
    # Experiments CRUD
    # -----------------------------------------------------------------------

    def save_experiment(self, exp: Experiment) -> None:
        """Insert or replace an experiment record."""
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiments (
                    experiment_id, title, description, target_module, target_file,
                    weakness_description, weakness_evidence, status, candidate_ids,
                    selected_candidate_id, parent_experiment_id, generation,
                    config, created_at, updated_at, completed_at, error, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exp.experiment_id, exp.title, exp.description,
                    exp.target_module, exp.target_file,
                    exp.weakness_description, json.dumps(exp.weakness_evidence),
                    exp.status.value if isinstance(exp.status, ExperimentStatus) else str(exp.status),
                    json.dumps(exp.candidate_ids),
                    exp.selected_candidate_id, exp.parent_experiment_id, exp.generation,
                    json.dumps(exp.config), exp.created_at, exp.updated_at,
                    exp.completed_at, exp.error, json.dumps(exp.metadata),
                ),
            )

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Retrieve an experiment by ID."""
        with self._lock, self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
            if not row:
                return None
            return self._row_to_experiment(row)

    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Experiment]:
        """List experiments, optionally filtered by status."""
        with self._lock, self._get_connection() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM experiments WHERE status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (status.value, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
            return [self._row_to_experiment(r) for r in rows]

    def _row_to_experiment(self, row: sqlite3.Row) -> Experiment:
        """Convert a database row to an Experiment object."""
        return Experiment(
            experiment_id=row["experiment_id"],
            title=row["title"],
            description=row["description"],
            target_module=row["target_module"],
            target_file=row["target_file"],
            weakness_description=row["weakness_description"],
            weakness_evidence=json.loads(row["weakness_evidence"]),
            status=ExperimentStatus(row["status"]) if row["status"] in [e.value for e in ExperimentStatus] else ExperimentStatus.CREATED,
            candidate_ids=json.loads(row["candidate_ids"]),
            selected_candidate_id=row["selected_candidate_id"],
            parent_experiment_id=row["parent_experiment_id"],
            generation=row["generation"],
            config=json.loads(row["config"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            completed_at=row["completed_at"],
            error=row["error"],
            metadata=json.loads(row["metadata"]),
        )

    # -----------------------------------------------------------------------
    # Candidates CRUD
    # -----------------------------------------------------------------------

    def save_candidate(self, cand: Candidate) -> None:
        """Insert or replace a candidate record."""
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO candidates (
                    candidate_id, experiment_id, parent_id, generation, target_file,
                    original_code, proposed_code, diff_summary, rationale, status,
                    benchmark_ids, correctness_score, performance_score,
                    resource_score, composite_score, held_out_score,
                    created_at, updated_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cand.candidate_id, cand.experiment_id, cand.parent_id,
                    cand.generation, cand.target_file,
                    cand.original_code, cand.proposed_code,
                    cand.diff_summary, cand.rationale,
                    cand.status.value if isinstance(cand.status, CandidateStatus) else str(cand.status),
                    json.dumps(cand.benchmark_ids),
                    cand.correctness_score, cand.performance_score,
                    cand.resource_score, cand.composite_score, cand.held_out_score,
                    cand.created_at, cand.updated_at, json.dumps(cand.metadata),
                ),
            )

    def get_candidate(self, candidate_id: str) -> Optional[Candidate]:
        """Retrieve a candidate by ID."""
        with self._lock, self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM candidates WHERE candidate_id = ?", (candidate_id,)
            ).fetchone()
            if not row:
                return None
            return self._row_to_candidate(row)

    def list_candidates(
        self,
        experiment_id: Optional[str] = None,
        status: Optional[CandidateStatus] = None,
        limit: int = 50,
    ) -> List[Candidate]:
        """List candidates, optionally filtered by experiment and/or status."""
        with self._lock, self._get_connection() as conn:
            conditions: List[str] = []
            params: List[Any] = []
            if experiment_id:
                conditions.append("experiment_id = ?")
                params.append(experiment_id)
            if status:
                conditions.append("status = ?")
                params.append(status.value)
            where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            params.append(limit)
            rows = conn.execute(
                f"SELECT * FROM candidates {where} ORDER BY composite_score DESC LIMIT ?",
                params,
            ).fetchall()
            return [self._row_to_candidate(r) for r in rows]

    def get_candidates_for_experiment(self, experiment_id: str) -> List[Candidate]:
        """List all candidates for an experiment."""
        return self.list_candidates(experiment_id=experiment_id)

    def get_candidate_lineage(self, candidate_id: str) -> List[Candidate]:
        """Walk up the parent chain to reconstruct full version lineage."""
        lineage: List[Candidate] = []
        current_id: Optional[str] = candidate_id
        visited: set[str] = set()
        while current_id and current_id not in visited:
            visited.add(current_id)
            cand = self.get_candidate(current_id)
            if not cand:
                break
            lineage.append(cand)
            current_id = cand.parent_id
        lineage.reverse()
        return lineage

    def _row_to_candidate(self, row: sqlite3.Row) -> Candidate:
        """Convert a database row to a Candidate object."""
        return Candidate(
            candidate_id=row["candidate_id"],
            experiment_id=row["experiment_id"],
            parent_id=row["parent_id"],
            generation=row["generation"],
            target_file=row["target_file"],
            original_code=row["original_code"],
            proposed_code=row["proposed_code"],
            diff_summary=row["diff_summary"],
            rationale=row["rationale"],
            status=CandidateStatus(row["status"]) if row["status"] in [s.value for s in CandidateStatus] else CandidateStatus.PENDING,
            benchmark_ids=json.loads(row["benchmark_ids"]),
            correctness_score=row["correctness_score"],
            performance_score=row["performance_score"],
            resource_score=row["resource_score"],
            composite_score=row["composite_score"],
            held_out_score=row["held_out_score"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=json.loads(row["metadata"]),
        )

    # -----------------------------------------------------------------------
    # Benchmark Results CRUD
    # -----------------------------------------------------------------------

    def save_benchmark_result(self, result: BenchmarkResult) -> None:
        """Insert or replace a benchmark result."""
        # Preserve optional metrics without changing existing database schemas.
        result.metadata["baseline_held_out_pass_rate"] = result.baseline_held_out_pass_rate
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO benchmark_results (
                    result_id, candidate_id, experiment_id, suite_name,
                    total_tests, passed_tests, failed_tests, pass_rate,
                    held_out_total, held_out_passed, held_out_pass_rate,
                    avg_execution_time_ms, max_execution_time_ms, p95_execution_time_ms,
                    peak_memory_mb, avg_memory_mb,
                    baseline_pass_rate, improvement_delta, is_improvement, is_overfitting,
                    failed_test_names, error_messages, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.result_id, result.candidate_id, result.experiment_id,
                    result.suite_name,
                    result.total_tests, result.passed_tests, result.failed_tests,
                    result.pass_rate,
                    result.held_out_total, result.held_out_passed, result.held_out_pass_rate,
                    result.avg_execution_time_ms, result.max_execution_time_ms,
                    result.p95_execution_time_ms,
                    result.peak_memory_mb, result.avg_memory_mb,
                    result.baseline_pass_rate, result.improvement_delta,
                    int(result.is_improvement), int(result.is_overfitting),
                    json.dumps(result.failed_test_names), json.dumps(result.error_messages),
                    result.created_at, json.dumps(result.metadata),
                ),
            )

    def get_benchmark_result(self, result_id: str) -> Optional[BenchmarkResult]:
        """Retrieve by result ID, or the latest result for a candidate ID."""
        with self._lock, self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM benchmark_results WHERE result_id = ?", (result_id,)
            ).fetchone()
            if not row:
                row = conn.execute(
                    "SELECT * FROM benchmark_results WHERE candidate_id = ? ORDER BY created_at DESC LIMIT 1",
                    (result_id,),
                ).fetchone()
            if not row:
                return None
            return self._row_to_benchmark(row)

    def list_benchmark_results(
        self,
        candidate_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[BenchmarkResult]:
        """List benchmark results with optional filters."""
        with self._lock, self._get_connection() as conn:
            conditions: List[str] = []
            params: List[Any] = []
            if candidate_id:
                conditions.append("candidate_id = ?")
                params.append(candidate_id)
            if experiment_id:
                conditions.append("experiment_id = ?")
                params.append(experiment_id)
            where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            params.append(limit)
            rows = conn.execute(
                f"SELECT * FROM benchmark_results {where} ORDER BY created_at DESC LIMIT ?",
                params,
            ).fetchall()
            return [self._row_to_benchmark(r) for r in rows]

    def get_benchmark_results_for_experiment(self, experiment_id: str) -> List[BenchmarkResult]:
        """List all benchmark results for an experiment."""
        return self.list_benchmark_results(experiment_id=experiment_id)

    def _row_to_benchmark(self, row: sqlite3.Row) -> BenchmarkResult:
        return BenchmarkResult(
            result_id=row["result_id"],
            candidate_id=row["candidate_id"],
            experiment_id=row["experiment_id"],
            suite_name=row["suite_name"],
            total_tests=row["total_tests"],
            passed_tests=row["passed_tests"],
            failed_tests=row["failed_tests"],
            pass_rate=row["pass_rate"],
            held_out_total=row["held_out_total"],
            held_out_passed=row["held_out_passed"],
            held_out_pass_rate=row["held_out_pass_rate"],
            avg_execution_time_ms=row["avg_execution_time_ms"],
            max_execution_time_ms=row["max_execution_time_ms"],
            p95_execution_time_ms=row["p95_execution_time_ms"],
            peak_memory_mb=row["peak_memory_mb"],
            avg_memory_mb=row["avg_memory_mb"],
            baseline_pass_rate=row["baseline_pass_rate"],
            baseline_held_out_pass_rate=json.loads(row["metadata"]).get("baseline_held_out_pass_rate"),
            improvement_delta=row["improvement_delta"],
            is_improvement=bool(row["is_improvement"]),
            is_overfitting=bool(row["is_overfitting"]),
            failed_test_names=json.loads(row["failed_test_names"]),
            error_messages=json.loads(row["error_messages"]),
            created_at=row["created_at"],
            metadata=json.loads(row["metadata"]),
        )

    # -----------------------------------------------------------------------
    # Evolution History CRUD
    # -----------------------------------------------------------------------

    def save_evolution_record(self, record: EvolutionRecord) -> None:
        """Insert or replace an evolution history record."""
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO evolution_history (
                    record_id, parent_id, child_id, experiment_id,
                    parent_version, child_version, target_module, target_file,
                    improvement_type, improvement_summary,
                    correctness_delta, performance_delta, generation,
                    was_accepted, accepted_by, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.record_id, record.parent_id, record.child_id,
                    record.experiment_id,
                    record.parent_version, record.child_version,
                    record.target_module, record.target_file,
                    record.improvement_type, record.improvement_summary,
                    record.correctness_delta, record.performance_delta,
                    record.generation,
                    int(record.was_accepted), record.accepted_by,
                    record.created_at, json.dumps(record.metadata),
                ),
            )

    def get_evolution_history(
        self,
        target_module: Optional[str] = None,
        limit: int = 100,
    ) -> List[EvolutionRecord]:
        """List evolution history, optionally filtered by target module."""
        with self._lock, self._get_connection() as conn:
            if target_module:
                rows = conn.execute(
                    "SELECT * FROM evolution_history WHERE target_module = ? ORDER BY created_at DESC LIMIT ?",
                    (target_module, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM evolution_history ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [self._row_to_evolution_record(r) for r in rows]

    def get_ancestry_chain(self, child_id: str) -> List[EvolutionRecord]:
        """Walk up the parent chain in evolution history."""
        chain: List[EvolutionRecord] = []
        visited: set[str] = set()
        current = child_id
        while current and current not in visited:
            visited.add(current)
            with self._lock, self._get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM evolution_history WHERE child_id = ?", (current,)
                ).fetchone()
            if not row:
                break
            record = self._row_to_evolution_record(row)
            chain.append(record)
            current = record.parent_id or ""
        chain.reverse()
        return chain

    def _row_to_evolution_record(self, row: sqlite3.Row) -> EvolutionRecord:
        return EvolutionRecord(
            record_id=row["record_id"],
            parent_id=row["parent_id"],
            child_id=row["child_id"],
            experiment_id=row["experiment_id"],
            parent_version=row["parent_version"],
            child_version=row["child_version"],
            target_module=row["target_module"],
            target_file=row["target_file"],
            improvement_type=row["improvement_type"],
            improvement_summary=row["improvement_summary"],
            correctness_delta=row["correctness_delta"],
            performance_delta=row["performance_delta"],
            generation=row["generation"],
            was_accepted=bool(row["was_accepted"]),
            accepted_by=row["accepted_by"],
            created_at=row["created_at"],
            metadata=json.loads(row["metadata"]),
        )

    # -----------------------------------------------------------------------
    # Approvals CRUD
    # -----------------------------------------------------------------------

    def save_approval(self, approval: ApprovalRequest) -> None:
        """Insert or replace an approval request."""
        meta = dict(approval.metadata)
        if approval.target_file:
            meta["target_file"] = approval.target_file
        if approval.diff:
            meta["diff"] = approval.diff

        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO approvals (
                    request_id, experiment_id, candidate_id, request_type,
                    title, description, diff_summary, benchmark_summary,
                    status, requested_at, decided_at, decided_by,
                    rejection_reason, rollback_info, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    approval.request_id, approval.experiment_id,
                    approval.candidate_id, approval.request_type,
                    approval.title, approval.description,
                    approval.diff_summary or approval.diff, json.dumps(approval.benchmark_summary),
                    approval.status.value if isinstance(approval.status, ApprovalStatus) else str(approval.status),
                    approval.requested_at, approval.decided_at,
                    approval.decided_by, approval.rejection_reason,
                    json.dumps(approval.rollback_info), json.dumps(meta),
                ),
            )

    def get_approval(self, request_id: str) -> Optional[ApprovalRequest]:
        """Retrieve an approval request by ID."""
        with self._lock, self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM approvals WHERE request_id = ?", (request_id,)
            ).fetchone()
            if not row:
                return None
            return self._row_to_approval(row)

    def list_approvals(
        self,
        status: Optional[ApprovalStatus] = None,
        experiment_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[ApprovalRequest]:
        """List approval requests with optional filters."""
        with self._lock, self._get_connection() as conn:
            conditions: List[str] = []
            params: List[Any] = []
            if status:
                conditions.append("status = ?")
                params.append(status.value)
            if experiment_id:
                conditions.append("experiment_id = ?")
                params.append(experiment_id)
            where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            params.append(limit)
            rows = conn.execute(
                f"SELECT * FROM approvals {where} ORDER BY requested_at DESC LIMIT ?",
                params,
            ).fetchall()
            return [self._row_to_approval(r) for r in rows]

    def get_pending_approvals(self, limit: int = 50) -> List[ApprovalRequest]:
        """List pending approval requests."""
        return self.list_approvals(status=ApprovalStatus.PENDING, limit=limit)

    def count_pending_approvals(self) -> int:
        """Return count of pending approval requests."""
        with self._lock, self._get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM approvals WHERE status = ?",
                (ApprovalStatus.PENDING.value,),
            ).fetchone()
            return row["cnt"] if row else 0

    def _row_to_approval(self, row: sqlite3.Row) -> ApprovalRequest:
        meta = json.loads(row["metadata"]) if row["metadata"] else {}
        return ApprovalRequest(
            request_id=row["request_id"],
            experiment_id=row["experiment_id"],
            candidate_id=row["candidate_id"],
            target_file=meta.get("target_file", ""),
            diff=meta.get("diff", row["diff_summary"]),
            request_type=row["request_type"],
            title=row["title"],
            description=row["description"],
            diff_summary=row["diff_summary"],
            benchmark_summary=json.loads(row["benchmark_summary"]) if row["benchmark_summary"] else {},
            status=ApprovalStatus(row["status"]) if row["status"] in [s.value for s in ApprovalStatus] else ApprovalStatus.PENDING,
            requested_at=row["requested_at"],
            decided_at=row["decided_at"],
            decided_by=row["decided_by"],
            rejection_reason=row["rejection_reason"],
            rollback_info=json.loads(row["rollback_info"]) if row["rollback_info"] else {},
            metadata=meta,
        )

    # -----------------------------------------------------------------------
    # Aggregate Queries
    # -----------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return aggregate statistics across all evolution data."""
        with self._lock, self._get_connection() as conn:
            exp_count = conn.execute("SELECT COUNT(*) as cnt FROM experiments").fetchone()["cnt"]
            cand_count = conn.execute("SELECT COUNT(*) as cnt FROM candidates").fetchone()["cnt"]
            bench_count = conn.execute("SELECT COUNT(*) as cnt FROM benchmark_results").fetchone()["cnt"]
            hist_count = conn.execute("SELECT COUNT(*) as cnt FROM evolution_history").fetchone()["cnt"]
            pending_approvals = self.count_pending_approvals()
            merged_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM experiments WHERE status = ?",
                (ExperimentStatus.MERGED.value,),
            ).fetchone()["cnt"]
            rejected_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM experiments WHERE status = ?",
                (ExperimentStatus.REJECTED.value,),
            ).fetchone()["cnt"]

            return {
                "total_experiments": exp_count,
                "total_candidates": cand_count,
                "total_benchmarks": bench_count,
                "total_evolution_records": hist_count,
                "pending_approvals": pending_approvals,
                "merged_experiments": merged_count,
                "rejected_experiments": rejected_count,
            }
