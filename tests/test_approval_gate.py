"""
tests/test_approval_gate.py
===========================
Tests for human-in-the-loop ApprovalGate, backup creation, merge, and rollback.
"""

import tempfile
from pathlib import Path
import pytest

from axen.evolution.approval import ApprovalGate
from axen.evolution.models import Candidate, CandidateStatus, Experiment, ExperimentStatus, BenchmarkResult
from axen.evolution.store import EvolutionStore


def test_approval_gate_workflow():
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir)
        db_path = base_dir / "approval_test.db"
        backup_dir = base_dir / "backups"
        store = EvolutionStore(db_path=db_path)
        gate = ApprovalGate(store=store, backup_dir=str(backup_dir), repo_root=str(base_dir))

        # Target file
        target_file = base_dir / "target_module.py"
        target_file.write_text("def compute(): return 'original'\n", encoding="utf-8")

        # Create experiment and candidate
        exp = Experiment(
            target_file=str(target_file),
            target_module="target_module",
        )
        store.save_experiment(exp)

        cand = Candidate(
            experiment_id=exp.experiment_id,
            target_file=str(target_file),
            code="def compute(): return 'improved'\n",
            original_code="def compute(): return 'original'\n",
            diff="+ improved\n- original",
        )
        store.save_candidate(cand)
        store.save_benchmark_result(BenchmarkResult(candidate_id=cand.candidate_id,
            experiment_id=exp.experiment_id, total_tests=1, passed_tests=1,
            held_out_total=1, held_out_passed=1, is_improvement=True))

        # 1. Create approval request
        req = gate.create_request(experiment=exp, candidate=cand)
        pending = gate.list_pending()
        assert len(pending) == 1
        assert pending[0].request_id == req.request_id

        # 2. Approve and merge
        success = gate.approve(
            request_id=req.request_id,
            reviewer="lead_reviewer",
            notes="LGTM",
            apply_merge=True,
        )
        assert success is True

        # Check target file was updated
        assert "improved" in target_file.read_text(encoding="utf-8")
        assert len(gate.list_pending()) == 0

        # Check backup was created
        backup_file = backup_dir / f"{exp.experiment_id}_{target_file.name}.bak"
        assert backup_file.exists()
        assert "original" in backup_file.read_text(encoding="utf-8")

        # 3. Rollback
        rollback_success = gate.rollback(exp.experiment_id)
        assert rollback_success is True
        assert "original" in target_file.read_text(encoding="utf-8")
