"""
axen/evolution/approval.py
==========================
Human-in-the-loop approval gate for self-evolution changes.

Enforces:
1. No code modification or weight activation without explicit human approval
2. Rollback capability: creates backup copies of original files before merging
3. Audit logging of who approved/rejected, with timestamps and rationales
"""

from __future__ import annotations

import logging
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional

from .models import (
    ApprovalRequest,
    ApprovalStatus,
    Candidate,
    CandidateStatus,
    Experiment,
    ExperimentStatus,
)
from .permissions import AgentRole, EvolutionPermission, EvolutionSecurityPolicy
from .store import EvolutionStore

logger = logging.getLogger("axen.evolution.approval")


class ApprovalGate:
    """
    Manages review and approval lifecycle for candidate improvements.
    """

    def __init__(
        self,
        store: EvolutionStore,
        backup_dir: str = ".backups/evolution",
        repo_root: Optional[str] = None,
    ) -> None:
        self.store = store
        self.repo_root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def create_request(
        self,
        experiment: Experiment,
        candidate: Candidate,
        risk_level: str = "medium",
    ) -> ApprovalRequest:
        """
        Create a new pending approval request for a candidate.
        """
        req = ApprovalRequest(
            experiment_id=experiment.experiment_id,
            candidate_id=candidate.candidate_id,
            target_file=candidate.target_file or experiment.target_file,
            diff=candidate.diff,
            risk_level=risk_level,
        )
        self.store.save_approval(req)
        logger.info(
            "Created approval request %s for experiment %s (candidate: %s)",
            req.request_id,
            experiment.experiment_id,
            candidate.candidate_id,
        )
        return req

    def list_pending(self) -> List[ApprovalRequest]:
        """List all requests awaiting human approval."""
        return self.store.get_pending_approvals()

    def approve(
        self,
        request_id: str,
        reviewer: str = "human_operator",
        notes: str = "",
        apply_merge: bool = True,
    ) -> bool:
        """
        Approve candidate and optionally merge into production file.

        Creates an automatic backup before overwriting.
        """
        req = self.store.get_approval(request_id)
        if req is None:
            raise ValueError(f"Approval request '{request_id}' not found.")

        if req.status != ApprovalStatus.PENDING:
            raise ValueError("Approval decision is already final")
        candidate = self.store.get_candidate(req.candidate_id)
        exp = self.store.get_experiment(req.experiment_id)
        if apply_merge:
            if req.request_type == "weight_activation":
                self._apply_weights(req)
            else:
                if candidate is None: raise ValueError("Candidate missing")
                target = self._target(req.target_file or candidate.target_file or (exp.target_file if exp else ""))
                if not target.is_file() or not candidate.original_code or target.read_text(encoding="utf-8") != candidate.original_code:
                    raise ValueError("Source changed since experiment; re-evaluation required")
                benchmarks = self.store.list_benchmark_results(candidate_id=candidate.candidate_id)
                if not any(b.is_improvement and b.total_tests > 0 and b.held_out_total > 0
                           and not b.is_overfitting for b in benchmarks):
                    raise ValueError("Measured improvement including held-out tests required")

        req.approve(reviewer=reviewer, notes=notes)
        self.store.save_approval(req)

        # Update candidate status
        candidate = self.store.get_candidate(req.candidate_id)
        if candidate is not None:
            candidate.status = CandidateStatus.APPROVED
            self.store.save_candidate(candidate)

        # Update experiment status
        exp = self.store.get_experiment(req.experiment_id)
        if exp is not None:
            exp.transition_to(ExperimentStatus.APPROVED)
            self.store.save_experiment(exp)

        target_f = req.target_file or (candidate.target_file if candidate else "") or (exp.target_file if exp else "")
        if apply_merge and req.request_type == "code_merge" and candidate is not None and target_f:
            req.target_file = target_f
            self._apply_merge(req, candidate)

        return True

    def _target(self, path: str) -> Path:
        if not path: raise ValueError("Empty production target")
        target = Path(path)
        if not target.is_absolute(): target = self.repo_root / target
        target = target.resolve()

        if not target.is_relative_to(self.repo_root):
            raise PermissionError("Target outside repository")
        relative = target.relative_to(self.repo_root).as_posix().lower()
        if any(part.startswith(".") for part in Path(relative).parts) or target.suffix.lower() not in (".py", ".pt"):
            raise PermissionError("Only visible Python source or reviewed checkpoint targets are eligible")
        EvolutionSecurityPolicy().validate_code_target(relative)
        protected = (".git", ".env", "configs", "axen/evolution", "app/security", "tests", "scripts", "checkpoints/foundation", "axen/context", "axen/model/trainer.py", "axen/model/dataset.py")
        if any(relative == p or relative.startswith(p + "/") for p in protected):
            raise PermissionError("Policy, tests, credentials and autonomous infrastructure cannot evolve")
        return target

    def _apply_weights(self, req: ApprovalRequest) -> None:
        config = json.loads((self.repo_root / "configs/autonomy.json").read_text(encoding="utf-8"))
        relative_target = config.get("neural_training", {}).get("activation_target")
        if not relative_target: raise ValueError("Operator must configure a candidate activation target first")
        target = self._target(relative_target)
        if target.suffix != ".pt": raise ValueError("Checkpoint activation target must end in .pt")
        staged = Path(req.metadata.get("checkpoint_path", "")).resolve()
        if not staged.is_relative_to(self.repo_root / "checkpoints/evolution_staging"):
            raise PermissionError("Checkpoint outside trusted staging directory")
        if not staged.is_file() or hashlib.sha256(staged.read_bytes()).hexdigest() != req.metadata.get("checkpoint_sha256"):
            raise ValueError("Staged checkpoint changed after evaluation")
        if target.exists():
            shutil.copy2(target, self.backup_dir / f"{req.experiment_id}_{target.name}.bak")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(staged, target)
        req.target_file = str(target)
        req.metadata["deployed_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        exp = self.store.get_experiment(req.experiment_id)
        if exp:
            exp.target_file = str(target); self.store.save_experiment(exp)

    def reject(
        self,
        request_id: str,
        reviewer: str = "human_operator",
        reason: str = "",
    ) -> bool:
        """Reject candidate modification."""
        req = self.store.get_approval(request_id)
        if req is None:
            raise ValueError(f"Approval request '{request_id}' not found.")

        if req.status != ApprovalStatus.PENDING:
            raise ValueError("Approval decision is already final")

        req.reject(reviewer=reviewer, reason=reason)
        self.store.save_approval(req)

        candidate = self.store.get_candidate(req.candidate_id)
        if candidate is not None:
            candidate.status = CandidateStatus.REJECTED
            self.store.save_candidate(candidate)

        exp = self.store.get_experiment(req.experiment_id)
        if exp is not None:
            exp.transition_to(ExperimentStatus.REJECTED)
            self.store.save_experiment(exp)

        return True

    def _apply_merge(self, req: ApprovalRequest, candidate: Candidate) -> None:
        """Back up target file and write approved candidate code."""
        target_f = req.target_file or candidate.target_file
        target_path = self._target(target_f)
        if not target_path.exists():
            logger.warning("Target file %s does not exist, writing new file.", target_path)

        # Create backup
        if target_path.exists():
            backup_file = self.backup_dir / f"{req.experiment_id}_{target_path.name}.bak"
            shutil.copy2(target_path, backup_file)
            logger.info("Created backup at %s", backup_file)

        # Write approved code
        code_to_write = candidate.code or candidate.proposed_code
        target_path.write_text(code_to_write, encoding="utf-8")
        req.metadata["deployed_sha256"] = hashlib.sha256(target_path.read_bytes()).hexdigest()
        self.store.save_approval(req)
        candidate.status = CandidateStatus.MERGED
        self.store.save_candidate(candidate)

        exp = self.store.get_experiment(req.experiment_id)
        if exp is not None:
            exp.transition_to(ExperimentStatus.MERGED)
            self.store.save_experiment(exp)

        logger.info("Successfully merged candidate %s to %s", candidate.candidate_id, target_path)

    def rollback(self, experiment_id: str) -> bool:
        """
        Restore original file from backup for a merged experiment.
        """
        exp = self.store.get_experiment(experiment_id)
        if exp is None:
            return False

        target_path = self._target(exp.target_file)
        backup_file = self.backup_dir / f"{experiment_id}_{target_path.name}.bak"

        if not backup_file.exists():
            logger.error("No backup found for experiment %s at %s", experiment_id, backup_file)
            return False

        approvals = self.store.list_approvals(experiment_id=experiment_id)
        approved = next((a for a in approvals if a.status == ApprovalStatus.APPROVED), None)
        if not approved or not target_path.exists() or hashlib.sha256(target_path.read_bytes()).hexdigest() != approved.metadata.get("deployed_sha256"):
            raise ValueError("Current file differs from deployed candidate; refusing to overwrite later edits")

        shutil.copy2(backup_file, target_path)
        exp.transition_to(ExperimentStatus.CANCELLED)
        exp.metadata["rolled_back_at"] = time.time()
        self.store.save_experiment(exp)
        approved.rollback(); self.store.save_approval(approved)
        logger.info("Successfully rolled back %s from backup.", target_path)
        return True
