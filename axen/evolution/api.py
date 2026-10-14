"""
axen/evolution/api.py
=====================
FastAPI router for the AXEN Self-Evolution Engine.

Provides REST endpoints for:
- Viewing experiments and candidate proposals
- Inspecting benchmark metrics and anti-overfitting evaluations
- Human approval / rejection workflows
- Rollback of merged experiments
- Initiating new evolution runs
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

from .approval import ApprovalGate
from .controller import EvolutionController
from .models import ApprovalStatus, CandidateStatus, ExperimentStatus
from .store import EvolutionStore
from .weakness_detector import Weakness
from .learning_api import operator_control

logger = logging.getLogger("axen.evolution.api")

router = APIRouter(prefix="/api/evolution", tags=["Evolution Engine"])

# Global store and gate singleton
_store: Optional[EvolutionStore] = None
_gate: Optional[ApprovalGate] = None


def get_store() -> EvolutionStore:
    global _store
    if _store is None:
        _store = EvolutionStore()
    return _store


def get_gate() -> ApprovalGate:
    global _gate
    if _gate is None:
        _gate = ApprovalGate(store=get_store())
    return _gate


# Pydantic request models
class StartExperimentRequest(BaseModel):
    target_module: str = Field(..., min_length=1)
    target_file: str = Field(..., min_length=1)
    weakness_description: str = Field(..., min_length=1)
    weakness_evidence: List[str] = Field(default_factory=list)
    num_candidates: int = Field(default=3, ge=1, le=10)


class ApprovalDecisionRequest(BaseModel):
    reviewer: str = Field(default="human_operator")
    notes: Optional[str] = Field(default="")
    apply_merge: bool = Field(default=True)


class RejectionRequest(BaseModel):
    reviewer: str = Field(default="human_operator")
    reason: str = Field(..., min_length=1)


@router.get("/status")
def get_evolution_status() -> Dict[str, Any]:
    """Summary metrics of evolution engine state."""
    store = get_store()
    experiments = store.list_experiments(limit=100)
    pending = store.get_pending_approvals()

    completed = sum(1 for e in experiments if e.status == ExperimentStatus.MERGED)
    failed = sum(1 for e in experiments if e.status == ExperimentStatus.FAILED)

    return {
        "status": "online",
        "total_experiments": len(experiments),
        "pending_approvals": len(pending),
        "successful_improvements": completed,
        "failed_experiments": failed,
    }


@router.get("/experiments")
def list_experiments(
    limit: int = Query(default=20, ge=1, le=100),
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """List recent evolution experiments."""
    store = get_store()
    status_filter = None
    if status:
        try:
            status_filter = ExperimentStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    experiments = store.list_experiments(status=status_filter, limit=limit)
    return {
        "count": len(experiments),
        "experiments": [e.to_dict() for e in experiments],
    }


@router.get("/experiments/{experiment_id}")
def get_experiment_detail(experiment_id: str) -> Dict[str, Any]:
    """Get detailed experiment record including all generated candidates."""
    store = get_store()
    exp = store.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    candidates = store.get_candidates_for_experiment(experiment_id)
    benchmarks = store.get_benchmark_results_for_experiment(experiment_id)
    bench_map = {b.candidate_id: b.to_dict() for b in benchmarks}

    cand_dicts = []
    for c in candidates:
        cd = c.to_dict()
        cd["benchmark_result"] = bench_map.get(c.candidate_id)
        cand_dicts.append(cd)

    return {
        "experiment": exp.to_dict(),
        "candidates": cand_dicts,
    }


@router.get("/candidates/{candidate_id}")
def get_candidate(candidate_id: str) -> Dict[str, Any]:
    """Get single candidate code, diff, and benchmark result."""
    store = get_store()
    candidate = store.get_candidate(candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found")

    bench = store.get_benchmark_result(candidate_id)
    result = candidate.to_dict()
    result["benchmark_result"] = bench.to_dict() if bench else None
    return result


@router.get("/benchmarks")
def list_benchmarks(limit: int = Query(default=30, ge=1, le=100)) -> Dict[str, Any]:
    """List recent benchmark evaluation records."""
    store = get_store()
    # List experiments and collect benchmarks
    experiments = store.list_experiments(limit=limit)
    all_benchmarks: List[Dict[str, Any]] = []
    for exp in experiments:
        for b in store.get_benchmark_results_for_experiment(exp.experiment_id):
            all_benchmarks.append(b.to_dict())

    return {
        "count": len(all_benchmarks),
        "benchmarks": all_benchmarks[:limit],
    }


@router.get("/approvals")
def list_pending_approvals() -> Dict[str, Any]:
    """List all candidate improvements awaiting human review."""
    gate = get_gate()
    pending = gate.list_pending()
    return {
        "count": len(pending),
        "approvals": [p.to_dict() for p in pending],
    }


@router.post("/approve/{request_id}", dependencies=[Depends(operator_control)])
def approve_request(request_id: str, payload: ApprovalDecisionRequest) -> Dict[str, Any]:
    """Human approve and optionally merge candidate."""
    gate = get_gate()
    try:
        gate.approve(
            request_id=request_id,
            reviewer=payload.reviewer,
            notes=payload.notes or "",
            apply_merge=payload.apply_merge,
        )
        return {"status": "success", "message": f"Approved request {request_id}"}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Approval failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/reject/{request_id}", dependencies=[Depends(operator_control)])
def reject_request(request_id: str, payload: RejectionRequest) -> Dict[str, Any]:
    """Human reject candidate proposal."""
    gate = get_gate()
    try:
        gate.reject(
            request_id=request_id,
            reviewer=payload.reviewer,
            reason=payload.reason,
        )
        return {"status": "success", "message": f"Rejected request {request_id}"}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Rejection failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/rollback/{experiment_id}", dependencies=[Depends(operator_control)])
def rollback_experiment(experiment_id: str) -> Dict[str, Any]:
    """Roll back production file to pre-evolution backup."""
    gate = get_gate()
    success = gate.rollback(experiment_id)
    if not success:
        raise HTTPException(status_code=400, detail="Rollback failed: No backup found or invalid experiment.")
    return {"status": "success", "message": f"Experiment {experiment_id} rolled back successfully."}


@router.post("/start-experiment", dependencies=[Depends(operator_control)])
def start_experiment(req: StartExperimentRequest) -> Dict[str, Any]:
    """
    Initiate an evolution experiment targeting a weakness.
    """
    store = get_store()
    from pathlib import Path
    try:
        target_path = get_gate()._target(req.target_file)
    except (ValueError, PermissionError) as exc:
        raise HTTPException(403, str(exc)) from exc
    if not target_path.exists():
        raise HTTPException(status_code=400, detail=f"Target file does not exist: {req.target_file}")

    original_code = target_path.read_text(encoding="utf-8")
    from .learning_worker import get_worker
    targets = get_worker().config.get("code_targets", [])
    if not any(t.get("contract") == "inclusive_sum_nonnegative"
               and get_gate()._target(t["path"]) == target_path for t in targets):
        raise HTTPException(403, "Configure an approved code target and fixed evaluation contract first")

    from .candidate_generator import CandidateGenerator
    from app.ui.app import model_manager
    generator = CandidateGenerator(model=model_manager.get_model('router'))
    controller = EvolutionController(store=store, generator=generator)

    weakness = Weakness(
        target_module=req.target_module,
        target_file=req.target_file,
        description=req.weakness_description,
        evidence=req.weakness_evidence,
    )

    from .benchmark_suite import BenchmarkSuite
    from .evaluator import CandidateEvaluator
    from .practice import TRAIN_CASES, HELD_OUT_CASES
    suite = BenchmarkSuite(name="inclusive-sum-v1", split_ratio=0.6)
    for index, (arg, expected) in enumerate(TRAIN_CASES + HELD_OUT_CASES):
        suite.add_test(str(index), arg, expected)
    evaluator = CandidateEvaluator()
    experiments = controller.run_observation_loop(lambda: [weakness], lambda path: original_code,
        lambda candidate: evaluator.evaluate_candidate(candidate, suite, function_name="inclusive_sum"),
        max_cycles=1, max_experiments=1, num_candidates=req.num_candidates)
    experiment = experiments[0]

    return {
        "status": "created",
        "experiment": experiment.to_dict(),
    }
