"""Trust boundaries for approvals, checkpoint activation and evolution linkage."""
import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from axen.evolution.approval import ApprovalGate
from axen.evolution.models import ApprovalRequest, Candidate, BenchmarkResult, Experiment
from axen.evolution.store import EvolutionStore
from tests.test_autonomous_learning import worker_at, finish


def make_candidate(tmp_path):
    store = EvolutionStore(tmp_path / "evolution.db")
    gate = ApprovalGate(store, backup_dir=str(tmp_path / "backups"), repo_root=str(tmp_path))
    target = tmp_path / "helper.py"; target.write_text("def helper(n):\n return n\n", encoding="utf-8")
    exp = Experiment(target_file=str(target)); store.save_experiment(exp)
    candidate = Candidate(experiment_id=exp.experiment_id, target_file=str(target),
        original_code=target.read_text(encoding="utf-8"), code="def helper(n):\n return n+1\n")
    store.save_candidate(candidate)
    request = gate.create_request(exp, candidate)
    return store, gate, target, exp, candidate, request


def test_requires_measured_evidence_and_unchanged_source(tmp_path):
    store, gate, target, exp, candidate, request = make_candidate(tmp_path)
    with pytest.raises(ValueError, match="Measured improvement"): gate.approve(request.request_id)
    assert "return n+1" not in target.read_text(encoding="utf-8")
    store.save_benchmark_result(BenchmarkResult(candidate_id=candidate.candidate_id, total_tests=2,
        passed_tests=2, held_out_total=1, held_out_passed=1, is_improvement=True))
    target.write_text("# Later edit\n" + candidate.original_code, encoding="utf-8")
    with pytest.raises(ValueError, match="Source changed"): gate.approve(request.request_id)
    assert "Later edit" in target.read_text(encoding="utf-8")


def test_repository_and_policy_containment(tmp_path):
    store, gate, *_ = make_candidate(tmp_path)
    for path in (tmp_path.parent / "outside.py", tmp_path / "configs/autonomy.json",
                 tmp_path / "axen/evolution/approval.py", tmp_path / "tests/test_policy.py",
                 tmp_path / "AXEN/EVOLUTION/sandbox.py", tmp_path / "axen/context/context_policy.py"):
        with pytest.raises(PermissionError): gate._target(str(path))


def test_weight_activation_checks_staging_hash_and_human_decision(tmp_path):
    store = EvolutionStore(tmp_path / "evolution.db")
    gate = ApprovalGate(store, repo_root=str(tmp_path), backup_dir=str(tmp_path / "backups"))
    staged = tmp_path / "checkpoints/evolution_staging/candidate.pt"
    staged.parent.mkdir(parents=True); staged.write_bytes(b"candidate fixture")
    exp = Experiment(); store.save_experiment(exp)
    request = ApprovalRequest(experiment_id=exp.experiment_id, request_type="weight_activation",
        metadata={"checkpoint_path": str(staged), "checkpoint_sha256": hashlib.sha256(staged.read_bytes()).hexdigest()})
    store.save_approval(request)
    config = tmp_path / "configs/autonomy.json"; config.parent.mkdir()
    config.write_text(json.dumps({"neural_training": {"activation_target": "checkpoints/reviewed/approved.pt"}}), encoding="utf-8")
    staged.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed after evaluation"): gate.approve(request.request_id)
    assert not (tmp_path / "checkpoints/reviewed/approved.pt").exists()
    staged.write_bytes(b"candidate fixture")
    assert gate.approve(request.request_id, reviewer="operator")
    assert (tmp_path / "checkpoints/reviewed/approved.pt").read_bytes() == b"candidate fixture"
    with pytest.raises(ValueError, match="already final"): gate.approve(request.request_id)


def test_sse_uses_actual_persisted_event_and_cursor(tmp_path, monkeypatch):
    import axen.evolution.learning_api as api
    worker = worker_at(tmp_path)
    worker.store.event("source", {"url": "https://docs.python.org/"})
    monkeypatch.setattr(api, "get_worker", lambda: worker)
    class Request:
        headers = {}
        async def is_disconnected(self): return False
    async def first_event():
        response = await api.stream(Request())
        item = await anext(response.body_iterator)
        await response.body_iterator.aclose()
        return item
    payload = asyncio.run(first_event())
    assert "id: 1" in payload and '"kind": "source"' in payload
    assert "docs.python.org" in payload


def test_production_evolution_candidate_is_measured_and_awaits_approval(tmp_path, monkeypatch):
    import axen.evolution.learning_worker as runtime
    from axen.evolution.practice import DEBUGGING_SOURCE
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    docs = tmp_path / "knowledge/learning"; docs.mkdir(parents=True)
    (docs / "python_range_offline.md").write_text("Python range stop is exclusive", encoding="utf-8")
    target = tmp_path / "inclusive_sum.py"; target.write_text(DEBUGGING_SOURCE, encoding="utf-8")
    # Tests inject configuration; production obtains it only from operator file.
    from axen.evolution.learning_worker import load_config
    config = json.loads((Path(__file__).resolve().parents[1] / "configs/autonomy.json").read_text(encoding="utf-8"))
    config["code_targets"] = [{"path": "inclusive_sum.py", "contract": "inclusive_sum_nonnegative"}]
    (tmp_path / "configs").mkdir(); (tmp_path / "configs/autonomy.json").write_text(json.dumps(config), encoding="utf-8")
    worker = worker_at(tmp_path)
    worker.command("start", "DEVELOPMENT", offline_demo=True); finish(worker)
    store = EvolutionStore(tmp_path / "data/axen_evolution.db")
    approvals = store.get_pending_approvals()
    assert len(approvals) == 1
    results = store.list_benchmark_results()
    assert any(result.is_improvement and result.held_out_passed == 3 for result in results)
    assert target.read_text(encoding="utf-8") == DEBUGGING_SOURCE
    assert worker.store.goals()[0]["status"] == "completed"
