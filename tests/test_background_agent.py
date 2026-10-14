import json
import time
import hashlib

import pytest

from axen.evolution.background_agent import BackgroundAgent, arithmetic_value, evaluate_answer
from axen.evolution.practice import DEBUGGING_SOURCE, SkillPracticeAgent
from axen.evolution.models import BenchmarkResult, Candidate, Experiment, ExperimentStatus
from axen.evolution.approval import ApprovalGate
from axen.evolution.store import EvolutionStore
from tests.test_autonomous_learning import worker_at


def agent_at(tmp_path, **kwargs):
    worker = worker_at(tmp_path, config_overrides={'background_agent': {
        'enabled': True, 'auto_apply_tested_changes': True, 'cycle_interval_seconds': 3600}}, **kwargs)
    worker.store.set_state('background_last_cycle', time.time())
    return BackgroundAgent(worker)


@pytest.mark.parametrize('expression', ['__import__("os")', '2**1000000', 'a+2', '(' * 250, '1/0'])
def test_arithmetic_checker_cannot_execute_or_exhaust_resources(expression):
    with pytest.raises((ValueError, SyntaxError, ArithmeticError)): arithmetic_value(expression)


def test_actual_answers_checked_and_unknowns_not_called_verified(tmp_path):
    agent = agent_at(tmp_path)
    agent.observe('what is 25 * 48?', 'The answer is 1200.')
    agent.observe('what is 25 * 48?', 'The answer is 1200.')
    agent.observe('2 + 2', '5')
    agent.observe('Who will win tomorrow?', 'My answer is definitely correct', {'verification': {'passed': True}})
    while agent.evaluate_next(): pass
    assert agent.snapshot()['reviews'] == {'passed': 1, 'failed': 1, 'unverified': 1}
    assert len(agent.store.lessons()) == 2
    assert any(e['kind'] == 'improvement_needed' and e['data']['verified_correction'] == 4 for e in agent.store.events())
    restored = BackgroundAgent(agent.worker)
    assert not restored.evaluate_next()
    assert restored.snapshot()['reviews']['passed'] == 1


def test_auto_worker_starts_without_user_click_and_stops_cleanly(tmp_path):
    agent = agent_at(tmp_path, resources=lambda: {'available': False})
    agent.observe('2+3', '5'); agent.start()
    deadline = time.monotonic() + 3
    while not agent.snapshot()['reviews'].get('passed') and time.monotonic() < deadline: time.sleep(.02)
    agent.shutdown()
    assert agent.snapshot()['reviews']['passed'] == 1
    assert not agent.thread.is_alive()


def test_chat_priority_resources_and_emergency(tmp_path):
    agent = agent_at(tmp_path, resources=lambda: {'available': False})
    agent.observe('2+3', '5'); agent.busy = lambda: True
    agent.tick(); assert agent.status == 'yielding_to_chat'
    assert agent.snapshot()['reviews'] == {'queued': 1}
    agent.busy = lambda: False
    agent.tick(); assert agent.status == 'resource_paused'
    agent.worker.command('emergency-stop')
    agent.observe('1+1', '2'); agent.tick()
    assert agent.status == 'emergency_stopped'
    assert agent.snapshot()['reviews'] == {'passed': 1}


def test_cycles_automatic_and_rate_limited(tmp_path, monkeypatch):
    agent = agent_at(tmp_path, resources=lambda: {'available': True, 'cpu_percent': 1,
        'memory_percent': 20, 'available_memory_mb': 2048})
    calls = []
    agent.store.set_state('background_last_cycle', 0)
    monkeypatch.setattr(agent, 'promote_pending', lambda: None)
    monkeypatch.setattr(agent.worker, 'command', lambda *a, **kw: calls.append((a,kw)))
    agent.tick(); agent.tick()
    assert calls == [(('start','SUPERVISED EVOLUTION'), {'offline_demo': False})]


@pytest.mark.parametrize('tampered', [False, True])
def test_auto_code_deployment_reruns_fixed_tests_and_keeps_backup(tmp_path, monkeypatch, tampered):
    import axen.evolution.learning_worker as runtime
    agent = agent_at(tmp_path)
    monkeypatch.setattr(runtime, 'ROOT', tmp_path)
    target = tmp_path / 'inclusive_sum.py'; target.write_text(DEBUGGING_SOURCE, encoding='utf-8')
    agent.worker.config['code_targets'] = [{'path':'inclusive_sum.py','contract':'inclusive_sum_nonnegative'}]
    store = EvolutionStore(tmp_path / 'data/axen_evolution.db')
    exp = Experiment(target_file=str(target), status=ExperimentStatus.AWAITING_APPROVAL); store.save_experiment(exp)
    candidate = Candidate(experiment_id=exp.experiment_id, target_file=str(target), original_code=DEBUGGING_SOURCE,
        code=DEBUGGING_SOURCE if tampered else SkillPracticeAgent().propose(DEBUGGING_SOURCE,'range_stop_exclusive'))
    store.save_candidate(candidate)
    store.save_benchmark_result(BenchmarkResult(candidate_id=candidate.candidate_id, passed_tests=4,
        total_tests=4, held_out_passed=3, held_out_total=3, is_improvement=True))
    gate = ApprovalGate(store, backup_dir=str(tmp_path/'backup'), repo_root=str(tmp_path))
    request = gate.create_request(exp,candidate)
    agent.promote_pending()
    if tampered:
        assert target.read_text(encoding='utf-8') == DEBUGGING_SOURCE
        assert store.get_pending_approvals()
    else:
        assert target.read_text(encoding='utf-8') == candidate.code
        assert list((tmp_path/'.backups/evolution').glob('*.bak'))
        assert not store.get_pending_approvals()


def test_unconfigured_weight_activation_not_automatically_approved(tmp_path, monkeypatch):
    import axen.evolution.learning_worker as runtime
    from axen.evolution.models import ApprovalRequest
    agent = agent_at(tmp_path)
    monkeypatch.setattr(runtime,'ROOT',tmp_path)
    store = EvolutionStore(tmp_path/'data/axen_evolution.db')
    request = ApprovalRequest(request_type='weight_activation')
    store.save_approval(request)
    agent.promote_pending()
    assert store.get_pending_approvals()[0].request_id == request.request_id


@pytest.mark.parametrize('replay_count,tamper,applied', [(1,False,True),(0,False,False),(1,True,False)])
def test_automatic_weight_deployment_requires_replay_and_unchanged_evidence(tmp_path,monkeypatch,replay_count,tamper,applied):
    import axen.evolution.learning_worker as runtime
    from axen.evolution.models import ApprovalRequest
    agent = agent_at(tmp_path)
    monkeypatch.setattr(runtime,'ROOT',tmp_path)
    agent.worker.config['neural_training'] = {'enabled': True,'activation_target':'checkpoints/reviewed/approved.pt'}
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/autonomy.json').write_text(json.dumps(agent.worker.config),encoding='utf-8')
    staged=tmp_path/'checkpoints/evolution_staging/candidate.pt';staged.parent.mkdir(parents=True)
    staged.write_bytes(b'test fixture evaluated checkpoint')
    store=EvolutionStore(tmp_path/'data/axen_evolution.db')
    exp=Experiment(status=ExperimentStatus.AWAITING_APPROVAL,metadata={'weights_changed':True,'replay_count':replay_count})
    store.save_experiment(exp)
    request=ApprovalRequest(experiment_id=exp.experiment_id,request_type='weight_activation',
        benchmark_summary={'baseline_loss':2.0,'candidate_loss':1.5,'held_out_baseline':2.5,'held_out_candidate':2.0},
        metadata={'checkpoint_path':str(staged),'checkpoint_sha256':hashlib.sha256(staged.read_bytes()).hexdigest()})
    store.save_approval(request)
    if tamper:staged.write_bytes(b'changed after evaluation')
    agent.promote_pending()
    assert (tmp_path/'checkpoints/reviewed/approved.pt').exists() is applied
    assert bool(store.get_pending_approvals()) is not applied
