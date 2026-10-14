"""Automatic local evaluation and bounded improvement scheduling beside chat.

No second inference model is loaded. Unsupported answers remain unverified;
test-backed promotion uses the existing audited gate and configured targets.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import operator
import re
import threading
import time

from .learning_store import now


def arithmetic_value(expression):
    """Independent bounded arithmetic check; never execute model/user code."""
    expression = expression.strip().replace('×', '*').replace('÷', '/')
    if len(expression) > 200 or not re.fullmatch(r'[\d\s.()+*/%^-]+', expression):
        raise ValueError('Not a supported arithmetic expression')
    tree = ast.parse(expression.replace('^', '**'), mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 70:
        raise ValueError('Expression exceeds evaluation budget')
    binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
              ast.Mod: operator.mod, ast.Pow: operator.pow}
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in binary:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 12:
                raise ValueError('Exponent exceeds budget')
            value = binary[type(node.op)](left, right)
        else:
            raise ValueError('Unsupported arithmetic')
        if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 1e15:
            raise ValueError('Result exceeds budget')
        return value
    return visit(tree.body)


def evaluate_answer(question, answer, metadata):
    report = {'evaluated_at': now(), 'scope': 'answer checks', 'status': 'unverified',
              'findings': [], 'independently_verified': False}
    if not answer.strip():
        report.update(status='failed', findings=['Empty answer'])
        return report
    expression = re.sub(r'^(?:what is|calculate|compute)\s+', '', question.strip(), flags=re.I).rstrip(' ?')
    try:
        expected = arithmetic_value(expression)
        result_text = re.sub(r'^(?:the answer is|the result is)\s+', '', answer.strip(), flags=re.I).rstrip(' .')
        actual = float(result_text.replace(',', ''))
        passed = math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)
        report.update(status='passed' if passed else 'failed', scope='independent arithmetic',
                      independently_verified=True, expected=expected, actual=actual,
                      findings=[] if passed else ['Arithmetic result differs from independently computed result'])
    except (ValueError, SyntaxError, ArithmeticError, OverflowError):
        verification = metadata.get('verification')
        if verification is False or (isinstance(verification, dict) and verification.get('passed') is False):
            report.update(status='failed', findings=['Answer agent reported a verification failure'])
        else:
            report['findings'] = ['No independent validator for this answer; correctness is not established']
    return report


class BackgroundAgent:
    def __init__(self, worker, *, busy=lambda: False):
        self.worker, self.store, self.busy = worker, worker.store, busy
        self.policy = worker.config.get('background_agent', {})
        self.stop_event = threading.Event()
        self.thread = None
        self.status = 'stopped'
        self.last_error = None
        self.worker.foreground_busy = busy
        with self.store.connection() as db:
            db.execute('CREATE TABLE IF NOT EXISTS answer_reviews (id TEXT PRIMARY KEY, status TEXT NOT NULL, data TEXT NOT NULL, report TEXT)')

    def start(self):
        if not self.policy.get('enabled'): return
        if self.thread and self.thread.is_alive(): return
        self.status = 'waiting_for_answers'
        self.thread = threading.Thread(target=self._run, name='axen-background-evaluator', daemon=False)
        self.thread.start()
        self.store.event('background_started', {'automatic': True, 'role': 'evaluation_and_improvement'})

    def observe(self, question, answer, metadata=None):
        if not self.policy.get('enabled') or self.worker.emergency: return
        data = {'question': str(question)[:8000], 'answer': str(answer)[:16000], 'metadata': metadata or {}}
        key = hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()
        with self.store.connection() as db:
            count = db.execute("SELECT COUNT(*) FROM answer_reviews WHERE status='queued'").fetchone()[0]
            if count >= 200: return
            db.execute('INSERT OR IGNORE INTO answer_reviews VALUES (?,?,?,NULL)',
                       (key, 'queued', json.dumps(data, default=str)))

    def evaluate_next(self):
        with self.store.connection() as db:
            row = db.execute("SELECT * FROM answer_reviews WHERE status='queued' ORDER BY rowid LIMIT 1").fetchone()
        if row is None: return False
        data = json.loads(row['data'])
        report = evaluate_answer(data['question'], data['answer'], data['metadata'])
        with self.store.connection() as db:
            db.execute('UPDATE answer_reviews SET status=?,report=? WHERE id=?',
                       (report['status'], json.dumps(report), row['id']))
        self.store.event('answer_evaluation', dict(report, request_hash=row['id']))
        if report['status'] == 'failed':
            self.store.event('improvement_needed', {'request_hash': row['id'], 'findings': report['findings'],
                             'verified_correction': report.get('expected'), 'production_changed': False})
        if report.get('independently_verified'):
            from axen.memory.memory_models import MemoryItem
            entry = {'goal_id': 'answer_' + row['id'], 'topic': 'Arithmetic calculation',
                     'summary': f"For the expression {data['question'][:200]!r}, the independently calculated result is {report['expected']}.",
                     'verification_status': 'verified_by_practice',
                     'source_references': [{'source': 'bounded_arithmetic_validator', 'request_hash': row['id']}],
                     'confidence_metadata': {'scope': 'This exact arithmetic expression only'},
                     'last_validation_date': report['evaluated_at']}
            key = 'answer_lesson_' + row['id']
            if not self.worker.knowledge.memory.get_by_key(key):
                self.worker.knowledge.memory.add(MemoryItem(key=key, value=json.dumps(entry),
                    memory_type='technical_fact', source='verified_answer_evaluation', confidence=0.99))
            self.store.lesson(entry)
        return True

    def promote_pending(self):
        """Operator-authorized automatic approval; skip unconfigured/unmeasured work."""
        if not self.policy.get('auto_apply_tested_changes'): return
        from .approval import ApprovalGate
        from .store import EvolutionStore
        from .learning_worker import ROOT
        store = EvolutionStore(ROOT / 'data/axen_evolution.db')
        gate = ApprovalGate(store, backup_dir=str(ROOT / '.backups/evolution'), repo_root=str(ROOT))
        for request in gate.list_pending():
            if self.worker.emergency or self.stop_event.is_set() or self.busy(): return
            if self.store.state('auto_decision_' + request.request_id): continue
            eligible = False
            if request.request_type == 'code_merge':
                try:
                    target = gate._target(request.target_file)
                    eligible = any(t.get('contract') == 'inclusive_sum_nonnegative' and gate._target(t['path']) == target
                                   for t in self.worker.config.get('code_targets', []))
                    candidate = store.get_candidate(request.candidate_id)
                    benchmarks = store.list_benchmark_results(candidate_id=request.candidate_id)
                    eligible = eligible and candidate is not None and any(
                        b.is_improvement and not b.is_overfitting and b.total_tests > 0 and b.held_out_total > 0
                        and b.passed_tests == b.total_tests and b.held_out_passed == b.held_out_total for b in benchmarks)
                    if eligible:
                        from .practice import TRAIN_CASES, HELD_OUT_CASES
                        cancel = lambda: self.stop_event.is_set() or self.worker.emergency or self.busy()
                        checked = [self.worker.practice.evaluate(candidate.code, cases, cancel=cancel)
                                   for cases in (TRAIN_CASES, HELD_OUT_CASES)]
                        eligible = all(r['total'] > 0 and r['passed'] == r['total'] for r in checked)
                except (ValueError, PermissionError, KeyError): eligible = False
            elif request.request_type == 'weight_activation':
                exp = store.get_experiment(request.experiment_id)
                evidence = exp.metadata if exp else {}
                train = self.worker.config.get('neural_training', {})
                summary = request.benchmark_summary
                eligible = bool(train.get('enabled') and train.get('activation_target')
                    and evidence.get('weights_changed') and evidence.get('replay_count', 0) > 0
                    and all(type(summary.get(k)) in (int,float) and math.isfinite(summary[k]) and summary[k] > 0
                            for k in ('baseline_loss','candidate_loss','held_out_baseline','held_out_candidate'))
                    and summary['candidate_loss'] < summary['baseline_loss']
                    and summary['held_out_candidate'] <= summary['held_out_baseline'])
            if not eligible: continue
            try:
                # Serialize deployment with the emergency latch.
                with self.worker.lock:
                    if self.worker.emergency or self.busy(): return
                    gate.approve(request.request_id, reviewer='operator_authorized_background_policy',
                                 notes='Automatic promotion: fixed tests and held-out checks passed; backup retained')
                self.store.set_state('auto_decision_' + request.request_id, 'applied')
                self.store.event('automatic_change_applied', {'request_id': request.request_id,
                                 'kind': request.request_type, 'rollback_available': True})
            except (ValueError, PermissionError, OSError) as exc:
                self.store.set_state('auto_decision_' + request.request_id, 'deferred')
                self.store.event('automatic_change_deferred', {'request_id': request.request_id, 'reason': str(exc)})

    def tick(self):
        self.worker.foreground_busy = self.busy
        if self.worker.emergency:
            self.status = 'emergency_stopped'; return
        if self.busy():
            self.status = 'yielding_to_chat'; return
        # Small answer checks do not allocate a second model or run source code.
        self.status = 'evaluating' if self.evaluate_next() else 'waiting_for_answers'
        resources = self.worker.resources()
        if (not resources.get('available') or resources.get('cpu_percent',100) >= self.worker.config['cpu_busy_percent']
            or resources.get('memory_percent',100) >= self.worker.config['memory_busy_percent']
            or resources.get('available_memory_mb',0) < self.worker.config.get('minimum_available_mb',128)):
            self.status = 'resource_paused'; return
        self.promote_pending()
        # Each research/training cycle retains the existing finite worker budgets.
        last = self.store.state('background_last_cycle', 0)
        if time.time() - last < self.policy.get('cycle_interval_seconds', 3600): return
        if self.worker.thread and self.worker.thread.is_alive(): return
        if self.worker.mode != 'OFF': return
        self.store.set_state('background_last_cycle', time.time())
        self.worker.command('start', 'SUPERVISED EVOLUTION', offline_demo=False)
        self.status = 'improving'

    def _run(self):
        while not self.stop_event.is_set():
            try: self.tick(); self.last_error = None
            except Exception as exc:
                self.status = 'error'
                if self.last_error != str(exc):
                    self.store.event('background_error', {'reason': str(exc)})
                self.last_error = str(exc)
            self.stop_event.wait(2)
        self.status = 'stopped'

    def snapshot(self):
        with self.store.connection() as db:
            counts = {r[0]: r[1] for r in db.execute('SELECT status,COUNT(*) FROM answer_reviews GROUP BY status')}
        return {'enabled': bool(self.policy.get('enabled')), 'automatic': True,
                'status': self.status, 'reviews': counts, 'last_error': self.last_error,
                'auto_apply_tested_changes': bool(self.policy.get('auto_apply_tested_changes')),
                'code_targets': len(self.worker.config.get('code_targets', [])),
                'training_configured': bool(self.worker.config.get('neural_training', {}).get('enabled')),
                'deployment_effect': 'Tested files/checkpoints are backed up and written on disk; loaded code/models change on restart.',
                'scope': 'Independent arithmetic checks; other answers remain unverified. Code/model promotion requires configured evaluation contracts.'}

    def shutdown(self):
        self.stop_event.set()
        if self.thread: self.thread.join(timeout=15)
        if self.thread and self.thread.is_alive(): raise RuntimeError('Background evaluator did not stop')


def get_background_agent(worker=None):
    if worker is None:
        from .learning_worker import get_worker
        worker = get_worker()
    with worker.lock:
        if not hasattr(worker, 'background_agent'): worker.background_agent = BackgroundAgent(worker)
        return worker.background_agent
