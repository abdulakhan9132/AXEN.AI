"""
tests/test_task_agent.py
========================
Tests for TaskAgent problem solving, telemetry capture, and sandbox permissions.
"""

import pytest
from axen.evolution.permissions import AgentRole, EvolutionPermission, EvolutionSecurityPolicy
from axen.evolution.task_agent import TaskAgent, TaskExecutionResult


def test_task_agent_execution_success():
    class DummyModel:
        def generate(self, prompt, **kwargs):
            return "42"

    agent = TaskAgent(model=DummyModel())
    result = agent.execute_task(
        task_id="task_1",
        prompt="What is the answer?",
        expected_output="42",
    )

    assert result.success is True
    assert result.verifier_passed is True
    assert result.output == "42"
    assert result.duration_seconds > 0.0


def test_task_agent_permission_boundary():
    # If policy is restricted without EXECUTE_SANDBOX, task fails or raises
    policy = EvolutionSecurityPolicy(current_role=AgentRole.TASK_AGENT)
    # TaskAgent has EXECUTE_SANDBOX by default
    assert policy.can(EvolutionPermission.EXECUTE_SANDBOX)
    # But TaskAgent CANNOT approve merge
    assert not policy.can(EvolutionPermission.APPROVE_MERGE)

    with pytest.raises(PermissionError):
        policy.authorize(EvolutionPermission.APPROVE_MERGE)
