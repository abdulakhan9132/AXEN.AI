"""
tests/test_meta_agent.py
========================
Tests for MetaAgent performance analysis and strategy proposals.
"""

import pytest
from axen.evolution.meta_agent import MetaAgent, StrategyProposal
from axen.evolution.permissions import AgentRole, EvolutionPermission, EvolutionSecurityPolicy
from axen.evolution.task_agent import TaskExecutionResult


def test_meta_agent_analysis_and_proposal():
    meta = MetaAgent()
    results = [
        TaskExecutionResult(task_id="t1", success=False, output="", duration_seconds=12.0, steps_taken=6),
        TaskExecutionResult(task_id="t2", success=False, output="", duration_seconds=14.0, steps_taken=7),
        TaskExecutionResult(task_id="t3", success=True, output="ok", duration_seconds=11.0, steps_taken=5),
    ]

    analysis = meta.analyze_performance(results)
    assert analysis["total_tasks"] == 3
    assert analysis["success_rate"] < 0.5
    assert analysis["failure_count"] == 2

    proposal = meta.propose_strategy_improvement(
        execution_results=results,
        current_strategy={"max_steps": 5, "temperature": 0.5},
    )

    assert isinstance(proposal, StrategyProposal)
    assert proposal.proposal_id.startswith("meta_")
    assert "planning_depth" in proposal.target_area or "tool_selection" in proposal.target_area


def test_meta_agent_cannot_self_grant_approve_merge():
    policy = EvolutionSecurityPolicy(current_role=AgentRole.META_AGENT)
    assert not policy.can(EvolutionPermission.APPROVE_MERGE)

    with pytest.raises(PermissionError):
        policy.validate_elevation_attempt({EvolutionPermission.APPROVE_MERGE})
