"""
axen/evolution/meta_agent.py
===========================
MetaAgent for the AXEN Self-Evolution Engine (HyperAgents architecture).

The MetaAgent observes TaskAgent execution traces, identifies structural
inefficiencies in problem-solving strategies, and proposes higher-level
modifications to prompt templates, reasoning budgets, planning depth,
or tool selection heuristics.

Operates strictly under `AgentRole.META_AGENT`.
It CANNOT modify security policies, cannot modify production code directly,
and CANNOT self-grant `APPROVE_MERGE`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from .permissions import AgentRole, EvolutionPermission, EvolutionSecurityPolicy
from .task_agent import TaskAgent, TaskExecutionResult

logger = logging.getLogger("axen.evolution.meta_agent")


@dataclass
class StrategyProposal:
    """A proposed modification to agent strategy or configuration."""

    proposal_id: str
    target_area: str  # e.g. "prompt_template", "planning_depth", "tool_selection", "reasoning_budget"
    rationale: str
    proposed_changes: Dict[str, Any]
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    projected_improvements: Dict[str, float] = field(default_factory=dict)
    status: str = "PROPOSED"  # PROPOSED, TESTED, REJECTED, ACCEPTED
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "target_area": self.target_area,
            "rationale": self.rationale,
            "proposed_changes": self.proposed_changes,
            "baseline_metrics": self.baseline_metrics,
            "projected_improvements": self.projected_improvements,
            "status": self.status,
            "created_at": self.created_at,
        }


class MetaAgent:
    """
    Higher-order agent that formulates optimization proposals for TaskAgent.

    Analyzes batches of execution results and generates improved strategy templates.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        security_policy: Optional[EvolutionSecurityPolicy] = None,
    ) -> None:
        self.model = model
        self.security_policy = security_policy or EvolutionSecurityPolicy(
            current_role=AgentRole.META_AGENT
        )

    def analyze_performance(
        self, execution_results: List[TaskExecutionResult]
    ) -> Dict[str, Any]:
        """
        Synthesize performance metrics across a run of tasks.
        """
        self.security_policy.authorize(EvolutionPermission.READ_CANDIDATES)

        total = len(execution_results)
        if total == 0:
            return {
                "total_tasks": 0,
                "success_rate": 0.0,
                "avg_duration": 0.0,
                "avg_steps": 0.0,
                "failure_count": 0,
            }

        successes = sum(1 for r in execution_results if r.success)
        durations = [r.duration_seconds for r in execution_results]
        steps = [r.steps_taken for r in execution_results]
        failures = [r for r in execution_results if not r.success]

        return {
            "total_tasks": total,
            "success_rate": round(successes / total, 4),
            "avg_duration": round(sum(durations) / total, 4),
            "avg_steps": round(sum(steps) / total, 2),
            "failure_count": len(failures),
            "sample_errors": [f.error for f in failures[:5] if f.error],
        }

    def propose_strategy_improvement(
        self,
        execution_results: List[TaskExecutionResult],
        current_strategy: Dict[str, Any],
    ) -> StrategyProposal:
        """
        Generate a concrete strategy modification proposal based on observed results.
        """
        self.security_policy.authorize(EvolutionPermission.PROPOSE_META_CHANGE)

        metrics = self.analyze_performance(execution_results)
        proposal_id = f"meta_{uuid.uuid4().hex[:8]}"

        # If model is available, use LLM prompt reasoning
        if self.model is not None and hasattr(self.model, "generate"):
            prompt = (
                f"Analyze the following TaskAgent run metrics:\n"
                f"- Success Rate: {metrics['success_rate']*100:.1f}%\n"
                f"- Average Duration: {metrics['avg_duration']:.2f}s\n"
                f"- Average Steps: {metrics['avg_steps']}\n"
                f"- Recent Errors: {metrics.get('sample_errors', [])}\n"
                f"Current Configuration: {json.dumps(current_strategy)}\n\n"
                f"Propose ONE specific tuning to improve task pass rate or execution speed.\n"
                f"Return JSON format:\n"
                f'{{"target_area": "...", "rationale": "...", "proposed_changes": {{...}}, "expected_gain": 0.1}}'
            )
            try:
                response = self.model.generate(
                    prompt=prompt,
                    system="You are the AXEN MetaAgent strategy optimizer. Return valid JSON only.",
                    temperature=0.3,
                )
                text = getattr(response, "text", str(response)).strip()
                # Parse JSON
                if "```json" in text:
                    text = text.split("```json", 1)[1].split("```", 1)[0].strip()
                elif "```" in text:
                    text = text.split("```", 1)[1].split("```", 1)[0].strip()
                parsed = json.loads(text)

                return StrategyProposal(
                    proposal_id=proposal_id,
                    target_area=parsed.get("target_area", "system_prompt"),
                    rationale=parsed.get("rationale", "Model-suggested strategy improvement"),
                    proposed_changes=parsed.get("proposed_changes", {}),
                    baseline_metrics={"success_rate": metrics["success_rate"]},
                    projected_improvements={"success_rate": metrics["success_rate"] + float(parsed.get("expected_gain", 0.05))},
                )
            except Exception as exc:
                logger.warning("MetaAgent model-based generation failed (%s), using rule-based fallback", exc)

        # Rule-based fallback proposals based on metrics
        if metrics["success_rate"] < 0.6:
            return StrategyProposal(
                proposal_id=proposal_id,
                target_area="planning_depth",
                rationale="Pass rate below 60%. Increasing reasoning budget and step threshold.",
                proposed_changes={
                    "max_steps": current_strategy.get("max_steps", 5) + 3,
                    "temperature": max(0.1, current_strategy.get("temperature", 0.5) - 0.1),
                    "enable_verification": True,
                },
                baseline_metrics={"success_rate": metrics["success_rate"]},
                projected_improvements={"success_rate": min(1.0, metrics["success_rate"] + 0.15)},
            )
        elif metrics["avg_duration"] > 10.0:
            return StrategyProposal(
                proposal_id=proposal_id,
                target_area="tool_selection",
                rationale="High average task duration. Pruning unnecessary tool exploration paths.",
                proposed_changes={
                    "max_tool_calls": max(1, current_strategy.get("max_tool_calls", 3) - 1),
                    "early_stopping": True,
                },
                baseline_metrics={"avg_duration": metrics["avg_duration"]},
                projected_improvements={"avg_duration": metrics["avg_duration"] * 0.75},
            )
        else:
            return StrategyProposal(
                proposal_id=proposal_id,
                target_area="prompt_template",
                rationale="System stable; refining concise chain-of-thought prompt guidance.",
                proposed_changes={
                    "system_prompt_addon": "Think step-by-step and produce verified concise solutions."
                },
                baseline_metrics={"success_rate": metrics["success_rate"]},
                projected_improvements={"success_rate": min(1.0, metrics["success_rate"] + 0.02)},
            )
