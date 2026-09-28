"""Prompt templates used by the local agent."""

SYSTEM_PROMPT = """
You are an autonomous local AI agent operating on a Windows workstation.
Follow safety rules, keep decisions concise, and surface only brief summaries.
Never expose hidden reasoning or chain-of-thought.
"""

PLAN_TEMPLATE = """
Goal: {goal}

Create a succinct, structured plan as JSON with these fields:
{
  "goal": "...",
  "steps": [
    {"id": 1, "objective": "...", "tool": "...", "requires_approval": false, "arguments": {}}
  ]
}

Only use tools from this local environment: filesystem, calculator, shell, browser, memory.
Return valid JSON only.
"""

VERIFICATION_TEMPLATE = """
Task: {task}
Context: {context}

Return JSON with fields: success, checks, evidence, notes.
If the result is uncertain, set success to false and explain why.
"""

SUMMARY_TEMPLATE = """
Provide a concise summary of the task outcome.
Keep it to 2-4 sentences.
Include final result and any warnings.
"""
