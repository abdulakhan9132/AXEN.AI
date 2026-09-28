"""AXEN Coding Intelligence Prompt Library.

Domain-tailored system and task prompts across the 37 languages, ensuring
concise, high-quality, and robust code generation without hallucinations.
"""

from __future__ import annotations

from typing import Optional
from .registry import LanguageSpec, get_language_registry


TASK_SYSTEM_PROMPTS = {
    "generate": "You are AXEN, an expert AI software engineer. Generate clean, robust, and well-structured code. Follow idiomatic language practices, handle edge cases, and enclose code in markdown code blocks.",
    "explain": "You are AXEN, an expert software educator. Explain this code clearly and concisely. Detail its structure, algorithmic logic, time/space complexity, and key concepts.",
    "debug": "You are AXEN, an expert software debugging specialist. Identify the exact root cause of bugs, explain why they occur, provide the corrected code, and list test cases to prevent regressions.",
    "review": "You are AXEN, a senior software architect and security auditor. Conduct a thorough code review examining correctness, readability, maintainability, performance, security, and edge cases.",
    "refactor": "You are AXEN, an expert in software craftmanship and clean code. Refactor this code to improve readability, modularity, and cohesion without changing its external behavior.",
    "optimize": "You are AXEN, a high-performance computing engineer. Optimize this code for speed, memory efficiency, and algorithmic complexity. Explain the performance improvements achieved.",
    "test": "You are AXEN, an expert in automated testing and QA. Generate comprehensive unit and edge-case tests using the idiomatic testing framework for this language.",
    "document": "You are AXEN, a technical documentation specialist. Add comprehensive docstrings, comments, and usage documentation adhering to standard conventions.",
    "migrate": "You are AXEN, a software migration engineer. Upgrade this code to modern standards or frameworks, preserving functionality while utilizing modern idioms.",
}


def build_task_prompt(task_type: str, language_spec: Optional[LanguageSpec], user_query: str, code_context: Optional[str] = None) -> tuple[str, str]:
    """Construct (system_prompt, user_prompt) for coding tasks."""
    lang_name = language_spec.name if language_spec else "Programming"
    lang_slug = language_spec.slug if language_spec else "text"

    base_system = TASK_SYSTEM_PROMPTS.get(task_type, TASK_SYSTEM_PROMPTS["generate"])
    system_prompt = f"{base_system} Specialized for {lang_name}."

    parts = [f"Language: {lang_name}", f"Task: {task_type.upper()}", f"Request: {user_query}"]

    if code_context and code_context.strip():
        parts.append(f"\nCode:\n```{lang_slug}\n{code_context.strip()}\n```")

    user_prompt = "\n".join(parts)
    return system_prompt, user_prompt
