"""
Interactive test script for AXEN General-Purpose AI Assistant Brain.
Tests realistic questions across the core categories.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from axen.assistant.controller import AssistantController
from axen.memory.conversation_memory import ConversationMemory
from axen.memory.long_term_memory import LongTermMemory
from axen.tools import build_default_registry


class DemoModel:
    is_loaded = True
    model_name = "TinyLlama-1.1B"
    _device = "cpu"

    def chat(self, messages: list[dict[str, str]]) -> str:
        req = messages[-1]["content"]
        if "API" in req:
            return (
                "Think of an API like a waiter in a restaurant. You are the customer at the table, "
                "the kitchen is the system that prepares the food, and the waiter is the messenger "
                "that takes your order, delivers it to the kitchen, and brings back the dish."
            )
        elif "RAM" in req:
            return (
                "Increasing RAM provides more active workspace for programs. Applications that manipulate "
                "large datasets, browser tabs, or cache in-memory benefit significantly by avoiding disk paging. "
                "However, CPU-bound or single-threaded tasks that don't exhaust RAM will see zero speed improvement."
            )
        elif "leave email" in req:
            return (
                "Subject: Leave Application - [Your Name]\n\n"
                "Dear [Manager Name],\n\n"
                "I am writing to formally request leave from [Start Date] to [End Date] due to personal reasons. "
                "I have delegated critical tasks to [Colleague Name] and will remain reachable via email for urgent matters.\n\n"
                "Thank you for understanding.\n\n"
                "Warm regards,\n[Your Name]"
            )
        elif "Bangalore" in req:
            return (
                "Surviving 15 days in Bangalore on ₹3,000 gives a strict daily budget of ₹200/day:\n"
                "1. Food (₹130-₹150/day): Have breakfast & lunch at local Darshinis or Indira Canteens (₹10-₹40/meal).\n"
                "2. Travel (₹30-₹40/day): Use BMTC non-AC buses or walk for short distances.\n"
                "3. Essentials & Buffer (₹20-₹30/day): Keep for drinking water and emergency medicines."
            )
        elif "portfolio website" in req:
            return (
                "To build your portfolio website, here is the recommended architecture:\n"
                "1. Hero Section: Name, punchy title, call to action (Resume / Contact).\n"
                "2. About & Tech Stack: Concise bio and interactive skill badges.\n"
                "3. Featured Projects: Grid with screenshots, live demo links, and GitHub repos.\n"
                "4. Contact Form: Simple clean email contact."
            )
        elif "reverse a string" in req:
            return (
                "Here is the standard Python function using slicing:\n\n"
                "```python\n"
                "def reverse_string(s: str) -> str:\n"
                "    return s[::-1]\n"
                "```"
            )
        elif "error" in req:
            return (
                "In Python, this typically means a variable was referenced before assignment, "
                "or there is a mismatch in data types (TypeError)."
            )
        elif "Japanese" in req:
            return "In Japanese: この文を日本語に翻訳してください (Kono bun o nihongo ni hon'yaku shite kudasai)."
        elif "30-day plan" in req:
            return (
                "30-Day Python Learning Plan:\n"
                "- Week 1: Core Syntax (Variables, conditionals, loops)\n"
                "- Week 2: Data Structures (Lists, dictionaries, tuples, sets)\n"
                "- Week 3: Functional Programming & OOP (Functions, classes, modules)\n"
                "- Week 4: Project & Real-World Practice (CLI tool or data analysis mini-app)"
            )
        return "I understand your request and am ready to assist."

    def chat_with_stats(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        ans = self.chat(messages)
        return ans, {"input_tokens": 10, "generated_tokens": 15, "roles": [m["role"] for m in messages]}


def run_demo():
    controller = AssistantController(
        model=DemoModel(),
        tool_registry=build_default_registry(),
        conversation_memory=ConversationMemory(),
        long_term_memory=LongTermMemory(path=None),
    )

    test_cases = [
        ("CONVERSATION", "Hi, how are you?"),
        ("KNOWLEDGE", "What is an API?"),
        ("REASONING", "Why does increasing RAM help some applications but not others?"),
        ("MATH", "What is 238 * 47?"),
        ("SYMBOLIC MATH", "Differentiate f(x)=(2x^3-5x+4)(x^2+3x-1)"),
        ("PROGRAMMING", "Write a Python function to reverse a string."),
        ("DEBUGGING", "Why does this Python code produce an error?"),
        ("WRITING", "Write a professional leave email."),
        ("TRANSLATION", "Translate this sentence into Japanese."),
        ("PLANNING", "I have 3000 rupees and need to survive for 15 days in Bangalore"),
        ("PLANNING (PYTHON)", "Create a 30-day plan to learn Python."),
        ("AMBIGUOUS", "Make it better."),
    ]

    print("=================================================================")
    print("       AXEN GENERAL-PURPOSE AI ASSISTANT BRAIN DEMO             ")
    print("=================================================================\n")

    for cat, query in test_cases:
        ans = controller.respond(query)
        dbg = controller.last_debug_info
        print(f"[{cat}]")
        print(f"User: \"{query}\"")
        print(f"-> Intent: {dbg.get('intent')} | Complexity: {dbg.get('complexity')} | Tool: {dbg.get('selected_tool') or 'none'} | Verification: {dbg.get('verification')}")
        first_line = ans.strip().split("\n")[0]
        print(f"Axen: {first_line[:90]}...\n")

    # Follow-up test
    print("[FOLLOW-UP TEST]")
    controller.reset_conversation()
    q1 = "Who invented Python?"
    a1 = controller.respond(q1)
    print(f"User: {q1}\nAxen: {a1}")

    q2 = "When was it released?"
    a2 = controller.respond(q2)
    dbg2 = controller.last_debug_info
    print(f"User: {q2}\n-> Intent: {dbg2.get('intent')} | Subject: {dbg2.get('selected_tool') or 'context resolved'}")
    print(f"Axen: {a2.splitlines()[0]}\n")

    print("[MEMORY INTEGRATION TEST]")
    controller.long_memory.remember("user_style", "I prefer minimal dark UI with sleek typography.")
    controller.long_memory.remember("pet", "Cat named Luna.")
    q_mem = "Design a dashboard for me."
    a_mem = controller.respond(q_mem)
    dbg_mem = controller.last_debug_info
    print(f"User: {q_mem}")
    print(f"-> Memory retrieved count: {dbg_mem.get('memory_retrieved')}")
    print(f"-> Retrieved: {[m['key'] for m in controller.last_agent_state.metadata.get('retrieved_memories', [])]}")
    print()


if __name__ == "__main__":
    run_demo()
