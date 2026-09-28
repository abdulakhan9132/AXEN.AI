"""
Verification of the 4 exact messages requested by the user:
1. hi
2. Hello, who are you?
3. What is 2 + 2?
4. Tell me a joke.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from axen.assistant.controller import AssistantController
from axen.model.hf_model import HFModelConfig


def main():
    print("==================================================")
    print("FRESH PROCESS TEST: 4 TARGET CONVERSATION PROMPTS")
    print("==================================================")

    config = HFModelConfig.from_yaml()
    controller = AssistantController(config=config)
    controller.load_model()
    print("Controller and local TinyLlama model ready.\n")

    test_queries = [
        "hi",
        "Hello, who are you?",
        "What is 2 + 2?",
        "Tell me a joke.",
    ]

    for q in test_queries:
        messages = controller.memory.to_messages()
        sys_content = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
        print("=== AXEN INPUT ===")
        print(f"SYSTEM: {sys_content}")
        print(f"USER: {q}")
        print()

        response = controller.respond(q, debug=True)

        print("=== AXEN OUTPUT ===")
        print(response)
        print()
        dbg = controller.last_debug_info
        print(f"Task: {dbg.get('task_type')} | Tool: {dbg.get('selected_tool')} | Status: {dbg.get('final_status')}")
        print("-" * 50)
        print()


if __name__ == "__main__":
    main()
