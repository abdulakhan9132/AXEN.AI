"""
Reproduction script for:
User: hi
Captures:
1. exact prompt/messages sent to TinyLlama
2. token count
3. generation settings
4. raw generated tokens
5. decoded generated text
6. final displayed response
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from axen.assistant.controller import AssistantController
from axen.model.hf_model import HFModelConfig


def main():
    print("=== REPRODUCING USER: 'hi' ===")
    config = HFModelConfig.from_yaml()
    controller = AssistantController(config=config)
    controller.load_model()

    # Capture prompt and generation
    user_input = "hi"
    print(f"\nUser input: {repr(user_input)}")

    messages = controller.memory.to_messages()
    messages.append({"role": "user", "content": user_input})
    print("\n1. EXACT MESSAGES SENT:")
    for m in messages:
        print(f"[{m['role'].upper()}]\n{m['content']}\n")

    tokenizer = controller.model._tokenizer
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("EXACT FORMATTED PROMPT:")
    print(repr(formatted_prompt))

    # Tokenize direct vs two-step
    direct_tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    if hasattr(direct_tokens, "input_ids"):
        d_ids = direct_tokens.input_ids
    else:
        d_ids = direct_tokens

    two_step_tokens = tokenizer(formatted_prompt, return_tensors="pt")["input_ids"]

    print("\n2. TOKEN COUNTS:")
    print(f"Direct tokenize input_ids shape: {d_ids.shape}")
    print(f"Two-step tokenize input_ids shape: {two_step_tokens.shape}")
    print(f"Direct first 5 tokens: {d_ids[0][:5].tolist()}")
    print(f"Two-step first 5 tokens: {two_step_tokens[0][:5].tolist()}")

    print("\n3. GENERATION SETTINGS:")
    print(f"max_new_tokens: {config.max_new_tokens}")
    print(f"temperature: {config.temperature}")
    print(f"top_p: {config.top_p}")

    # Run direct controller response
    print("\n4. RUNNING CONTROLLER RESPOND:")
    response = controller.respond(user_input, debug=True)
    print("\n6. FINAL DISPLAYED RESPONSE:")
    print(repr(response))
    print(response)


if __name__ == "__main__":
    main()
