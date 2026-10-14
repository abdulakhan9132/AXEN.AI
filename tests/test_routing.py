from app.ai.router import ModelRouter


def test_model_router_heuristics():
    router = ModelRouter()

    tests = [
        ("hi", "axen_own_brain"),
        ("Who are you?", "axen_own_brain"),
        ("What is AI?", "axen_own_brain"),
        ("What is 25 * 48?", "axen_own_brain"),
        ("Convert 101101 to decimal", "axen_own_brain"),
        ("Write a Python function", "qwen_ollama"),
        ("Debug this JavaScript", "qwen_ollama"),
        ("Review this repository", "qwen_ollama"),
        ("What do you remember about me?", "axen_own_brain"),
        ("Search my project for authentication code", "qwen_ollama"),
    ]

    for prompt, expected in tests:
        is_coding = router._is_coding_task(prompt)
        actual = "qwen_ollama" if is_coding else "axen_own_brain"
        assert actual == expected, f"Routing failed for '{prompt}': expected {expected}, got {actual}"
