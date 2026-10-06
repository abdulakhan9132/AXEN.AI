from app.ai.router import ModelRouter

def run_tests():
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
    
    print(f"{'PROMPT':<50} | {'EXPECTED LLM':<20} | {'ACTUAL ROUTED LLM':<20}")
    print("-" * 95)
    
    passed = 0
    for prompt, expected in tests:
        # We test the router's internal heuristic to see which provider it would select
        is_coding = router._is_coding_task(prompt)
        actual = "qwen_ollama" if is_coding else "axen_own_brain"
        
        match = "PASS" if actual == expected else "FAIL"
        if match == "PASS": passed += 1
        
        print(f"{prompt:<50} | {expected:<20} | {actual:<20} [{match}]")
        
    print(f"\nScore: {passed}/{len(tests)}")

if __name__ == "__main__":
    run_tests()
