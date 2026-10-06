# OPTIONAL EXTERNAL MODEL RESEARCH
# (This file is for experimental training, not used by AXEN Own Brain)
import json
import subprocess
import re
import os
import tempfile
import urllib.request
from typing import Dict, List, Optional

# Basic Tasks to evaluate across languages
TASKS = [
    {
        "prompt": "Write a function/program that takes an integer N and prints the Nth Fibonacci number (where F(0)=0, F(1)=1). Print only the result for N=10.",
        "expected_output": "55"
    },
    {
        "prompt": "Write a function/program that prints the reverse of the string 'hello world'.",
        "expected_output": "dlrow olleh"
    }
]

LANGUAGES = {
    "python": {"ext": ".py", "cmd": ["python"], "check": ["python", "--version"]},
    "javascript": {"ext": ".js", "cmd": ["node"], "check": ["node", "--version"]},
    "typescript": {"ext": ".ts", "cmd": ["ts-node"], "check": ["ts-node", "--version"]},
    "go": {"ext": ".go", "cmd": ["go", "run"], "check": ["go", "version"]},
    "rust": {"ext": ".rs", "cmd": ["rustc", "-o", "out_bin"], "run_bin": ["./out_bin"], "check": ["rustc", "--version"]},
    "java": {"ext": ".java", "cmd": ["java"], "check": ["java", "-version"]},
    "cpp": {"ext": ".cpp", "cmd": ["g++", "-o", "out_bin"], "run_bin": ["./out_bin"], "check": ["g++", "--version"]},
    "bash": {"ext": ".sh", "cmd": ["bash"], "check": ["bash", "--version"]},
}

def check_runtime(lang: str) -> bool:
    try:
        subprocess.run(LANGUAGES[lang]["check"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def query_ollama(model: str, prompt: str) -> str:
    url = "http://localhost:11434/api/generate"
    data = {"model": model, "prompt": prompt, "stream": False, "temperature": 0.1}
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))["response"]
    except Exception as e:
        print(f"Ollama request failed: {e}")
        return ""

def extract_code(text: str, lang: str) -> str:
    pattern = rf"```(?:{lang})?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0] if matches else text

def run_code(lang: str, code: str) -> str:
    ext = LANGUAGES[lang]["ext"]
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, f"main{ext}")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code)
            
        cmd = LANGUAGES[lang]["cmd"] + [filepath]
        try:
            if "run_bin" in LANGUAGES[lang]:
                # Compile first
                compile_cmd = LANGUAGES[lang]["cmd"] + [filepath]
                # Adjust output path to be in tempdir
                compile_cmd[2] = os.path.join(tmpdir, "out_bin")
                subprocess.run(compile_cmd, cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, check=True)
                # Run binary
                run_cmd = [os.path.join(tmpdir, "out_bin")]
                res = subprocess.run(run_cmd, cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, text=True)
            else:
                res = subprocess.run(cmd, cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, text=True)
            return res.stdout.strip()
        except subprocess.TimeoutExpired:
            return "TIMEOUT"
        except subprocess.CalledProcessError as e:
            return f"ERROR: {e.stderr}"
        except Exception as e:
            return str(e)

def evaluate_model(model: str) -> Dict[str, float]:
    print(f"\nEvaluating model: {model}")
    results = {}
    
    for lang, config in LANGUAGES.items():
        if not check_runtime(lang):
            print(f"Skipping {lang}: runtime not installed.")
            continue
            
        print(f"Testing {lang}...")
        passed = 0
        for task in TASKS:
            prompt = f"Write a {lang} program to: {task['prompt']}. Output ONLY the code inside markdown code blocks."
            response = query_ollama(model, prompt)
            code = extract_code(response, lang)
            output = run_code(lang, code)
            
            if task["expected_output"] in output:
                passed += 1
                
        pass_rate = (passed / len(TASKS)) * 100
        results[lang] = pass_rate
        print(f"  {lang} Pass Rate: {pass_rate:.1f}% ({passed}/{len(TASKS)})")
        
    return results

def main():
    base_model = "qwen2.5-coder:3b"
    ft_model = "axen-coder"
    
    print("--- BASE MODEL ---")
    base_results = evaluate_model(base_model)
    
    print("\n--- FINE-TUNED MODEL ---")
    ft_results = evaluate_model(ft_model)
    
    print("\n=== FINAL COMPARISON ===")
    print(f"{'Language':<15} | {'Base Model':<12} | {'Fine-Tuned':<12}")
    print("-" * 45)
    for lang in LANGUAGES.keys():
        if lang in base_results or lang in ft_results:
            b_score = f"{base_results.get(lang, 0):.1f}%"
            f_score = f"{ft_results.get(lang, 0):.1f}%"
            print(f"{lang:<15} | {b_score:<12} | {f_score:<12}")

if __name__ == "__main__":
    main()
