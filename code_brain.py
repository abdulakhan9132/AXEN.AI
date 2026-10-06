# code_brain.py - code generation brain for AXEN AI
import ast, json, re, subprocess, sys, tempfile, shutil
from pathlib import Path
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5-coder:3b"

SYSTEM = (
    "You are AXEN Code, an expert software engineer. Write complete, working, clean code. "
    "Return the code in ONE fenced block with the language tag, then at most 3 short lines of "
    "explanation. Do not leave TODOs or placeholders. Handle errors and edge cases. "
    "Follow the user's language and framework exactly."
)

FENCE = re.compile(r"```(\w+)?\n(.*?)```", re.S)

def extract_code(text: str):
    m = FENCE.search(text)
    if m:
        return (m.group(1) or "text").lower(), m.group(2).strip()
    return "text", text.strip()

def check_code(lang: str, code: str, run_python: bool = False, timeout: int = 10):
    """Return (ok, error_message)."""
    try:
        if lang in ("python", "py"):
            ast.parse(code)
            if run_python:
                with tempfile.TemporaryDirectory() as d:
                    f = Path(d) / "snippet.py"
                    f.write_text(code, encoding="utf-8")
                    r = subprocess.run([sys.executable, str(f)], capture_output=True,
                                       text=True, timeout=timeout, cwd=d)
                    if r.returncode != 0:
                        return False, r.stderr[-1500:]
        elif lang == "json":
            json.loads(code)
        elif lang in ("javascript", "js") and shutil.which("node"):
            with tempfile.TemporaryDirectory() as d:
                f = Path(d) / "snippet.js"
                f.write_text(code, encoding="utf-8")
                r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
                if r.returncode != 0:
                    return False, r.stderr[-1500:]
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    except subprocess.TimeoutExpired:
        return False, "Execution timed out (possible infinite loop)."
    except Exception as e:
        return False, str(e)

class CodeBrain:
    def __init__(self, model=MODEL, max_retries=3, run_python=False):
        self.model, self.max_retries, self.run_python = model, max_retries, run_python

    def _chat(self, messages, temperature=0.2):
        r = requests.post(OLLAMA_URL, json={
            "model": self.model, "messages": messages, "stream": False,
            "options": {"temperature": temperature, "num_ctx": 4096}}, timeout=300)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def generate(self, request: str, context: str = ""):
        user = request if not context else f"{request}\n\nProject context:\n{context}"
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        last = {"ok": False}
        for attempt in range(1, self.max_retries + 1):
            reply = self._chat(messages)
            lang, code = extract_code(reply)
            ok, err = check_code(lang, code, self.run_python)
            last = {"ok": ok, "lang": lang, "code": code, "attempts": attempt,
                    "raw": reply, "error": err}
            if ok:
                return last
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content":
                          f"That code failed with:\n{err}\nFix it and return the full corrected code."}]
        return last

    def save(self, result, path):
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(result["code"], encoding="utf-8")
        return str(p)

if __name__ == "__main__":
    brain = CodeBrain(run_python=False)
    res = brain.generate("Write a Python function that merges overlapping intervals, with a main demo.")
    print(res["lang"], "| ok:", res["ok"], "| attempts:", res["attempts"])
    print(res["code"])