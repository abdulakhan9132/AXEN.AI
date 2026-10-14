from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.memory import MemoryStore
from app.agent.orchestrator import AgentOrchestrator
from app.agent.task_router import TaskRouter
from app.ai.model_manager import ModelManager
from app.ai.ollama_doctor import inspect_ollama
from app.tools.registry import build_registry
from reasoning.coding import (
    CodingController,
    LanguageDetector,
    get_language_registry,
)
from reasoning.math.benchmark import run_benchmark
from reasoning.math.advanced_benchmark import run_advanced_benchmark
from reasoning.math.reasoning_controller import ReasoningController
from reasoning.math.solver import solve as solve_math
from reasoning.math.strategy_validator import supported_strategies


def run_goal(goal: str) -> dict:
    registry = build_registry()
    orchestrator = AgentOrchestrator(model_manager=ModelManager(), tool_registry=registry)
    return orchestrator.run(goal)


def run_demo_tasks() -> list[dict]:
    tasks = [
        "Create a folder called test_agent and put a text file inside it.",
        "Find all PDF files in the project.",
        "Explain the structure of this project.",
        "Create a simple Python calculator.",
        "Find an error in the calculator and fix it.",
    ]
    results = []
    for index, task in enumerate(tasks, start=1):
        print(f"\n=== DEMO TASK {index} ===")
        result = run_goal(task)
        print(json.dumps(result, indent=2))
        results.append(result)
    return results


def show_status() -> dict:
    model_manager = ModelManager()
    memory_store = MemoryStore()
    registry = build_registry()
    return {
        "status": "ready",
        "model_available": model_manager.health_check(),
        "model_message": model_manager.unavailable_message() if not model_manager.health_check() else "Local model available",
        "tools": [tool.name for tool in registry.list()],
        "memory_entries": len(memory_store.list_recent(limit=20)),
    }


def show_tools() -> list[dict]:
    return [{"name": tool.name, "risk": tool.risk, "requires_approval": tool.requires_approval} for tool in build_registry().list()]


def show_memory() -> list[dict]:
    return MemoryStore().list_recent(limit=20)


def run_math(problem: str) -> int:
    result = solve_math(problem)
    print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0 if result.verified else 1


def run_math_benchmark() -> int:
    report = run_benchmark()
    summary = {key: value for key, value in report.items() if key != "records"}
    print(json.dumps(summary, indent=2, default=str))
    return 0 if report["failed"] == 0 and report["verified"] == report["problems"] else 1


def run_math_ai(problem: str) -> int:
    manager = ModelManager()
    result = ReasoningController(
        max_strategy_attempts=manager.config.max_strategy_attempts,
        max_formalization_attempts=manager.config.max_formalization_attempts,
    ).solve(problem, manager)
    print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0 if result.status == "solved" else 1


def run_math_ai_benchmark() -> int:
    report = run_advanced_benchmark()
    summary = {key: value for key, value in report.items() if key != "records"}
    print(json.dumps(summary, indent=2, default=str))
    return 0 if report["failed"] == 0 and report["verified"] == report["total"] else 1


def run_math_strategies() -> int:
    print(json.dumps({"strategies": supported_strategies()}, indent=2))
    return 0


def run_code_command(query: str, language: str | None = None, execute: bool = False) -> int:
    manager = ModelManager()
    controller = CodingController(model_manager=manager)
    res = controller.process(query, language=language, execute_safe=execute)
    print(json.dumps(res.to_dict(), indent=2, default=str))
    return 0 if res.status in ("success", "llm_unavailable") else 1


def run_code_detect(target: str) -> int:
    path = Path(target)
    detector = LanguageDetector()
    if path.exists() and path.is_file():
        code = path.read_text(encoding="utf-8", errors="ignore")
        res = detector.detect(file_path=path, code_content=code)
    else:
        clean_target = target.strip()
        if "." in clean_target and " " not in clean_target:
            res = detector.detect(file_path=clean_target)
        else:
            res = detector.detect(text_or_prompt=target)
    print(json.dumps(res.to_dict(), indent=2, default=str))
    return 0


def run_code_review(target: str) -> int:
    path = Path(target)
    code = ""
    lang = None
    if path.exists() and path.is_file():
        code = path.read_text(encoding="utf-8", errors="ignore")
        det = LanguageDetector().detect(file_path=path, code_content=code)
        lang = det.language.slug if det.language else None
    else:
        code = target
    manager = ModelManager()
    controller = CodingController(model_manager=manager)
    res = controller.process("Review this code for quality and security", language=lang, code=code)
    print(json.dumps(res.to_dict(), indent=2, default=str))
    return 0


def run_code_test(target: str) -> int:
    path = Path(target)
    code = ""
    lang = None
    if path.exists() and path.is_file():
        code = path.read_text(encoding="utf-8", errors="ignore")
        det = LanguageDetector().detect(file_path=path, code_content=code)
        lang = det.language.slug if det.language else None
    else:
        code = target
    manager = ModelManager()
    controller = CodingController(model_manager=manager)
    res = controller.process("Generate comprehensive unit tests for this code", language=lang, code=code)
    print(json.dumps(res.to_dict(), indent=2, default=str))
    return 0


def run_code_translate(target: str, from_lang: str | None = None, to_lang: str | None = None) -> int:
    path = Path(target)
    code = ""
    if path.exists() and path.is_file():
        code = path.read_text(encoding="utf-8", errors="ignore")
        if not from_lang:
            det = LanguageDetector().detect(file_path=path, code_content=code)
            from_lang = det.language.slug if det.language else None
    else:
        code = target
    if not to_lang:
        raise SystemExit("The code-translate command requires --to <target_language>.")
    manager = ModelManager()
    controller = CodingController(model_manager=manager)
    res = controller.process(
        f"Translate from {from_lang or 'auto'} to {to_lang}",
        language=from_lang,
        code=code,
        target_language=to_lang,
    )
    print(json.dumps(res.to_dict(), indent=2, default=str))
    return 0


def run_code_languages() -> int:
    registry = get_language_registry()
    langs = [l.to_dict() for l in registry.list_all()]
    print(json.dumps({"total": len(langs), "languages": langs}, indent=2, default=str))
    return 0


def run_doctor() -> int:
    report = inspect_ollama()
    values = report.to_dict()
    for key in ("OLLAMA", "MODEL", "CONNECTION", "STRUCTURED OUTPUT"):
        print(f"{key}: {values[key]}")
    print(json.dumps(values, indent=2, default=str))
    return 0 if report.ollama_installed and report.service_running and report.model_available and report.model_responding else 1


def run_tests() -> int:
    result = subprocess.run([
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-v",
    ], cwd=str(ROOT))
    return result.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local AI agent CLI")
    parser.add_argument(
        "command",
        choices=[
            "chat", "task", "status", "tools", "memory", "test", "serve", "doctor",
            "math", "math-ai", "math-benchmark", "math-ai-benchmark", "math-strategies",
            "code", "code-detect", "code-review", "code-test", "code-translate", "code-languages",
        ],
        nargs="?",
        default="status",
    )
    parser.add_argument("query", nargs="?", help="Input query, math problem, code, or file path")
    parser.add_argument("--goal", type=str, help="Goal text for a single chat task")
    parser.add_argument("--language", "-l", type=str, help="Programming language for coding commands")
    parser.add_argument("--from", dest="from_lang", type=str, help="Source language for translation")
    parser.add_argument("--to", dest="to_lang", type=str, help="Target language for translation")
    parser.add_argument("--run", action="store_true", help="Execute generated code safely if runtime is available")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "chat":
        goal = args.goal or input("Enter a goal: ").strip()
        result = TaskRouter().route(goal)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "task":
        goal = args.goal or "Create a simple Python calculator."
        result = run_goal(goal)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "status":
        print(json.dumps(show_status(), indent=2))
        return 0
    if args.command == "tools":
        print(json.dumps(show_tools(), indent=2))
        return 0
    if args.command == "memory":
        print(json.dumps(show_memory(), indent=2))
        return 0
    if args.command == "doctor":
        return run_doctor()
    if args.command == "test":
        return run_tests()
    if args.command == "serve":
        from run_ui import main as run_web_ui
        run_web_ui()
        return 0
    if args.command == "math":
        problem = args.query or args.goal
        if not problem:
            raise SystemExit("The math command requires a problem string.")
        return run_math(problem)
    if args.command == "math-benchmark":
        return run_math_benchmark()
    if args.command == "math-ai":
        problem = args.query or args.goal
        if not problem:
            raise SystemExit("The math-ai command requires a problem string.")
        return run_math_ai(problem)
    if args.command == "math-ai-benchmark":
        return run_math_ai_benchmark()
    if args.command == "math-strategies":
        return run_math_strategies()
    if args.command == "code":
        query = args.query or args.goal
        if not query:
            raise SystemExit("The code command requires a coding prompt.")
        return run_code_command(query, language=args.language, execute=args.run)
    if args.command == "code-detect":
        target = args.query or args.goal
        if not target:
            raise SystemExit("The code-detect command requires a file path or code snippet.")
        return run_code_detect(target)
    if args.command == "code-review":
        target = args.query or args.goal
        if not target:
            raise SystemExit("The code-review command requires a file path or code snippet.")
        return run_code_review(target)
    if args.command == "code-test":
        target = args.query or args.goal
        if not target:
            raise SystemExit("The code-test command requires a file path or code snippet.")
        return run_code_test(target)
    if args.command == "code-translate":
        target = args.query or args.goal
        if not target:
            raise SystemExit("The code-translate command requires a file path or code snippet.")
        return run_code_translate(target, from_lang=args.from_lang, to_lang=args.to_lang)
    if args.command == "code-languages":
        return run_code_languages()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
