"""
training/log_misses.py
======================
Appends a misrouted user message and its correct intent to training/intents.jsonl
so the AxenBrain model can be retrained on new edge cases.

Usage:
  python training/log_misses.py "user query here" correct_intent
  python training/log_misses.py --interactive
"""

import argparse
import json
import sys
from pathlib import Path

VALID_INTENTS = [
    "chat",
    "code",
    "math",
    "file_op",
    "web_search",
    "ui_design",
    "memory",
    "job_search",
]

DATASET_PATH = Path("training/intents.jsonl")


def log_miss(text: str, correct_intent: str, file_path: Path = DATASET_PATH) -> bool:
    text = text.strip()
    correct_intent = correct_intent.strip().lower()

    if not text:
        print("Error: Message text cannot be empty.", file=sys.stderr)
        return False

    if correct_intent not in VALID_INTENTS:
        print(
            f"Error: Invalid intent '{correct_intent}'. Must be one of: {VALID_INTENTS}",
            file=sys.stderr,
        )
        return False

    file_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"text": text, "intent": correct_intent}

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Logged miss successfully: \"{text}\" -> {correct_intent}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Log misrouted queries to training dataset.")
    parser.add_argument("text", nargs="?", default="", help="The query text that was misclassified.")
    parser.add_argument("intent", nargs="?", default="", help=f"The correct intent ({', '.join(VALID_INTENTS)}).")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive prompt mode.")

    args = parser.parse_args()

    if args.interactive:
        print("--- AxenBrain Miss Logger (Interactive) ---")
        print(f"Available intents: {', '.join(VALID_INTENTS)}")
        while True:
            try:
                t = input("\nEnter query (or Ctrl+C to exit): ").strip()
                if not t:
                    continue
                i = input(f"Enter correct intent ({', '.join(VALID_INTENTS)}): ").strip()
                log_miss(t, i)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting logger.")
                break
        return

    if not args.text or not args.intent:
        parser.print_help()
        sys.exit(1)

    success = log_miss(args.text, args.intent)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
