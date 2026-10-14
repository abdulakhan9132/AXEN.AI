"""Run one actual bounded cycle. Offline is explicit; network is opt-in."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from axen.evolution.learning_store import LearningStore
from axen.evolution.learning_worker import AutonomousLearningWorker, load_config
from axen.memory.memory_store import SQLiteMemoryStore


def main():
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--offline", action="store_true")
    choice.add_argument("--allow-network", action="store_true")
    parser.add_argument("--output", default="outputs/learning-demo")
    parser.add_argument("--memory-busy-percent", type=int, default=None,
                        help="Explicit resource ceiling override for this lightweight demo only")
    parser.add_argument("--minimum-available-mb", type=int, default=None)
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    config = load_config()
    if args.memory_busy_percent is not None:
        if not 1 <= args.memory_busy_percent <= 99: parser.error("Memory ceiling must be 1..99")
        config["memory_busy_percent"] = args.memory_busy_percent
    if args.minimum_available_mb is not None:
        if args.minimum_available_mb < 64: parser.error("Demo requires at least 64 MB available RAM")
        config["minimum_available_mb"] = args.minimum_available_mb
    worker = AutonomousLearningWorker(LearningStore(output / "queue.db"), config,
                                      SQLiteMemoryStore(output / "memory.db"))
    worker.command("start", "LEARNING", offline_demo=args.offline)
    try:
        deadline = time.monotonic() + 90
        while worker.thread.is_alive() and time.monotonic() < deadline:
            worker.thread.join(timeout=0.2)
        snapshot = worker.snapshot()
    finally: worker.shutdown()
    report = {"title": "AXEN Learns Python Debugging", "offline": args.offline,
              "resource_policy": {k: config[k] for k in ("memory_busy_percent", "minimum_available_mb")},
              "snapshot": snapshot, "events": worker.store.events(0, 200)}
    path = output / "report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    completed = [g for g in snapshot["goals"] if g["status"] == "completed"]
    print(json.dumps({"report": str(path.resolve()), "completed_goals": len(completed),
                      "verified_lessons": len(snapshot["lessons"]), "status": snapshot["status"]}))
    return 0 if completed else 1


if __name__ == "__main__": raise SystemExit(main())
