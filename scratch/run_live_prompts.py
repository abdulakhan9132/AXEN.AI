"""
scratch/run_live_prompts.py
Tests the 4 live queries through AxenBackendService.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
from app.ui.backend_service import AxenBackendService

def main():
    service = AxenBackendService.get_instance()
    
    # Load model if not already loaded
    if not service.controller.model_ready:
        print("[INFO] Loading model into memory...")
        service.controller.load_model()
        print("[INFO] Model loaded successfully.")
    
    test_prompts = [
        "hi",
        "Hello, who are you?",
        "How are you?",
        "What can you do?",
    ]
    
    # Ensure fresh session
    sid = service.new_session(title="Live Test")
    print(f"[INFO] Created session {sid}")
    
    for prompt in test_prompts:
        print("=" * 60)
        print(f"USER: {prompt}")
        print("-" * 60)
        
        result = service.chat(prompt, session_id=sid)
        resp = result.get("response", "")
        task_type = result.get("task_type", "")
        tool_used = result.get("tool_used", False)
        
        print(f"AXEN: {resp}")
        print(f"[Metadata: task_type={task_type}, tool_used={tool_used}]")
        print("=" * 60)
        print()

if __name__ == "__main__":
    main()
