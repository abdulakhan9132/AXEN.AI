import sys
import json
from pathlib import Path

def main():
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
        prompt = payload.get("prompt", "")
        enhance = payload.get("enhance", True)
        preset = payload.get("preset", "turbo")
        if not prompt:
            print(json.dumps({"error": "Prompt is required"}))
            sys.exit(1)
        
        ROOT_DIR = Path(__file__).resolve().parent.parent
        if str(ROOT_DIR) not in sys.path:
            sys.path.insert(0, str(ROOT_DIR))
        
        from image_brain import ImageBrain
        brain = ImageBrain(preset=preset)
        res = brain.generate(prompt, enhance=enhance)
        paths = res.get("paths", [])
        first_path = paths[0] if paths else ""
        filename = Path(first_path).name if first_path else ""
        url = f"/images/{filename}" if filename else ""
        
        print(json.dumps({
            "prompt": res.get("prompt", prompt),
            "url": url,
            "path": first_path,
            "paths": paths
        }))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
