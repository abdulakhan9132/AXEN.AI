import json
import os
import subprocess
from typing import Any, Dict, Iterator, Optional
import requests
import time

from .base_model import AIModel
from app.context.models import ModelRequest

class AxenOwnBrainProvider(AIModel):
    """
    Connects to the native AXEN Node.js custom brain.
    If the server is not running on port 8000, it can optionally start it.
    """
    def __init__(self, base_url: str = "http://127.0.0.1:8001"):
        self.base_url = base_url.rstrip("/")
        
    def health_check(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/brain", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    def generate(self, request: ModelRequest) -> str:
        if not self.health_check():
            raise RuntimeError("AXEN Own Brain (Node.js) is not running on port 8001. Start it with `PORT=8001 node server.js`.")

        payload = {"messages": request.context.messages_for_model}

        try:
            resp = requests.post(f"{self.base_url}/api/provider/chat", json=payload, stream=True, timeout=60)
            resp.raise_for_status()
            
            final_answer = ""
            for line in resp.iter_lines():
                if line and line.startswith(b"data: "):
                    data_str = line[6:].decode("utf-8")
                    try:
                        data = json.loads(data_str)
                        if data.get("type") == "done":
                            final_answer = data.get("data", "")
                    except json.JSONDecodeError:
                        continue
            
            if not final_answer:
                # Fallback if no done event but tokens were sent
                pass
            return final_answer
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to communicate with AXEN own brain: {e}")

    def stream(self, request: ModelRequest) -> Iterator[str]:
        if not self.health_check():
            raise RuntimeError("AXEN Own Brain (Node.js) is not running.")

        payload = {"messages": request.context.messages_for_model}
        try:
            resp = requests.post(f"{self.base_url}/api/provider/chat", json=payload, stream=True, timeout=60)
            resp.raise_for_status()
            
            for line in resp.iter_lines():
                if line and line.startswith(b"data: "):
                    data_str = line[6:].decode("utf-8")
                    try:
                        data = json.loads(data_str)
                        if data.get("type") == "token":
                            yield data.get("data", "")
                    except json.JSONDecodeError:
                        continue
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to communicate with AXEN own brain: {e}")

    def structured_output(self, request: ModelRequest, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        text = self.generate(request)
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`\n ")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return {"raw": text}

    def __repr__(self) -> str:
        return f"AxenOwnBrainProvider(base_url={self.base_url!r})"
