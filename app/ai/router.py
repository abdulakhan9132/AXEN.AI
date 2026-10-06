import os
import re
from typing import Dict, Any, Optional, Iterator

from .base_model import AIModel
from .own_brain import AxenOwnBrainProvider
from .local_model import LocalModel as OllamaCodingProvider
from app.context.models import ModelRequest

class ModelRouter(AIModel):
    """
    Routes tasks to either the AXEN OWN BRAIN (for general conversation)
    or the QWEN CODING SPECIALIST (for software engineering tasks).
    """
    
    def __init__(self, config_path: str = "configs/axen_local.yaml"):
        self.own_brain = AxenOwnBrainProvider()
        
        # Determine the coding model configuration
        coding_model = os.getenv("CODING_MODEL", "qwen2.5:1.5b")
        self.coding_specialist = OllamaCodingProvider(model_name=coding_model)
        
        # To avoid regex overhead for clearly simple things, we check common greetings/identity
        self.basic_phrases = re.compile(
            r'^(hi|hello|hey|who are you|what is ai|tell me a story|what do you remember)',
            re.IGNORECASE
        )
        
        # Coding keywords
        self.coding_keywords = re.compile(
            r'\b(python|javascript|typescript|java|c\+\+|rust|go|php|ruby|swift|'
            r'fastapi|react|vue|angular|node|express|'
            r'function|script|endpoint|api|database|sql|nosql|'
            r'code|coding|program|programming|debug|refactor|repository|architecture|'
            r'bug|error|exception|stacktrace|patch|commit|test)\b',
            re.IGNORECASE
        )
        
        self.model_name = "ModelRouter (AXEN Own Brain + Qwen Coding)"
        self.last_routed_provider = "axen_own_brain"
        self.last_routed_role = "primary_brain"

    def _is_coding_task(self, prompt: str) -> bool:
        """Heuristic to detect if a task requires the Coding Specialist."""
        # Fast exit for basic identity/general conversation
        if self.basic_phrases.search(prompt.strip()):
            return False
            
        # If it explicitly talks about coding
        if self.coding_keywords.search(prompt):
            return True
            
        # Default to own brain
        return False

    def health_check(self) -> bool:
        # The router is healthy if the own brain is healthy
        return self.own_brain.health_check()

    def generate(self, request: ModelRequest) -> str:
        # Determine intent from the most recent user message
        prompt = ""
        for msg in reversed(request.context.messages_for_model):
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
                break
                
        is_coding = self._is_coding_task(prompt)
        
        if is_coding:
            self.last_routed_provider = "qwen_ollama"
            self.last_routed_role = "coding_specialist"
            try:
                # Attempt to use the coding specialist
                if not self.coding_specialist.health_check():
                    raise RuntimeError("Ollama not running")
                return self.coding_specialist.generate(request)
            except Exception as e:
                return (
                    "Qwen coding specialist is currently unavailable. "
                    "I can still try using AXEN's own brain for a simple coding task."
                )
        else:
            self.last_routed_provider = "axen_own_brain"
            self.last_routed_role = "primary_brain"
            return self.own_brain.generate(request)

    def stream(self, request: ModelRequest) -> Iterator[str]:
        prompt = ""
        for msg in reversed(request.context.messages_for_model):
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
                break
                
        is_coding = self._is_coding_task(prompt)
        
        if is_coding:
            self.last_routed_provider = "qwen_ollama"
            self.last_routed_role = "coding_specialist"
            try:
                if not self.coding_specialist.health_check():
                    raise RuntimeError("Ollama not running")
                yield from self.coding_specialist.stream(request)
            except Exception as e:
                yield "Qwen coding specialist is currently unavailable. I can still try using AXEN's own brain for a simple coding task."
        else:
            self.last_routed_provider = "axen_own_brain"
            self.last_routed_role = "primary_brain"
            yield from self.own_brain.stream(request)

    def structured_output(self, request: ModelRequest, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        prompt = ""
        for msg in reversed(request.context.messages_for_model):
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
                break
                
        is_coding = self._is_coding_task(prompt)
        
        if is_coding:
            self.last_routed_provider = "qwen_ollama"
            self.last_routed_role = "coding_specialist"
            try:
                if not self.coding_specialist.health_check():
                    raise RuntimeError("Ollama not running")
                return self.coding_specialist.structured_output(request, schema=schema)
            except Exception as e:
                # Fallback to own_brain if available
                pass
                
        self.last_routed_provider = "axen_own_brain"
        self.last_routed_role = "primary_brain"
        return self.own_brain.structured_output(request, schema=schema)

    def __repr__(self) -> str:
        return f"ModelRouter(own_brain={self.own_brain}, coding_specialist={self.coding_specialist})"
