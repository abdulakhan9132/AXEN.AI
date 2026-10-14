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
    
    def __init__(self, config_path: str = "configs/axen_local.yaml", provider_resolver=None):
        self.own_brain = AxenOwnBrainProvider()
        
        # Determine the coding model configuration
        coding_model = os.getenv("CODING_MODEL", "qwen2.5:1.5b")
        self.coding_specialist = OllamaCodingProvider(model_name=coding_model)
        self.conversational_model = OllamaCodingProvider(model_name=os.getenv("CONVERSATIONAL_MODEL", "qwen2.5:1.5b"))
        reasoning_name = os.getenv("REASONING_MODEL")
        self.reasoning_model = OllamaCodingProvider(model_name=reasoning_name) if reasoning_name else self.conversational_model
        self._provider_resolver = provider_resolver
        self._native = None
        
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
        return self.conversational_model.health_check() or self.own_brain.health_check()

    @staticmethod
    def task_kind(prompt: str) -> str:
        text = prompt.lower()
        if re.search(r"\b(generate|create|draw|make)\b.*\b(image|picture|illustration|artwork)\b", text):
            return "image"
        if re.search(r"\b(pdf|document|attachment|spreadsheet)\b", text):
            return "document"
        if re.search(r"\b(shakespeare|sonnet|iambic|literary continuation)\b", text):
            return "literary"
        if re.search(r"\b(code|coding|program|programming|script|function|automation|python|javascript|typescript|html|css|java|php|sql|bash|swift|kotlin|rust|ruby|dart|flutter|scala|lua|perl|matlab|haskell|elixir|powershell|assembly|cobol|fortran|prolog|lisp|verilog|vhdl|solidity|ocaml|erlang|react|debug|refactor|repository|website|bug|files?|tests?)\b|\bc(?:\+\+|#)(?!\w)|\bobjective-c\b|\blanding\s+page\b", text):
            return "coding"
        if re.search(r"\b(code|program|script|example|function)\b.*\b(c|go|r)\b|\b(c|go|r)\s+(program|script|example|function)\b", text):
            return "coding"
        if re.search(r"\b(build|create|generate|make|develop)\b.*\b(app|application|webpage|dashboard|portfolio)\b", text):
            return "coding"
        if ModelRouter.requires_workspace(prompt):
            return "coding"
        if re.search(r"\b(calculate|derivative|integral|equation|mathematics)\b|\d\s*[+*/]\s*\d", text):
            return "math"
        return "conversation"

    @staticmethod
    def requires_workspace(prompt: str) -> bool:
        """Examples are chat outputs; explicit project work uses reviewed edits."""
        return bool(re.search(r"\b(my|our|existing|current|selected|this|the)\s+(?:\w+\s+){0,2}(project|repository|repo|workspace|codebase|files?)\b|\b(in|inside)\s+(?:the\s+)?(workspace|project|repository)\b|\b(run|execute)\b.*\btests?\b|\b(fix|debug|refactor|modify|update)\b.*\b(my|our|existing|current|selected)\s+(?:\w+\s+){0,2}function\b", prompt, re.I))

    def _select_provider(self, request):
        prompt = next((m.get("content", "") for m in reversed(request.context.messages_for_model)
                       if m.get("role") == "user"), "")
        kind = 'coding' if request.generation_settings.get('task_type') == 'coding' else self.task_kind(prompt)
        if kind == "literary":
            if self._native is None:
                if self._provider_resolver:
                    self._native = self._provider_resolver("aexn_gpt_1.0")
                else:
                    from .aexn_gpt_provider import AEXNGPTProvider
                    self._native = AEXNGPTProvider()
            candidates = [(self._native, "aexn_gpt_1.0", "literary_model")]
        elif self._is_coding_task(prompt) or kind == "coding":
            candidates = [(self.coding_specialist, "qwen_ollama", "coding_specialist"),
                          (self.reasoning_model, "local_reasoning", "coding_fallback")]
        else:
            difficult = bool(re.search(r"\b(prove|compare|reason|analyze|explain|complex)\b", prompt, re.I))
            candidates = [(self.reasoning_model if difficult else self.conversational_model,
                           "local_reasoning" if difficult else "local_conversation", "primary_brain")]
            if re.fullmatch(r"\s*(hi|hello|hey)(?:\s+aexn)?[!.?]*\s*", prompt, re.I):
                candidates.append((self.own_brain, "axen_own_brain", "primary_brain"))
        for provider, name, role in candidates:
            if provider.health_check():
                self.last_routed_provider, self.last_routed_role = name, role
                return provider
        # Cloud requires consent on THIS request, even when a key is configured.
        if request.session_metadata.get("allow_cloud") is True and self._provider_resolver:
            cloud = self._provider_resolver("cloud_openrouter")
            if cloud.health_check():
                self.last_routed_provider = "cloud_openrouter"
                return cloud
        raise RuntimeError("AEXN AI has no available local model for this task. Start the configured local model runtime. No cloud request was sent.")

    def generate(self, request: ModelRequest) -> str:
        return self._select_provider(request).generate(request)

    def stream(self, request: ModelRequest) -> Iterator[str]:
        yield from self._select_provider(request).stream(request)

    def structured_output(self, request: ModelRequest, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._select_provider(request).structured_output(request, schema=schema)

    def __repr__(self) -> str:
        return "ModelRouter(AEXN AI, local-first)"
