"""AXEN Coding Controller.

Central orchestrator for multi-language coding intelligence.
Coordinates language detection, task classification, security pre-scanning,
local LLM generation, syntax validation, and safe sandboxed execution.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from .analyzer import CodeAnalyzer
from .detector import DetectionResult, LanguageDetector
from .prompts import build_task_prompt
from .registry import LanguageRegistry, LanguageSpec, get_language_registry
from .runner import ExecutionResult, SafeRunner
from .security import SecurityChecker, SecurityRisk
from .translator import CodeTranslator
from .validator import CodeValidator, ValidationResult


@dataclass
class CodingResult:
    status: str  # "success", "security_blocked", "validation_warning", "llm_unavailable", "error"
    task_type: str
    language: Optional[str]
    confidence: float
    response: str
    code: Optional[str] = None
    validation: Optional[Dict[str, Any]] = None
    security: Optional[Dict[str, Any]] = None
    execution: Optional[Dict[str, Any]] = None
    analysis: Optional[Dict[str, Any]] = None
    timings: dict[str, float] = field(default_factory=dict)
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "status": self.status,
            "task_type": self.task_type,
            "language": self.language,
            "confidence": self.confidence,
            "response": self.response,
            "timings": self.timings,
        }
        if self.code is not None:
            res["code"] = self.code
        if self.validation is not None:
            res["validation"] = self.validation
        if self.security is not None:
            res["security"] = self.security
        if self.execution is not None:
            res["execution"] = self.execution
        if self.analysis is not None:
            res["analysis"] = self.analysis
        if self.reason is not None:
            res["reason"] = self.reason
        return res


def _extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """Extract (language_tag, code_snippet) from markdown code blocks."""
    pattern = re.compile(r"```([a-zA-Z0-9_#+]*)\n(.*?)```", re.DOTALL)
    matches = pattern.findall(text)
    return [(tag.strip(), code.strip()) for tag, code in matches]


class CodingController:
    """End-to-end coding intelligence pipeline."""

    def __init__(self, model_manager: Any = None) -> None:
        self.model_manager = model_manager
        self.registry = get_language_registry()
        self.detector = LanguageDetector()
        self.security = SecurityChecker()
        self.validator = CodeValidator()
        self.runner = SafeRunner()
        self.analyzer = CodeAnalyzer()
        self.translator = CodeTranslator()

    def classify_task(self, prompt: str) -> str:
        """Classify the software engineering task requested."""
        lowered = prompt.lower()
        if re.search(r"\b(?:test|tests|unit test|unit tests|test cases|coverage|pytest)\b", lowered):
            return "test"
        if re.search(r"\b(?:translate|convert|port)\b.*?\b(?:to|into)\b", lowered):
            return "translate"
        if re.search(r"\b(?:explain|walkthrough|what does|how does)\b", lowered):
            return "explain"
        if re.search(r"\b(?:debug|fix|find bug|error|issue|failing)\b", lowered):
            return "debug"
        if re.search(r"\b(?:review|audit|inspect|check)\b", lowered):
            return "review"
        if re.search(r"\b(?:refactor|clean up|restructure|reorganize)\b", lowered):
            return "refactor"
        if re.search(r"\b(?:optimize|speed up|accelerate|performance)\b", lowered):
            return "optimize"
        if re.search(r"\b(?:document|docstring|comments|documenting)\b", lowered):
            return "document"
        if re.search(r"\b(?:migrate|upgrade|modernize)\b", lowered):
            return "migrate"
        return "generate"

    def process(
        self,
        prompt: str,
        language: Optional[str] = None,
        file_path: Optional[str] = None,
        code: Optional[str] = None,
        target_language: Optional[str] = None,
        execute_safe: bool = False,
    ) -> CodingResult:
        t0 = time.perf_counter()
        timings: dict[str, float] = {}

        # 1. Language Detection
        detected_res: DetectionResult
        if language:
            spec = self.registry.get(language)
            detected_res = DetectionResult(spec, 1.0, f"User-specified: {language}", [spec.name] if spec else [])
        else:
            detected_res = self.detector.detect(text_or_prompt=prompt, file_path=file_path, code_content=code)

        lang_spec = detected_res.language
        lang_name = lang_spec.name if lang_spec else "Universal"
        timings["detection"] = round(time.perf_counter() - t0, 4)

        # 2. Task Classification
        task_type = self.classify_task(prompt)

        # 3. Security Pre-Scan on Input Code
        input_code = code or ""
        sec_report = self.security.inspect(input_code)
        if not sec_report.is_safe_to_execute and task_type not in ("explain", "review", "debug"):
            timings["total"] = round(time.perf_counter() - t0, 4)
            return CodingResult(
                status="security_blocked",
                task_type=task_type,
                language=lang_name,
                confidence=detected_res.confidence,
                response="Operation halted: Dangerous code patterns detected in input.",
                security=sec_report.to_dict(),
                timings=timings,
                reason="Input code contains destructive, system-tampering, or malicious commands.",
            )

        # 4. Handle pure analysis/review if no LLM needed or requested
        if task_type == "review" and input_code and (not self.model_manager or not self.model_manager.health_check()):
            analysis_rep = self.analyzer.analyze(input_code, lang_name)
            val_rep = self.validator.validate(input_code, lang_name) if lang_spec else None
            timings["total"] = round(time.perf_counter() - t0, 4)
            return CodingResult(
                status="success",
                task_type=task_type,
                language=lang_name,
                confidence=detected_res.confidence,
                response=f"Static code review completed for {lang_name}.",
                analysis=analysis_rep.to_dict(),
                validation=val_rep.to_dict() if val_rep else None,
                security=sec_report.to_dict(),
                timings=timings,
            )

        # 5. LLM Invocation
        if not self.model_manager or not self.model_manager.health_check():
            analysis_rep = self.analyzer.analyze(input_code, lang_name) if input_code else None
            val_rep = self.validator.validate(input_code, lang_name) if input_code and lang_spec else None
            timings["total"] = round(time.perf_counter() - t0, 4)
            msg = self.model_manager.unavailable_message() if self.model_manager else "Local model manager not configured."
            return CodingResult(
                status="llm_unavailable",
                task_type=task_type,
                language=lang_name,
                confidence=detected_res.confidence,
                response=f"Static analysis completed. LLM is currently unavailable: {msg}",
                analysis=analysis_rep.to_dict() if analysis_rep else None,
                validation=val_rep.to_dict() if val_rep else None,
                security=sec_report.to_dict(),
                timings=timings,
                reason=msg,
            )

        # Build prompt
        t_llm = time.perf_counter()
        if task_type == "translate" and target_language:
            t_prompt = self.translator.build_translation_prompt(lang_name, target_language, input_code)
            sys_p, usr_p = t_prompt.system_prompt, t_prompt.user_prompt
        else:
            sys_p, usr_p = build_task_prompt(task_type, lang_spec, prompt, input_code)

        try:
            model = self.model_manager.get_model()
            raw_response = model.generate(usr_p, system=sys_p)
        except Exception as exc:
            timings["total"] = round(time.perf_counter() - t0, 4)
            return CodingResult(
                status="error",
                task_type=task_type,
                language=lang_name,
                confidence=detected_res.confidence,
                response=f"Model generation error: {exc}",
                timings=timings,
                reason=str(exc),
            )
        timings["llm_generation"] = round(time.perf_counter() - t_llm, 4)

        # 6. Post-processing: extract generated code blocks
        blocks = _extract_code_blocks(raw_response)
        primary_code = blocks[0][1] if blocks else None

        # 7. Post-generation Security & Syntax Validation
        val_data = None
        sec_data = None
        exec_data = None

        code_to_check = primary_code or input_code
        if code_to_check and lang_spec:
            sec_rep = self.security.inspect(code_to_check)
            sec_data = sec_rep.to_dict()

            val_rep = self.validator.validate(code_to_check, lang_spec.slug)
            val_data = val_rep.to_dict()

            # 8. Controlled Safe Execution (if requested and code is safe)
            if execute_safe and sec_rep.is_safe_to_execute:
                t_exec = time.perf_counter()
                exec_rep = self.runner.run(code_to_check, lang_spec.slug)
                exec_data = exec_rep.to_dict()
                timings["execution"] = round(time.perf_counter() - t_exec, 4)

        # 9. Ensure analysis is available for review or code tasks
        analysis_data = None
        code_for_metrics = input_code or primary_code
        if code_for_metrics and lang_spec:
            analysis_data = self.analyzer.analyze(code_for_metrics, lang_spec.slug).to_dict()

        timings["total"] = round(time.perf_counter() - t0, 4)

        return CodingResult(
            status="success",
            task_type=task_type,
            language=lang_name,
            confidence=detected_res.confidence,
            response=raw_response,
            code=primary_code,
            validation=val_data,
            security=sec_data,
            execution=exec_data,
            analysis=analysis_data,
            timings=timings,
        )


_DEFAULT_CONTROLLER = CodingController()


def process_coding_request(prompt: str, **kwargs) -> CodingResult:
    return _DEFAULT_CONTROLLER.process(prompt, **kwargs)
