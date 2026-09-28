"""AXEN Multi-Language Coding Intelligence Package.

Supports 37 programming languages: detection, syntax validation,
security checking, code analysis, translation, and safe execution.
"""

from .analyzer import CodeAnalyzer, analyze_code
from .coding_controller import CodingController, CodingResult, process_coding_request
from .detector import DetectionResult, LanguageDetector, detect_language
from .registry import LanguageRegistry, LanguageSpec, get_language_registry
from .runner import ExecutionResult, SafeRunner, run_code
from .security import SecurityChecker, SecurityReport, SecurityRisk, check_security
from .translator import CodeTranslator, prepare_translation
from .validator import CodeValidator, ValidationResult, validate_code

__all__ = [
    "LanguageSpec",
    "LanguageRegistry",
    "get_language_registry",
    "LanguageDetector",
    "detect_language",
    "DetectionResult",
    "SecurityChecker",
    "SecurityReport",
    "SecurityRisk",
    "check_security",
    "CodeValidator",
    "ValidationResult",
    "validate_code",
    "SafeRunner",
    "ExecutionResult",
    "run_code",
    "CodeAnalyzer",
    "analyze_code",
    "CodeTranslator",
    "prepare_translation",
    "CodingController",
    "CodingResult",
    "process_coding_request",
]
