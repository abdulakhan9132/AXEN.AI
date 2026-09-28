"""AXEN Code Translator.

Translates code between any of the 37 supported languages while preserving:
- Computational logic & algorithmic behavior
- Edge case handling
- Idiomatic language conventions (e.g., Python lists -> Rust Vec, memory management, types)
Explains the architectural and idiomatic differences between source and target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .registry import LanguageSpec, get_language_registry


# Idiomatic translation notes between common language paradigms
_IDIOM_EXPLANATIONS: dict[tuple[str, str], list[str]] = {
    ("python", "rust"): [
        "Memory: Python's automatic garbage collection is translated into Rust's compile-time ownership, borrowing, and lifetime rules.",
        "Types: Dynamic typing is converted into strict static types with explicit structs, enums, and Option/Result error handling.",
        "Performance: Bound checking, zero-cost abstractions, and direct stack allocation are prioritized in Rust.",
    ],
    ("python", "cpp"): [
        "Memory: Garbage collection is replaced with RAII (Resource Acquisition Is Initialization) and smart pointers (std::unique_ptr, std::shared_ptr).",
        "Containers: Python dict and list are mapped to std::unordered_map and std::vector with explicit templated value types.",
        "Error Handling: Exceptions or explicit status codes replace dynamic Python exceptions.",
    ],
    ("python", "javascript"): [
        "Async: Python's asyncio / async-await translates to JavaScript Promises and async/await.",
        "Prototypes & Classes: Python class methods map to ES6 class syntax with explicit `this` binding.",
    ],
    ("javascript", "typescript"): [
        "Type Safety: Adds explicit interfaces, types, generic parameters, and compile-time type narrowing without runtime overhead.",
        "Nullability: Enforces strict null checks (null vs undefined).",
    ],
    ("java", "kotlin"): [
        "Null Safety: Eliminates NullPointerExceptions through Kotlin's non-nullable type system (String vs String?).",
        "Conciseness: Verbose POJOs with getters/setters are converted to concise data classes.",
        "Coroutines: Java threads and CompletableFuture are translated to lightweight Kotlin coroutines.",
    ],
    ("c", "cpp"): [
        "Modernization: Replaces manual malloc/free with modern C++ RAII, std::vector, and std::string.",
        "Type Safety: Uses strong typing, function overloading, and templates instead of void* pointers and macro-based generic tricks.",
    ],
}


@dataclass
class TranslationPrompt:
    source_language: str
    target_language: str
    code: str
    system_prompt: str
    user_prompt: str
    idiom_notes: list[str]


class CodeTranslator:
    """Prepares and structures cross-language translation requests."""

    def __init__(self) -> None:
        self.registry = get_language_registry()

    def get_idiom_notes(self, source_slug: str, target_slug: str) -> list[str]:
        if (source_slug, target_slug) in _IDIOM_EXPLANATIONS:
            return _IDIOM_EXPLANATIONS[(source_slug, target_slug)]

        src_spec = self.registry.get(source_slug)
        tgt_spec = self.registry.get(target_slug)
        src_name = src_spec.name if src_spec else source_slug
        tgt_name = tgt_spec.name if tgt_spec else target_slug

        return [
            f"Preserve all algorithmic complexity and edge cases during translation from {src_name} to {tgt_name}.",
            f"Adhere to idiomatic conventions, naming standards, and error-handling patterns of {tgt_name}.",
        ]

    def build_translation_prompt(self, source_lang: str, target_lang: str, code: str) -> TranslationPrompt:
        src = self.registry.get(source_lang)
        tgt = self.registry.get(target_lang)

        src_name = src.name if src else source_lang
        tgt_name = tgt.name if tgt else target_lang
        src_slug = src.slug if src else source_lang.lower()
        tgt_slug = tgt.slug if tgt else target_lang.lower()

        idiom_notes = self.get_idiom_notes(src_slug, tgt_slug)
        notes_text = "\n".join(f"- {n}" for n in idiom_notes)

        system_prompt = (
            f"You are AXEN's expert multi-language code translator. "
            f"Translate {src_name} code into idiomatic, correct, and robust {tgt_name}. "
            f"Preserve algorithms, variable intent, and error handling. "
            f"Do not perform literal word replacement. "
            f"Provide clean {tgt_name} code wrapped in ```{tgt_slug} ... ``` blocks, "
            f"followed by a concise summary of idiomatic differences."
        )

        user_prompt = (
            f"Translate the following {src_name} code into {tgt_name}.\n\n"
            f"Key translation guidelines:\n{notes_text}\n\n"
            f"Source {src_name} Code:\n```{src_slug}\n{code}\n```"
        )

        return TranslationPrompt(
            source_language=src_name,
            target_language=tgt_name,
            code=code,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            idiom_notes=idiom_notes,
        )


_DEFAULT_TRANSLATOR = CodeTranslator()


def prepare_translation(source_lang: str, target_lang: str, code: str) -> TranslationPrompt:
    return _DEFAULT_TRANSLATOR.build_translation_prompt(source_lang, target_lang, code)
