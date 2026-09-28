"""AXEN Language Detector.

Detects programming language from:
1. Explicit request text
2. File paths and extensions
3. Code syntax patterns and keywords
4. Project directory structure and build manifests

Reports uncertainty when confidence is low instead of guessing silently.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .registry import LanguageSpec, get_language_registry


@dataclass
class DetectionResult:
    language: Optional[LanguageSpec]
    confidence: float
    evidence: str
    candidates: list[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected": self.language.name if self.language else None,
            "slug": self.language.slug if self.language else None,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence,
            "candidates": self.candidates,
        }


# Distinctive regex markers for syntax inspection
_SYNTAX_SIGNATURES: list[tuple[str, list[re.Pattern]]] = [
    ("cuda", [re.compile(r"__global__\s+void"), re.compile(r"<<<.*>>>"), re.compile(r"cudaMalloc"), re.compile(r"threadIdx\.[xyz]")]),
    ("python", [re.compile(r"^\s*def\s+[a-zA-Z_]\w*\s*\([^)]*\)\s*:", re.M), re.compile(r"^\s*import\s+\w+", re.M), re.compile(r"^\s*from\s+\w+\s+import", re.M), re.compile(r"if\s+__name__\s*==\s*['\"]__main__['\"]")]),
    ("rust", [re.compile(r"\bfn\s+[a-zA-Z_]\w*\s*\("), re.compile(r"\blet\s+mut\s+"), re.compile(r"\buse\s+std::"), re.compile(r"\bprintln!\s*\("), re.compile(r"\bimpl\b.*?\bfor\b")]),
    ("cpp", [re.compile(r"#include\s*<iostream>"), re.compile(r"#include\s*<vector>"), re.compile(r"std::cout"), re.compile(r"std::vector"), re.compile(r"template\s*<typename")]),
    ("c", [re.compile(r"#include\s*<stdio\.h>"), re.compile(r"#include\s*<stdlib\.h>"), re.compile(r"\bprintf\s*\("), re.compile(r"\bmalloc\s*\(")]),
    ("go", [re.compile(r"^package\s+\w+", re.M), re.compile(r"\bfunc\s+(?:\([^)]+\)\s*)?[a-zA-Z_]\w*\s*\("), re.compile(r"fmt\.Print(?:ln|f)?\s*\(")]),
    ("java", [re.compile(r"public\s+class\s+\w+"), re.compile(r"public\s+static\s+void\s+main"), re.compile(r"System\.out\.println\s*\(")]),
    ("csharp", [re.compile(r"using\s+System;"), re.compile(r"namespace\s+[\w\.]+"), re.compile(r"Console\.WriteLine\s*\("), re.compile(r"public\s+async\s+Task")]),
    ("kotlin", [re.compile(r"\bfun\s+[a-zA-Z_]\w*\s*\("), re.compile(r"\bval\s+\w+\s*[:=]"), re.compile(r"\bdata\s+class\s+")]),
    ("swift", [re.compile(r"\bimport\s+SwiftUI\b"), re.compile(r"\bguard\s+let\b"), re.compile(r"@State\s+private\s+var"), re.compile(r"\bfunc\s+\w+\s*\(.*?\)\s*->")]),
    ("typescript", [re.compile(r"\binterface\s+\w+\s*\{"), re.compile(r"\btype\s+\w+\s*="), re.compile(r":\s*(?:string|number|boolean|any|void)\b")]),
    ("javascript", [re.compile(r"\bconst\s+\w+\s*="), re.compile(r"\bfunction\s+[a-zA-Z_]\w*\s*\("), re.compile(r"console\.log\s*\("), re.compile(r"module\.exports\s*=")]),
    ("sql", [re.compile(r"\bSELECT\b.*?\bFROM\b", re.I | re.S), re.compile(r"\bCREATE\s+TABLE\b", re.I), re.compile(r"\bINSERT\s+INTO\b", re.I), re.compile(r"\bGROUP\s+BY\b", re.I)]),
    ("r", [re.compile(r"\b[a-zA-Z_.]+\s*<-\s*"), re.compile(r"\blibrary\s*\([a-zA-Z0-9_.]+\)"), re.compile(r"\bdata\.frame\s*\("), re.compile(r"\bggplot\s*\(")]),
    ("julia", [re.compile(r"\bfunction\s+[a-zA-Z_]\w*\s*\(.*?\).*?\bend\b", re.S), re.compile(r"\busing\s+[\w\s,]+"), re.compile(r"@time\b")]),
    ("scala", [re.compile(r"\bcase\s+class\s+\w+"), re.compile(r"\bdef\s+\w+\s*\(.*?\)\s*:\s*\w+"), re.compile(r"extends\s+App\b")]),
    ("haskell", [re.compile(r"^[a-zA-Z_]\w*\s*::\s*", re.M), re.compile(r"\bmodule\s+\w+\s+where"), re.compile(r"\bputStrLn\s+")]),
    ("ocaml", [re.compile(r"\blet\s+rec\s+\w+"), re.compile(r"\bopen\s+[A-Z]\w*"), re.compile(r";;\s*$")]),
    ("fsharp", [re.compile(r"\blet\s+rec\s+\w+"), re.compile(r"open\s+System\b"), re.compile(r"printfn\s+")]),
    ("prolog", [re.compile(r":-\s*"), re.compile(r"\bmember\s*\("), re.compile(r"\bwrite\s*\(")]),
    ("lisp", [re.compile(r"\(\s*defun\s+"), re.compile(r"\(\s*format\s+t\b"), re.compile(r"\(\s*let\s+\(")]),
    ("scheme", [re.compile(r"\(\s*define\s+\("), re.compile(r"\(\s*lambda\s+\("), re.compile(r"\(\s*display\s+")]),
    ("clojure", [re.compile(r"\(\s*defn\s+"), re.compile(r"\(\s*ns\s+"), re.compile(r"\(\s*println\s+")]),
    ("erlang", [re.compile(r"-module\s*\(\w+\)\."), re.compile(r"-export\s*\(\[\w+/\d+\]\)\.")]),
    ("elixir", [re.compile(r"\bdefmodule\s+[A-Z]\w*"), re.compile(r"\bdef\s+[a-z_]\w*"), re.compile(r"IO\.puts\s*\(")]),
    ("ruby", [re.compile(r"\bdef\s+[a-zA-Z_]\w*.*?\bend\b", re.S), re.compile(r"\battr_accessor\s+:\w+"), re.compile(r"\bputs\s+['\"]")]),
    ("php", [re.compile(r"<\?php"), re.compile(r"\$[a-zA-Z_]\w*\s*->")]),
    ("dart", [re.compile(r"\bvoid\s+main\s*\("), re.compile(r"import\s+['\"]package:flutter"), re.compile(r"Widget\s+build\s*\(")]),
    ("perl", [re.compile(r"#!/usr/bin/perl"), re.compile(r"\bmy\s+\$[a-zA-Z_]\w*"), re.compile(r"\buse\s+strict;")]),
    ("lua", [re.compile(r"\blocal\s+\w+\s*="), re.compile(r"\bfunction\s+[a-zA-Z_.:]+\s*\("), re.compile(r"print\s*\(")]),
    ("mojo", [re.compile(r"\bfn\s+[a-zA-Z_]\w*\s*\("), re.compile(r"from\s+python\s+import"), re.compile(r"\bstruct\s+\w+:")]),
    ("fortran", [re.compile(r"\bprogram\s+\w+", re.I), re.compile(r"\bimplicit\s+none\b", re.I), re.compile(r"\bend\s+program\b", re.I)]),
    ("sas", [re.compile(r"\bdata\s+\w+;\s*set\s+\w+;\s*run;", re.I | re.S), re.compile(r"\bproc\s+\w+;.*?\brun;", re.I | re.S)]),
    ("wolfram", [re.compile(r"[A-Z]\w*\[.*?\]"), re.compile(r"\bFunction\s*\[")]),
    ("bash", [re.compile(r"^#!\s*/bin/(?:bash|sh)"), re.compile(r"\becho\s+"), re.compile(r"\bif\s+\[.*?\];\s*then\b"), re.compile(r"\bchmod\s+[0-7]+")]),
    ("matlab", [re.compile(r"\bfunction\s+.*?\bend\b", re.S), re.compile(r"\bzeros\s*\("), re.compile(r"\bplot\s*\(")]),
    ("octave", [re.compile(r"%\s*GNU\s*Octave", re.I), re.compile(r"\boctave\b", re.I)]),
]

# Build file manifests
_MANIFEST_TO_LANGUAGE = {
    "cargo.toml": "rust",
    "package.json": "javascript",
    "tsconfig.json": "typescript",
    "go.mod": "go",
    "pom.xml": "java",
    "build.gradle": "java",
    "build.gradle.kts": "kotlin",
    "cmakelists.txt": "cpp",
    "requirements.txt": "python",
    "pyproject.toml": "python",
    "package.swift": "swift",
    "mix.exs": "elixir",
    "rebar.config": "erlang",
    "dune-project": "ocaml",
    "project.clj": "clojure",
    "pubspec.yaml": "dart",
    "composer.json": "php",
    "gemfile": "ruby",
}


class LanguageDetector:
    """Multi-stage language detection with confidence scoring."""

    def __init__(self) -> None:
        self.registry = get_language_registry()

    def detect_from_text_explicit(self, prompt: str) -> Optional[DetectionResult]:
        """Check if user prompt explicitly specifies a language."""
        lowered = prompt.lower()

        # Check for explicit flags like --language rust or in python
        flag_match = re.search(r"--(?:language|lang)\s+([a-zA-Z0-9#+]+)", lowered)
        if flag_match:
            spec = self.registry.get(flag_match.group(1))
            if spec:
                return DetectionResult(spec, 0.99, f"Explicit flag: --language {flag_match.group(1)}", [spec.name])

        patterns = [
            (r"\b(?:in|using|with|write|translate to|convert to|code in)\s+([a-zA-Z0-9#+]+)(?:\s|$|[.,?!;])", 0.95),
            (r"\b([a-zA-Z0-9#+]+)\s+(?:code|program|function|script|query|algorithm)\b", 0.90),
        ]

        # Prioritize longer language names/aliases first (e.g. c++ before c)
        for pat, conf in patterns:
            for match in re.finditer(pat, lowered):
                word = match.group(1).rstrip(".,?!;")
                spec = self.registry.get(word)
                if spec:
                    return DetectionResult(spec, conf, f"Explicit prompt match: {spec.name}", [spec.name])

        return None

    def detect_from_file_path(self, path: str | Path) -> Optional[DetectionResult]:
        """Detect language from file path or manifest name."""
        p = Path(path)
        base = p.name.lower()

        # Manifest check
        if base in _MANIFEST_TO_LANGUAGE:
            spec = self.registry.get(_MANIFEST_TO_LANGUAGE[base])
            if spec:
                return DetectionResult(spec, 0.92, f"Build manifest: {base}", [spec.name])

        # Extension check
        ext = p.suffix.lower()
        if ext:
            candidates = self.registry.get_by_extension(ext)
            if len(candidates) == 1:
                return DetectionResult(candidates[0], 0.90, f"File extension: {ext}", [candidates[0].name])
            elif len(candidates) > 1:
                names = [c.name for c in candidates]
                return DetectionResult(candidates[0], 0.70, f"Shared extension: {ext} (candidates: {', '.join(names)})", names)

        return None

    def detect_from_code_snippet(self, code: str) -> Optional[DetectionResult]:
        """Detect language using syntax heuristics and token matching."""
        if not code or not code.strip():
            return None

        scores: dict[str, int] = {}
        for slug, patterns in _SYNTAX_SIGNATURES:
            count = 0
            for pat in patterns:
                if pat.search(code):
                    count += 1
            if count > 0:
                scores[slug] = count

        if not scores:
            return None

        # Sort by match count
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_slug, top_count = sorted_scores[0]
        top_spec = self.registry.get(top_slug)

        if not top_spec:
            return None

        # Calculate confidence
        if top_count >= 3:
            conf = 0.88
        elif top_count == 2:
            conf = 0.75
        else:
            conf = 0.55

        candidates = [self.registry.get(s).name for s, _ in sorted_scores if self.registry.get(s)]
        return DetectionResult(top_spec, conf, f"Syntax signatures ({top_count} pattern matches)", candidates)

    def detect(
        self,
        text_or_prompt: Optional[str] = None,
        file_path: Optional[str | Path] = None,
        code_content: Optional[str] = None,
    ) -> DetectionResult:
        """Combined detection across all available signals."""
        # 1. Explicit request takes highest priority
        if text_or_prompt:
            res = self.detect_from_text_explicit(text_or_prompt)
            if res and res.confidence >= 0.85:
                return res

            # Check if text_or_prompt is a bare filename/path
            clean_str = text_or_prompt.strip()
            if "." in clean_str and " " not in clean_str:
                res_path = self.detect_from_file_path(clean_str)
                if res_path and res_path.confidence >= 0.60:
                    return res_path

        # 2. File path / extension
        if file_path:
            res = self.detect_from_file_path(file_path)
            if res and res.confidence >= 0.60:
                return res

        # 3. Code content syntax
        content = code_content or text_or_prompt or ""
        if content:
            res = self.detect_from_code_snippet(content)
            if res:
                return res

        # 4. Low confidence / uncertain
        return DetectionResult(
            language=None,
            confidence=0.0,
            evidence="No definitive language signals found in prompt or content",
            candidates=[],
        )


_DEFAULT_DETECTOR = LanguageDetector()


def detect_language(
    prompt: Optional[str] = None,
    file_path: Optional[str | Path] = None,
    code: Optional[str] = None,
) -> DetectionResult:
    return _DEFAULT_DETECTOR.detect(prompt, file_path, code)
