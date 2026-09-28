"""AXEN Syntax Validator.

Performs strict AST validation for Python and safe structural / compiler checks
for other languages when genuine compilers are installed.
Never fabricates compiler output if a compiler is missing.
"""

from __future__ import annotations

import ast
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .registry import LanguageSpec, get_language_registry


@dataclass
class ValidationResult:
    valid: bool
    method: str
    errors: list[str]
    compiler_used: Optional[str] = None
    line_number: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "method": self.method,
            "errors": self.errors,
            "compiler_used": self.compiler_used,
            "line_number": self.line_number,
        }


def _check_balanced_delimiters(code: str) -> tuple[bool, Optional[str], Optional[int]]:
    """Check balancing of parentheses, brackets, and braces while ignoring strings/comments."""
    stack: list[tuple[str, int]] = []
    pairs = {")": "(", "}": "{", "]": "["}
    openers = set(pairs.values())
    in_string: Optional[str] = None
    escape = False

    lines = code.splitlines()
    for line_idx, line in enumerate(lines, start=1):
        i = 0
        while i < len(line):
            ch = line[i]

            # String literal handling
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == in_string:
                    in_string = None
                i += 1
                continue

            if ch in ('"', "'", "`"):
                in_string = ch
                i += 1
                continue

            # Line comments
            if ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
                break
            if ch == "#":
                break

            # Delimiters
            if ch in openers:
                stack.append((ch, line_idx))
            elif ch in pairs:
                expected = pairs[ch]
                if not stack:
                    return False, f"Unmatched closing delimiter '{ch}' at line {line_idx}", line_idx
                top, opener_line = stack.pop()
                if top != expected:
                    return False, f"Mismatched closing delimiter '{ch}' at line {line_idx} (opened with '{top}' at line {opener_line})", line_idx

            i += 1

    if in_string:
        return False, f"Unterminated string literal ({in_string})", len(lines)
    if stack:
        top, line_no = stack[-1]
        return False, f"Unclosed delimiter '{top}' opened at line {line_no}", line_no

    return True, None, None


class CodeValidator:
    """Validates source code syntax across 37 supported languages."""

    def __init__(self) -> None:
        self.registry = get_language_registry()

    def validate_python(self, code: str) -> ValidationResult:
        """Strict Python syntax check via standard AST parsing."""
        import textwrap
        cleaned = textwrap.dedent(code).strip()
        try:
            ast.parse(cleaned)
            return ValidationResult(valid=True, method="python_ast", errors=[])
        except SyntaxError as exc:
            msg = f"{exc.msg} (line {exc.lineno})"
            return ValidationResult(
                valid=False,
                method="python_ast",
                errors=[msg],
                line_number=exc.lineno,
            )

    def validate_with_compiler(self, code: str, lang: LanguageSpec) -> Optional[ValidationResult]:
        """Attempt real syntax check if local compiler/linter binary exists."""
        installed, binary_path = lang.check_runtime()
        if not installed or not binary_path:
            return None

        # Safe compiler syntax-only flags without executing code
        ext = lang.extensions[0] if lang.extensions else ".txt"
        with tempfile.NamedTemporaryFile(suffix=ext, mode="w", encoding="utf-8", delete=False) as tmp:
            tmp.write(code)
            tmp_path = Path(tmp.name)

        try:
            if lang.slug in ("javascript", "typescript") and "node" in binary_path.lower():
                # Node.js check flag
                proc = subprocess.run([binary_path, "--check", str(tmp_path)], capture_output=True, text=True, timeout=5)
                if proc.returncode == 0:
                    return ValidationResult(valid=True, method="compiler_check", errors=[], compiler_used=binary_path)
                return ValidationResult(valid=False, method="compiler_check", errors=[proc.stderr.strip() or "Syntax error"], compiler_used=binary_path)

            if lang.slug == "rust" and "rustc" in binary_path.lower():
                # rustc syntax only
                proc = subprocess.run([binary_path, "--error-format=short", "-Zno-codegen", str(tmp_path)], capture_output=True, text=True, timeout=5)
                if proc.returncode == 0:
                    return ValidationResult(valid=True, method="compiler_check", errors=[], compiler_used=binary_path)
                return ValidationResult(valid=False, method="compiler_check", errors=[proc.stderr.strip()], compiler_used=binary_path)

            if lang.slug in ("c", "cpp") and ("gcc" in binary_path.lower() or "g++" in binary_path.lower() or "clang" in binary_path.lower()):
                proc = subprocess.run([binary_path, "-fsyntax-only", str(tmp_path)], capture_output=True, text=True, timeout=5)
                if proc.returncode == 0:
                    return ValidationResult(valid=True, method="compiler_check", errors=[], compiler_used=binary_path)
                return ValidationResult(valid=False, method="compiler_check", errors=[proc.stderr.strip()], compiler_used=binary_path)
        except Exception as exc:
            return None
        finally:
            if tmp_path.exists():
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        return None

    def validate(self, code: str, language_name_or_slug: str) -> ValidationResult:
        """Validate code syntax using AST, compiler, or structural heuristic."""
        spec = self.registry.get(language_name_or_slug)
        if not spec:
            return ValidationResult(valid=False, method="unknown_language", errors=[f"Unsupported language: {language_name_or_slug}"])

        # 1. Python AST
        if spec.slug == "python":
            return self.validate_python(code)

        # 2. Try genuine compiler if available
        compiler_res = self.validate_with_compiler(code, spec)
        if compiler_res:
            return compiler_res

        # 3. Structural heuristic check (balance of delimiters and basic syntax)
        ok, err, line = _check_balanced_delimiters(code)
        if not ok and err:
            return ValidationResult(
                valid=False,
                method="structural_heuristic (compiler not installed)",
                errors=[err],
                line_number=line,
            )

        return ValidationResult(
            valid=True,
            method="structural_heuristic (compiler not installed)",
            errors=[],
        )


_DEFAULT_VALIDATOR = CodeValidator()


def validate_code(code: str, language: str) -> ValidationResult:
    return _DEFAULT_VALIDATOR.validate(code, language)
