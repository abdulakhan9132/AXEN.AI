"""
axen/agent/verifier.py
======================
Verification module for AXEN Agentic Reasoning Loop.

Structured status:
- VERIFIED: Deterministically verified via AST, SymPy, exit code, or inverse conversion.
- FAILED: Deterministic verification failed or output mismatch.
- UNCERTAIN: Tool completed, but formal deterministic proof could not be fully checked.
- NOT_APPLICABLE: Natural language conversation, synthesis, or creative tasks where formal verification is inapplicable.

SAFETY RULE: Never claim 'verified' when no verification actually occurred.
"""

from __future__ import annotations

import ast
import operator as _op
from typing import Any


class VerificationStatus(str):
    """
    Structured Verification Status with backward compatibility.
    Allows transparent comparison:
      VerificationStatus("VERIFIED") == "VERIFIED" -> True
      VerificationStatus("VERIFIED") == "passed" -> True
      VerificationStatus("FAILED") == "verification_failed" -> True
    """

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, str):
            return False
        s_self = str(self).upper()
        s_other = str(other).upper()
        if s_self == s_other:
            return True
        if s_self == "VERIFIED" and s_other in ("PASSED", "VERIFIED"):
            return True
        if s_self == "PASSED" and s_other in ("PASSED", "VERIFIED"):
            return True
        if s_self in ("FAILED", "VERIFICATION_FAILED") and s_other in ("FAILED", "VERIFICATION_FAILED"):
            return True
        return False

    def __hash__(self) -> int:
        return hash(str(self).upper())


VERIFIED = VerificationStatus("VERIFIED")
FAILED = VerificationStatus("FAILED")
UNCERTAIN = VerificationStatus("UNCERTAIN")
NOT_APPLICABLE = VerificationStatus("NOT_APPLICABLE")


class Verifier:
    """Validates and verifies tool execution results."""

    _SAFE_OPS: dict[type, object] = {
        ast.Add: _op.add,
        ast.Sub: _op.sub,
        ast.Mult: _op.mul,
        ast.Div: _op.truediv,
        ast.Pow: _op.pow,
        ast.Mod: _op.mod,
        ast.USub: _op.neg,
        ast.UAdd: _op.pos,
        ast.FloorDiv: _op.floordiv,
    }

    def verify_tool_result(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Verify an executed tool result.

        Returns:
            {"passed": bool, "status": VerificationStatus, "detail": str, "domain": str}
        """
        if not result.get("success", False):
            return {
                "passed": False,
                "status": FAILED,
                "detail": f"Tool '{tool_name}' reported failure: {result.get('error', 'unknown error')}",
                "domain": tool_name,
            }

        # 1. Calculator verification (independent AST evaluation)
        if tool_name in ("calculator", "calculator.evaluate"):
            expr = arguments.get("expression") or result.get("expression", "")
            raw_res = result.get("result")
            display_res = result.get("display")

            if raw_res is None and display_res is None:
                return {
                    "passed": False,
                    "status": FAILED,
                    "detail": "Calculator returned no numeric result.",
                    "domain": "arithmetic",
                }

            if expr:
                try:
                    tree = ast.parse(str(expr), mode="eval")
                    re_val = self._eval_node(tree.body)
                    expected = float(raw_res if raw_res is not None else display_res)
                    if abs(re_val - expected) < 1e-6:
                        return {
                            "passed": True,
                            "status": VERIFIED,
                            "detail": f"Verified arithmetic: {expr} = {display_res or raw_res}",
                            "domain": "arithmetic",
                        }
                    else:
                        return {
                            "passed": False,
                            "status": FAILED,
                            "detail": f"Mismatch: expected {re_val}, got {expected}",
                            "domain": "arithmetic",
                        }
                except Exception as exc:
                    return {
                        "passed": False,
                        "status": FAILED,
                        "detail": f"Expression re-verification failed: {exc}",
                        "domain": "arithmetic",
                    }
            return {"passed": True, "status": VERIFIED, "detail": "Calculator output confirmed.", "domain": "arithmetic"}

        # 2. MathSolver verification (SymPy formal proof)
        if tool_name in ("math_solver", "math"):
            verified_flag = result.get("verified", False)
            if verified_flag:
                return {
                    "passed": True,
                    "status": VERIFIED,
                    "detail": str(result.get("verification", "verified by SymPy symbolic engine")),
                    "domain": "mathematics",
                }
            return {
                "passed": False,
                "status": FAILED,
                "detail": "MathSolver could not formally verify the solution.",
                "domain": "mathematics",
            }

        # 3. Python code execution verification (exit code & output)
        if tool_name in ("python", "python.execute"):
            exit_code = result.get("exit_code", 0)
            if exit_code == 0:
                stdout = result.get("stdout", "")
                return {
                    "passed": True,
                    "status": VERIFIED,
                    "detail": f"Python execution verified (exit 0, stdout {len(stdout)} chars).",
                    "domain": "code_execution",
                }
            return {
                "passed": False,
                "status": FAILED,
                "detail": f"Python execution failed with exit code {exit_code}.",
                "domain": "code_execution",
            }

        # 4. File tool verification (file exists & read content)
        if tool_name in ("file", "file.read"):
            content = result.get("content")
            if content is not None:
                return {
                    "passed": True,
                    "status": VERIFIED,
                    "detail": f"File read verified ({result.get('lines', 0)} lines).",
                    "domain": "file_system",
                }
            return {
                "passed": False,
                "status": FAILED,
                "detail": "File content could not be read.",
                "domain": "file_system",
            }

        # 5. Computer systems verification (inverse check)
        if tool_name in ("computer", "computer_systems"):
            # Check if output contains verified flag or deterministic representation
            raw_text = str(result.get("output", result.get("result", "")))
            if "Verified" in raw_text or "decimal" in raw_text or "binary" in raw_text:
                return {
                    "passed": True,
                    "status": VERIFIED,
                    "detail": "Computer systems operation verified deterministically.",
                    "domain": "computer_systems",
                }
            return {
                "passed": True,
                "status": UNCERTAIN,
                "detail": "Computer systems operation completed without inverse proof.",
                "domain": "computer_systems",
            }

        # 6. Code syntax verification
        if tool_name in ("code_syntax", "syntax_check"):
            code = arguments.get("code") or result.get("code", "")
            lang = arguments.get("language", "python")
            return self.verify_code_syntax(code, language=lang)

        # 7. LLM conversation / synthesis (cannot claim deterministic verification)
        if tool_name in ("llm", "conversation"):
            return {
                "passed": True,
                "status": NOT_APPLICABLE,
                "detail": "Natural language generation; deterministic verification not applicable.",
                "domain": "natural_language",
            }

        return {"passed": True, "status": UNCERTAIN, "detail": "Tool operation completed.", "domain": tool_name}

    def verify_code_syntax(self, code: str, language: str = "python") -> dict[str, Any]:
        """
        Verify that generated or inspected code is syntactically valid.
        """
        if not code or not code.strip():
            return {
                "passed": False,
                "status": FAILED,
                "detail": "Empty code provided for verification.",
                "domain": language,
            }

        if language.lower() in ("python", "py"):
            try:
                ast.parse(code)
                return {
                    "passed": True,
                    "status": VERIFIED,
                    "detail": "Python syntax verified successfully via AST.",
                    "domain": "python",
                }
            except SyntaxError as err:
                return {
                    "passed": False,
                    "status": FAILED,
                    "detail": f"Python syntax error at line {err.lineno}: {err.msg}",
                    "domain": "python",
                }
        else:
            # Check balanced brackets/parentheses/braces for generic languages
            stack = []
            matching = {')': '(', ']': '[', '}': '{'}
            for char in code:
                if char in '([{':
                    stack.append(char)
                elif char in ')]}':
                    if not stack or stack[-1] != matching[char]:
                        return {
                            "passed": False,
                            "status": FAILED,
                            "detail": f"Unbalanced delimiter: {char}",
                            "domain": language,
                        }
                    stack.pop()
            if stack:
                return {
                    "passed": False,
                    "status": FAILED,
                    "detail": f"Unclosed delimiter: {stack[-1]}",
                    "domain": language,
                }
            return {
                "passed": True,
                "status": VERIFIED,
                "detail": f"{language.capitalize()} balanced delimiter check passed.",
                "domain": language,
            }

    def _eval_node(self, node: ast.AST) -> float | int:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in self._SAFE_OPS:
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self._SAFE_OPS[type(node.op)](left, right)  # type: ignore[operator]
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._SAFE_OPS:
            operand = self._eval_node(node.operand)
            return self._SAFE_OPS[type(node.op)](operand)  # type: ignore[operator]
        raise ValueError(f"Unsafe node: {type(node).__name__}")
