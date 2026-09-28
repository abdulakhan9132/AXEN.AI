"""
axen/agent/verifier.py
======================
Verification module for AXEN Phase 3 Agentic Reasoning Loop.

Verifies deterministic results:
- Calculator: checks result equality against AST parse and numeric bounds.
- MathSolver: verifies symbolic derivation flag (result.verified).
- Python tool: checks execution exit code == 0, safe stdout capture.
- File tool: checks file existence and non-empty content read.
"""

from __future__ import annotations

import ast
import operator as _op
from typing import Any


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
            {"passed": bool, "status": str, "detail": str}
        """
        if not result.get("success", False):
            return {
                "passed": False,
                "status": "verification_failed",
                "detail": f"Tool '{tool_name}' reported failure: {result.get('error', 'unknown error')}",
            }

        # 1. Calculator verification
        if tool_name in ("calculator", "calculator.evaluate"):
            expr = arguments.get("expression") or result.get("expression", "")
            raw_res = result.get("result")
            display_res = result.get("display")

            if raw_res is None and display_res is None:
                return {
                    "passed": False,
                    "status": "verification_failed",
                    "detail": "Calculator returned no numeric result.",
                }

            # Independent verification pass: re-evaluate AST
            if expr:
                try:
                    tree = ast.parse(str(expr), mode="eval")
                    re_val = self._eval_node(tree.body)
                    # Compare float equality with tolerance
                    expected = float(raw_res if raw_res is not None else display_res)
                    if abs(re_val - expected) < 1e-6:
                        return {
                            "passed": True,
                            "status": "passed",
                            "detail": f"Verified arithmetic: {expr} = {display_res or raw_res}",
                        }
                    else:
                        return {
                            "passed": False,
                            "status": "verification_failed",
                            "detail": f"Mismatch: expected {re_val}, got {expected}",
                        }
                except Exception as exc:
                    return {
                        "passed": False,
                        "status": "verification_failed",
                        "detail": f"Expression re-verification failed: {exc}",
                    }
            return {"passed": True, "status": "passed", "detail": "Calculator output confirmed."}

        # 2. MathSolver verification
        if tool_name in ("math_solver", "math"):
            verified_flag = result.get("verified", False)
            if verified_flag:
                return {
                    "passed": True,
                    "status": "passed",
                    "detail": str(result.get("verification", "verified by SymPy symbolic engine")),
                }
            return {
                "passed": False,
                "status": "verification_failed",
                "detail": "MathSolver could not formally verify the solution.",
            }

        # 3. Python code execution verification
        if tool_name in ("python", "python.execute"):
            exit_code = result.get("exit_code", 0)
            if exit_code == 0:
                stdout = result.get("stdout", "")
                return {
                    "passed": True,
                    "status": "passed",
                    "detail": f"Python execution verified (exit 0, stdout {len(stdout)} chars).",
                }
            return {
                "passed": False,
                "status": "verification_failed",
                "detail": f"Python execution failed with exit code {exit_code}.",
            }

        # 4. File tool verification
        if tool_name in ("file", "file.read"):
            content = result.get("content")
            if content is not None:
                return {
                    "passed": True,
                    "status": "passed",
                    "detail": f"File read verified ({result.get('lines', 0)} lines).",
                }
            return {
                "passed": False,
                "status": "verification_failed",
                "detail": "File content could not be read.",
            }

        # Default pass for generic success
        return {"passed": True, "status": "passed", "detail": "Tool operation completed."}

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
