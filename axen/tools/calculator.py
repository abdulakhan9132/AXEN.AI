"""
axen/tools/calculator.py
=========================
Safe arithmetic calculator tool for AXEN (Phase 5).

SAFETY ARCHITECTURE:
  - Uses Python's AST parser to evaluate arithmetic expressions without calling eval().
  - Only numeric literals and safe binary/unary operators are permitted:
    +, -, *, /, **, %, //
  - Enforces exponentiation limits to prevent Denial of Service via massive numbers (e.g. 9**99999999).
  - Explicitly assigned ToolPermission.READ_ONLY and ToolRiskLevel.LOW.
  - Does NOT execute shell commands or access the filesystem.
"""

from __future__ import annotations

import ast
import operator as _op
from typing import Any, Dict, Optional

from .base import (
    ToolDefinition,
    ToolExecutionContext,
    ToolPermission,
    ToolRiskLevel,
)

# Whitelist of permitted AST node types -> Python operators
_SAFE_OPERATORS: dict[type, object] = {
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

# Max allowed exponent power to prevent algorithmic complexity / CPU freeze
MAX_EXPONENT = 1000


def _safe_eval(node: ast.AST) -> float | int:
    """Recursively evaluate an AST expression node with safety limits."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Non-numeric constant: {node.value!r}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {ast.dump(node.op)}")

        left = _safe_eval(node.left)
        right = _safe_eval(node.right)

        # DoS safeguard against huge exponents
        if op_type is ast.Pow:
            if isinstance(right, (int, float)) and abs(right) > MAX_EXPONENT:
                raise OverflowError(f"Exponent {right} exceeds safe execution threshold ({MAX_EXPONENT}).")
            if isinstance(left, (int, float)) and abs(left) > 100_000 and right > 10:
                raise OverflowError("Exponentiation result exceeds safe computation bounds.")

        return _SAFE_OPERATORS[op_type](left, right)  # type: ignore[operator]

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {ast.dump(node.op)}")
        operand = _safe_eval(node.operand)
        return _SAFE_OPERATORS[op_type](operand)  # type: ignore[operator]

    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def calculate(args: Dict[str, Any], context: Optional[ToolExecutionContext] = None) -> Dict[str, Any]:
    """
    Tool handler: evaluate an arithmetic expression safely.

    Args:
        args["expression"]: e.g. "12345 * 678" or "(1 + 2) ** 10"

    Returns:
        {"success": True, "result": <number>, "display": <str>, "expression": <str>}
        or
        {"success": False, "error": <message>}
    """
    expression = str(args.get("expression") or args.get("expr") or "").strip()
    if not expression:
        return {"success": False, "error": "Missing 'expression' argument."}

    # Pre-check cancellation
    if context and context.is_cancelled():
        return {"success": False, "error": "Execution was cancelled.", "cancelled": True}

    try:
        tree = ast.parse(expression, mode="eval")
        value = _safe_eval(tree.body)

        # Format result
        if isinstance(value, float) and value.is_integer():
            display = str(int(value))
        else:
            display = str(value)

        return {
            "success": True,
            "result": value,
            "display": display,
            "expression": expression,
        }
    except ZeroDivisionError:
        return {"success": False, "error": "Division by zero.", "expression": expression}
    except OverflowError as exc:
        return {"success": False, "error": str(exc), "expression": expression}
    except Exception as exc:
        return {"success": False, "error": str(exc), "expression": expression}


def build_calculator_tool() -> ToolDefinition:
    """Return the AXEN calculator ToolDefinition."""
    return ToolDefinition(
        name="calculator.evaluate",
        description=(
            "Safely evaluate an arithmetic expression. "
            "Supports +, -, *, /, **, %, //. "
            "Example: '25 * 48' -> 1200"
        ),
        handler=calculate,
        input_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression to evaluate, e.g. '25 * 48'.",
                },
            },
            "required": ["expression"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "result": {"type": "number"},
                "display": {"type": "string"},
            },
        },
        permission=ToolPermission.READ_ONLY,
        risk_level="low",
        risk="low",
        capabilities=["calculator", "math", "arithmetic", "evaluate"],
        deterministic=True,
        requires_confirmation=False,
        supports_verification=True,
    )


def build_calculator_named_tool() -> ToolDefinition:
    """Alias for 'calculator'."""
    tool = build_calculator_tool()
    tool.name = "calculator"
    return tool
