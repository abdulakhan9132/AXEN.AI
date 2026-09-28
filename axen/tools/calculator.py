"""
axen/tools/calculator.py
==========================
Safe arithmetic calculator tool for AXEN.

Uses Python's AST parser to evaluate arithmetic expressions without
calling eval() on arbitrary code. Only number literals and the
following operators are permitted: + - * / ** % (unary -)

Does NOT execute shell commands or access the filesystem.
"""

from __future__ import annotations

import ast
import operator as _op

from .tool_registry import ToolDefinition


# Whitelist of permitted AST node types → Python operators
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


def _safe_eval(node: ast.AST) -> float | int:
    """Recursively evaluate an AST expression node."""
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
        return _SAFE_OPERATORS[op_type](left, right)  # type: ignore[operator]
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {ast.dump(node.op)}")
        operand = _safe_eval(node.operand)
        return _SAFE_OPERATORS[op_type](operand)  # type: ignore[operator]
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def calculate(args: dict) -> dict:
    """
    Tool handler: evaluate an arithmetic expression.

    Args:
        args["expression"]: e.g. "12345 * 678" or "(1 + 2) ** 10"

    Returns:
        {"success": True, "result": <number>, "expression": <original>}
        or
        {"success": False, "error": <message>}
    """
    expression = str(args.get("expression", "")).strip()
    if not expression:
        return {"success": False, "error": "Missing 'expression' argument."}
    try:
        tree = ast.parse(expression, mode="eval")
        value = _safe_eval(tree.body)
        # Format result: show integer if whole number
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
    except Exception as exc:
        return {"success": False, "error": str(exc), "expression": expression}


def build_calculator_tool() -> ToolDefinition:
    """Return the AXEN calculator ToolDefinition."""
    return ToolDefinition(
        name="calculator.evaluate",
        description=(
            "Safely evaluate an arithmetic expression. "
            "Supports +, -, *, /, **, %, //. "
            "Example: '12345 * 678' → 8370110"
        ),
        handler=calculate,
        input_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression to evaluate.",
                }
            },
            "required": ["expression"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "result": {"type": "number"},
                "display": {"type": "string"},
            },
        },
        risk="low",
    )
