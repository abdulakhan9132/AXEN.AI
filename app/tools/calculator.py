from __future__ import annotations

import ast
import operator

from app.agent.tools import ToolDefinition


_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed.")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _ALLOWED_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        operand = _safe_eval(node.operand)
        return _ALLOWED_OPERATORS[type(node.op)](operand)
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def evaluate_expression(args: dict) -> dict:
    expr = args.get("expression", "")
    if not expr:
        return {"success": False, "error": "Missing expression"}
    try:
        tree = ast.parse(expr, mode="eval")
        value = _safe_eval(tree.body)
        return {"success": True, "result": value}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def build_calculator_tool() -> ToolDefinition:
    return ToolDefinition(
        name="calculator.evaluate",
        description="Safely evaluate arithmetic expressions.",
        input_schema={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Arithmetic to evaluate."},
            },
            "required": ["expression"],
        },
        output_schema={
            "type": "object",
            "properties": {"result": {"type": "number"}},
        },
        risk="low",
        handler=evaluate_expression,
    )
