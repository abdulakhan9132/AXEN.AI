from __future__ import annotations

from typing import Any

import sympy as sp


def verify_identity(left: sp.Expr, right: sp.Expr) -> tuple[bool, str]:
    difference = sp.simplify(left - right)
    return difference == 0, f"simplified difference = {difference}"


def verify_equation(expression: sp.Expr, solution: Any, symbol: sp.Symbol | None = None) -> tuple[bool, str]:
    values = solution if isinstance(solution, (list, tuple, set, sp.FiniteSet)) else [solution]
    checks = []
    for value in values:
        check = sp.simplify(expression.subs(symbol, value)) if symbol is not None else sp.simplify(expression.subs(value))
        checks.append(check)
    passed = all(check == 0 for check in checks)
    return passed, f"substitution residuals = {checks}"


def verify_modular(value: int, modulus: int, remainder: int) -> tuple[bool, str]:
    residual = (value - remainder) % modulus
    return residual == 0, f"residual modulo {modulus} = {residual}"


def verify_matrix_product(left: Any, right: Any, product: Any) -> tuple[bool, str]:
    expected = left * right
    passed = expected == product
    return passed, f"recomputed product = {expected}"


def verify_numeric(actual: Any, expected: Any, tolerance: float = 1e-9) -> tuple[bool, str]:
    difference = abs(float(actual) - float(expected))
    return difference <= tolerance, f"absolute error = {difference:.3g}"
