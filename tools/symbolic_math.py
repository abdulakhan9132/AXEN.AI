from __future__ import annotations

import sympy as sp


def simplify(expression: sp.Expr) -> sp.Expr:
    return sp.simplify(expression)


def expand(expression: sp.Expr) -> sp.Expr:
    return sp.expand(expression)


def factor(expression: sp.Expr) -> sp.Expr:
    return sp.factor(expression)
