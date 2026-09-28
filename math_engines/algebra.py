from __future__ import annotations

import sympy as sp


def solve_equation(left, right, symbol): return sp.solve(sp.Eq(left, right), symbol)
def solve_inequality(expression, symbol): return sp.solve_univariate_inequality(expression, symbol)
def polynomial_roots(expression, symbol): return sp.roots(expression, symbol)
def complete_square(expression, symbol): return sp.complete_square(expression, symbol) if hasattr(sp, "complete_square") else sp.expand(expression)
