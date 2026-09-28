from __future__ import annotations

import sympy as sp


def simplify(expression): return sp.simplify(expression)
def expand(expression): return sp.expand(expression)
def factor(expression): return sp.factor(expression)
def solve(expression, symbol=None): return sp.solve(expression, symbol)
def solveset(expression, symbol, domain=sp.S.Complexes): return sp.solveset(expression, symbol, domain=domain)
def substitute(expression, values): return sp.simplify(expression.subs(values))
def differentiate(expression, symbol): return sp.diff(expression, symbol)
def integrate(expression, symbol): return sp.integrate(expression, symbol)
def limit(expression, symbol, point): return sp.limit(expression, symbol, point)
def series(expression, symbol, point=0, order=6): return sp.series(expression, symbol, point, order)
