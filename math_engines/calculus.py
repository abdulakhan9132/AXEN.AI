from __future__ import annotations

import sympy as sp


def derivative(expression, symbol): return sp.diff(expression, symbol)
def partial_derivative(expression, symbol): return sp.diff(expression, symbol)
def integral(expression, symbol): return sp.integrate(expression, symbol)
def definite_integral(expression, symbol, start, end): return sp.integrate(expression, (symbol, start, end))
def limit(expression, symbol, point, direction="-"): return sp.limit(expression, symbol, point, dir=direction)
def taylor(expression, symbol, point=0, order=6): return sp.series(expression, symbol, point, order)
def critical_points(expression, symbol): return sp.solve(sp.diff(expression, symbol), symbol)
def differential_equation_solution(equation, function): return sp.dsolve(equation, function)
