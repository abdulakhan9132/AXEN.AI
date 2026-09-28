from __future__ import annotations

import ast

import sympy as sp


def matrix(values): return sp.Matrix(values)
def multiply(left, right): return sp.Matrix(left) * sp.Matrix(right)
def determinant(values): return sp.Matrix(values).det()
def inverse(values): return sp.Matrix(values).inv()
def rank(values): return sp.Matrix(values).rank()
def eigenvalues(values): return sp.Matrix(values).eigenvals()
def eigenvectors(values): return sp.Matrix(values).eigenvects()
def solve_system(matrix_values, vector_values): return sp.Matrix(matrix_values).LUsolve(sp.Matrix(vector_values))


def parse_matrix(text: str):
    start = text.find("[")
    if start < 0: raise ValueError("Matrix literal is required.")
    literal = text[start:]
    end = literal.rfind("]") + 1
    value = ast.literal_eval(literal[:end])
    if not isinstance(value, list) or not value or not all(isinstance(row, list) for row in value): raise ValueError("Invalid matrix literal.")
    return sp.Matrix(value)
