from __future__ import annotations

import math
from typing import Callable

import mpmath as mp


def high_precision(value, digits: int = 50):
    with mp.workdps(digits):
        return +mp.mpf(value)


def root(function: Callable[[float], float], guess: float): return mp.findroot(function, guess)
def integrate(function: Callable[[float], float], start: float, end: float): return mp.quad(function, [start, end])
def minimize(function: Callable[[float], float], start: float, end: float, steps: int = 1000):
    points = [start + (end - start) * index / steps for index in range(steps + 1)]
    return min(points, key=function)
def matrix_multiply(left, right):
    if len(left[0]) != len(right): raise ValueError("Matrix dimensions are incompatible.")
    return [[sum(left[row][inner] * right[inner][column] for inner in range(len(right))) for column in range(len(right[0]))] for row in range(len(left))]
