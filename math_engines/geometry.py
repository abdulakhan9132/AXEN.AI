from __future__ import annotations

import math


def distance(first: tuple[float, float], second: tuple[float, float]) -> float: return math.hypot(second[0] - first[0], second[1] - first[1])
def slope(first: tuple[float, float], second: tuple[float, float]) -> float:
    if second[0] == first[0]: raise ValueError("A vertical line has undefined slope.")
    return (second[1] - first[1]) / (second[0] - first[0])
def circle_area(radius: float) -> float: return math.pi * radius * radius
def triangle_area(base: float, height: float) -> float: return base * height / 2
def dot(left, right): return sum(a * b for a, b in zip(left, right))
def vector_norm(value): return math.sqrt(dot(value, value))
