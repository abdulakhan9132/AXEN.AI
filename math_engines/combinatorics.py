from __future__ import annotations

import functools
import math


def permutations(n: int, k: int | None = None) -> int:
    return math.factorial(n) if k is None else math.factorial(n) // math.factorial(n - k)

def combinations(n: int, k: int) -> int: return math.comb(n, k)

def binomial(n: int, k: int) -> int: return math.comb(n, k)

@functools.lru_cache(maxsize=None)
def fibonacci(n: int) -> int:
    if n < 0: raise ValueError("Fibonacci index must be nonnegative.")
    a, b = 0, 1
    for _ in range(n): a, b = b, a + b
    return a


def count_paths(rows: int, columns: int) -> int: return math.comb(rows + columns - 2, rows - 1)
