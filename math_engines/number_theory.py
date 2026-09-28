from __future__ import annotations

import math


def gcd(*values: int) -> int:
    result = 0
    for value in values: result = math.gcd(result, value)
    return abs(result)


def extended_gcd(a: int, b: int):
    if b == 0: return (abs(a), 1 if a >= 0 else -1, 0)
    divisor, x1, y1 = extended_gcd(b, a % b)
    return divisor, y1, x1 - (a // b) * y1


def modular_inverse(value: int, modulus: int) -> int:
    divisor, coefficient, _ = extended_gcd(value, modulus)
    if divisor != 1: raise ValueError("A modular inverse does not exist.")
    return coefficient % modulus


def is_prime(value: int) -> bool:
    if value < 2: return False
    if value % 2 == 0: return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0: return False
        divisor += 2
    return True


def factorize(value: int) -> dict[int, int]:
    if value == 0: raise ValueError("Zero does not have a finite prime factorization.")
    value = abs(value); factors = {}; divisor = 2
    while divisor * divisor <= value:
        while value % divisor == 0: factors[divisor] = factors.get(divisor, 0) + 1; value //= divisor
        divisor = 3 if divisor == 2 else divisor + 2
    if value > 1: factors[value] = factors.get(value, 0) + 1
    return factors


def euler_phi(value: int) -> int:
    if value < 1: raise ValueError("Euler phi requires a positive integer.")
    result = value
    for prime in factorize(value): result -= result // prime
    return result


def chinese_remainder(residues: list[int], moduli: list[int]) -> int:
    if len(residues) != len(moduli) or not moduli: raise ValueError("Residues and moduli must have equal nonzero length.")
    total = 0; product = math.prod(moduli)
    for residue, modulus in zip(residues, moduli):
        partial = product // modulus
        total += residue * partial * modular_inverse(partial, modulus)
    return total % product


def modular_power(base: int, exponent: int, modulus: int) -> int: return pow(base, exponent, modulus)
