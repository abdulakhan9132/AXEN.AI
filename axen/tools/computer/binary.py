"""
axen/tools/computer/binary.py
==============================
Deterministic binary conversion and arithmetic for AXEN.

Supports:
  - Decimal ↔ Binary
  - Binary ↔ Hex
  - Binary ↔ Octal
  - Binary addition, subtraction, multiplication, division
  - Step-by-step binary addition explanation
  - Input validation

All operations use Python's built-in integer arithmetic — zero reliance
on TinyLlama for numeric conversion.
"""

from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _is_valid_binary(s: str) -> bool:
    """Return True iff *s* contains only '0' and '1' characters."""
    return bool(s) and all(c in "01" for c in s)


def _clean_binary(s: str) -> str:
    """Strip whitespace and optional '0b' prefix."""
    s = s.strip()
    if s.startswith("0b") or s.startswith("0B"):
        s = s[2:]
    return s


# ---------------------------------------------------------------------------
# BinaryConverter
# ---------------------------------------------------------------------------

class BinaryConverter:
    """
    Deterministic binary conversion and arithmetic engine.

    Usage::
        bc = BinaryConverter()
        bc.decimal_to_binary(42)        → "101010"
        bc.binary_to_decimal("101010")  → 42
        bc.binary_add("1010", "0110")   → {"result": "10000", ...}
    """

    # -----------------------------------------------------------------------
    # Conversions
    # -----------------------------------------------------------------------

    def decimal_to_binary(self, value: int) -> str:
        """Convert a non-negative integer to its binary string representation."""
        if not isinstance(value, int):
            raise TypeError(f"Expected int, got {type(value).__name__}")
        if value < 0:
            raise ValueError("Use IntegerRepresentation for negative numbers (two's complement).")
        return bin(value)[2:]  # Remove '0b' prefix

    def binary_to_decimal(self, binary_str: str) -> int:
        """Convert a binary string to decimal integer."""
        s = _clean_binary(binary_str)
        if not _is_valid_binary(s):
            raise ValueError(f"Invalid binary string: '{binary_str}'. Must contain only 0s and 1s.")
        return int(s, 2)

    def binary_to_hex(self, binary_str: str) -> str:
        """Convert a binary string to hexadecimal (uppercase, no prefix)."""
        s = _clean_binary(binary_str)
        if not _is_valid_binary(s):
            raise ValueError(f"Invalid binary string: '{binary_str}'")
        return hex(int(s, 2))[2:].upper()

    def hex_to_binary(self, hex_str: str) -> str:
        """Convert a hexadecimal string to binary (no '0b' prefix)."""
        h = hex_str.strip().lstrip("0x").lstrip("0X")
        if not h:
            return "0"
        try:
            return bin(int(hex_str.strip(), 16))[2:]
        except ValueError:
            raise ValueError(f"Invalid hexadecimal string: '{hex_str}'")

    def binary_to_octal(self, binary_str: str) -> str:
        """Convert a binary string to octal."""
        s = _clean_binary(binary_str)
        if not _is_valid_binary(s):
            raise ValueError(f"Invalid binary string: '{binary_str}'")
        return oct(int(s, 2))[2:]  # Remove '0o' prefix

    def octal_to_binary(self, octal_str: str) -> str:
        """Convert an octal string to binary."""
        o = octal_str.strip().lstrip("0o").lstrip("0O")
        try:
            return bin(int(octal_str.strip(), 8))[2:]
        except ValueError:
            raise ValueError(f"Invalid octal string: '{octal_str}'")

    # -----------------------------------------------------------------------
    # Binary Arithmetic
    # -----------------------------------------------------------------------

    def binary_add(self, a: str, b: str, *, show_steps: bool = False) -> dict:
        """
        Add two binary numbers.

        Returns a dict with keys: a, b, result, decimal_a, decimal_b,
        decimal_result, and optionally steps (list[str]).
        """
        a = _clean_binary(a)
        b = _clean_binary(b)
        for name, val in (("a", a), ("b", b)):
            if not _is_valid_binary(val):
                raise ValueError(f"Invalid binary operand '{name}': '{val}'")

        da, db = int(a, 2), int(b, 2)
        dr = da + db
        result = bin(dr)[2:]

        out: dict = {
            "operation": "addition",
            "a": a,
            "b": b,
            "result": result,
            "decimal_a": da,
            "decimal_b": db,
            "decimal_result": dr,
        }

        if show_steps:
            out["steps"] = self._addition_steps(a, b, result)

        return out

    def binary_subtract(self, a: str, b: str) -> dict:
        """Subtract binary b from binary a. Supports results ≥ 0."""
        a = _clean_binary(a)
        b = _clean_binary(b)
        for name, val in (("a", a), ("b", b)):
            if not _is_valid_binary(val):
                raise ValueError(f"Invalid binary operand '{name}': '{val}'")

        da, db = int(a, 2), int(b, 2)
        dr = da - db
        result = bin(abs(dr))[2:] if dr < 0 else bin(dr)[2:]
        negative = dr < 0

        return {
            "operation": "subtraction",
            "a": a,
            "b": b,
            "result": ("-" if negative else "") + result,
            "decimal_a": da,
            "decimal_b": db,
            "decimal_result": dr,
            "is_negative": negative,
        }

    def binary_multiply(self, a: str, b: str) -> dict:
        """Multiply two binary numbers."""
        a = _clean_binary(a)
        b = _clean_binary(b)
        for name, val in (("a", a), ("b", b)):
            if not _is_valid_binary(val):
                raise ValueError(f"Invalid binary operand '{name}': '{val}'")

        da, db = int(a, 2), int(b, 2)
        dr = da * db
        result = bin(dr)[2:]

        return {
            "operation": "multiplication",
            "a": a,
            "b": b,
            "result": result,
            "decimal_a": da,
            "decimal_b": db,
            "decimal_result": dr,
        }

    def binary_divide(self, a: str, b: str) -> dict:
        """Integer divide binary a by binary b (floor division)."""
        a = _clean_binary(a)
        b = _clean_binary(b)
        for name, val in (("a", a), ("b", b)):
            if not _is_valid_binary(val):
                raise ValueError(f"Invalid binary operand '{name}': '{val}'")

        da, db = int(a, 2), int(b, 2)
        if db == 0:
            raise ZeroDivisionError("Binary division by zero.")

        quotient = da // db
        remainder = da % db

        return {
            "operation": "division",
            "a": a,
            "b": b,
            "quotient": bin(quotient)[2:],
            "remainder": bin(remainder)[2:],
            "decimal_a": da,
            "decimal_b": db,
            "decimal_quotient": quotient,
            "decimal_remainder": remainder,
        }

    # -----------------------------------------------------------------------
    # Step-by-step helpers
    # -----------------------------------------------------------------------

    def _addition_steps(self, a: str, b: str, result: str) -> list[str]:
        """Generate columnar addition steps string list."""
        width = max(len(a), len(b), len(result))
        a_padded = a.zfill(width)
        b_padded = b.zfill(width)
        r_padded = result.zfill(width)

        steps = [
            "Binary Addition Steps:",
            f"  {'':>{width - len(a_padded)}}  {a_padded}",
            f"+ {'':>{width - len(b_padded)}}  {b_padded}",
            "  " + "-" * (width + 2),
            f"  {'':>{width - len(r_padded)}}  {r_padded}",
            "",
            "Carry propagation (right to left):",
        ]

        carry = 0
        carries = []
        a_r = a_padded[::-1]
        b_r = b_padded[::-1]
        for i in range(width):
            bit_sum = int(a_r[i]) + int(b_r[i]) + carry
            carries.append(carry)
            carry = bit_sum >> 1

        carries_str = "".join(str(c) for c in reversed(carries))
        steps.append(f"  Carries: {carries_str}")
        return steps

    # -----------------------------------------------------------------------
    # Format helpers
    # -----------------------------------------------------------------------

    def format_conversion(self, value: int) -> dict:
        """Return a dict with decimal, binary, hex, and octal representations."""
        if value < 0:
            raise ValueError("Use format_signed_conversion for negative numbers.")
        return {
            "decimal": value,
            "binary": self.decimal_to_binary(value),
            "hexadecimal": hex(value)[2:].upper(),
            "octal": oct(value)[2:],
        }
