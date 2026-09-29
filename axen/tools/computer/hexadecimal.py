"""
axen/tools/computer/hexadecimal.py
=====================================
Deterministic hexadecimal operations for AXEN.

Supports:
  - Decimal ↔ Hexadecimal
  - Binary ↔ Hexadecimal
  - Hex arithmetic (add, subtract, multiply, divide)
  - Hex AND, OR, XOR, NOT operations
  - Hex nibble breakdown (4-bit groups)

All operations use Python integer arithmetic — no LLM involvement.
"""

from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_HEX_CHARS = set("0123456789abcdefABCDEF")

def _clean_hex(s: str) -> str:
    """Strip whitespace, '0x'/'0X' prefix, and '#' prefix."""
    s = s.strip().lstrip("#")
    if s.startswith("0x") or s.startswith("0X"):
        s = s[2:]
    return s


def _is_valid_hex(s: str) -> bool:
    return bool(s) and all(c in _HEX_CHARS for c in s)


# ---------------------------------------------------------------------------
# HexConverter
# ---------------------------------------------------------------------------

class HexConverter:
    """
    Deterministic hexadecimal conversion and arithmetic engine.

    Usage::
        hc = HexConverter()
        hc.decimal_to_hex(255)     → "FF"
        hc.hex_to_decimal("FF")    → 255
        hc.hex_add("1A", "0F")     → {"result": "29", ...}
    """

    # -----------------------------------------------------------------------
    # Conversions
    # -----------------------------------------------------------------------

    def decimal_to_hex(self, value: int) -> str:
        """Convert a non-negative integer to uppercase hex string (no prefix)."""
        if not isinstance(value, int):
            raise TypeError(f"Expected int, got {type(value).__name__}")
        if value < 0:
            raise ValueError("Use signed hex representation for negative numbers.")
        return hex(value)[2:].upper()

    def hex_to_decimal(self, hex_str: str) -> int:
        """Convert a hex string to decimal integer."""
        h = _clean_hex(hex_str)
        if not _is_valid_hex(h):
            raise ValueError(f"Invalid hexadecimal string: '{hex_str}'")
        return int(h, 16)

    def binary_to_hex(self, binary_str: str) -> str:
        """Convert binary string to hex (no prefix)."""
        b = binary_str.strip().lstrip("0b").lstrip("0B")
        if not all(c in "01" for c in b):
            raise ValueError(f"Invalid binary string: '{binary_str}'")
        return hex(int(b, 2))[2:].upper()

    def hex_to_binary(self, hex_str: str) -> str:
        """Convert hex string to binary string (no prefix)."""
        h = _clean_hex(hex_str)
        if not _is_valid_hex(h):
            raise ValueError(f"Invalid hexadecimal string: '{hex_str}'")
        return bin(int(h, 16))[2:]

    def hex_to_octal(self, hex_str: str) -> str:
        """Convert hex string to octal string."""
        h = _clean_hex(hex_str)
        if not _is_valid_hex(h):
            raise ValueError(f"Invalid hexadecimal string: '{hex_str}'")
        return oct(int(h, 16))[2:]

    def nibble_breakdown(self, hex_str: str) -> list[dict]:
        """
        Break a hex string into nibbles (4-bit groups) with their decimal values.

        Returns list of dicts: [{"nibble": "F", "binary": "1111", "decimal": 15}, ...]
        """
        h = _clean_hex(hex_str)
        if not _is_valid_hex(h):
            raise ValueError(f"Invalid hexadecimal string: '{hex_str}'")
        result = []
        for char in h.upper():
            val = int(char, 16)
            result.append({
                "nibble": char,
                "binary": bin(val)[2:].zfill(4),
                "decimal": val,
            })
        return result

    # -----------------------------------------------------------------------
    # Hex Arithmetic
    # -----------------------------------------------------------------------

    def hex_add(self, a: str, b: str) -> dict:
        """Add two hexadecimal numbers."""
        h_a = _clean_hex(a)
        h_b = _clean_hex(b)
        for name, val in (("a", h_a), ("b", h_b)):
            if not _is_valid_hex(val):
                raise ValueError(f"Invalid hex operand '{name}': '{val}'")
        da, db = int(h_a, 16), int(h_b, 16)
        dr = da + db
        return {
            "operation": "addition",
            "a": "0x" + h_a.upper(),
            "b": "0x" + h_b.upper(),
            "result": "0x" + hex(dr)[2:].upper(),
            "decimal_result": dr,
        }

    def hex_subtract(self, a: str, b: str) -> dict:
        """Subtract hex b from hex a."""
        h_a = _clean_hex(a)
        h_b = _clean_hex(b)
        for name, val in (("a", h_a), ("b", h_b)):
            if not _is_valid_hex(val):
                raise ValueError(f"Invalid hex operand '{name}': '{val}'")
        da, db = int(h_a, 16), int(h_b, 16)
        dr = da - db
        negative = dr < 0
        result = "0x" + hex(abs(dr))[2:].upper()
        if negative:
            result = "-" + result
        return {
            "operation": "subtraction",
            "a": "0x" + h_a.upper(),
            "b": "0x" + h_b.upper(),
            "result": result,
            "decimal_result": dr,
            "is_negative": negative,
        }

    def hex_multiply(self, a: str, b: str) -> dict:
        """Multiply two hexadecimal numbers."""
        h_a = _clean_hex(a)
        h_b = _clean_hex(b)
        for name, val in (("a", h_a), ("b", h_b)):
            if not _is_valid_hex(val):
                raise ValueError(f"Invalid hex operand '{name}': '{val}'")
        da, db = int(h_a, 16), int(h_b, 16)
        dr = da * db
        return {
            "operation": "multiplication",
            "a": "0x" + h_a.upper(),
            "b": "0x" + h_b.upper(),
            "result": "0x" + hex(dr)[2:].upper(),
            "decimal_result": dr,
        }

    def hex_divide(self, a: str, b: str) -> dict:
        """Integer divide hex a by hex b."""
        h_a = _clean_hex(a)
        h_b = _clean_hex(b)
        for name, val in (("a", h_a), ("b", h_b)):
            if not _is_valid_hex(val):
                raise ValueError(f"Invalid hex operand '{name}': '{val}'")
        da, db = int(h_a, 16), int(h_b, 16)
        if db == 0:
            raise ZeroDivisionError("Hex division by zero.")
        quotient = da // db
        remainder = da % db
        return {
            "operation": "division",
            "a": "0x" + h_a.upper(),
            "b": "0x" + h_b.upper(),
            "quotient": "0x" + hex(quotient)[2:].upper(),
            "remainder": "0x" + hex(remainder)[2:].upper(),
            "decimal_quotient": quotient,
            "decimal_remainder": remainder,
        }

    # -----------------------------------------------------------------------
    # Hex Bitwise
    # -----------------------------------------------------------------------

    def hex_and(self, a: str, b: str) -> dict:
        """Bitwise AND of two hex numbers."""
        h_a, h_b = _clean_hex(a), _clean_hex(b)
        da, db = int(h_a, 16), int(h_b, 16)
        dr = da & db
        return {"operation": "AND", "a": "0x" + h_a.upper(), "b": "0x" + h_b.upper(),
                "result": "0x" + hex(dr)[2:].upper(), "decimal_result": dr}

    def hex_or(self, a: str, b: str) -> dict:
        """Bitwise OR of two hex numbers."""
        h_a, h_b = _clean_hex(a), _clean_hex(b)
        da, db = int(h_a, 16), int(h_b, 16)
        dr = da | db
        return {"operation": "OR", "a": "0x" + h_a.upper(), "b": "0x" + h_b.upper(),
                "result": "0x" + hex(dr)[2:].upper(), "decimal_result": dr}

    def hex_xor(self, a: str, b: str) -> dict:
        """Bitwise XOR of two hex numbers."""
        h_a, h_b = _clean_hex(a), _clean_hex(b)
        da, db = int(h_a, 16), int(h_b, 16)
        dr = da ^ db
        return {"operation": "XOR", "a": "0x" + h_a.upper(), "b": "0x" + h_b.upper(),
                "result": "0x" + hex(dr)[2:].upper(), "decimal_result": dr}

    def hex_not(self, a: str, bits: int = 8) -> dict:
        """Bitwise NOT of a hex number within the given bit width."""
        if bits not in (4, 8, 16, 32, 64):
            raise ValueError(f"Unsupported bit width {bits}. Choose from 4, 8, 16, 32, 64.")
        h_a = _clean_hex(a)
        da = int(h_a, 16)
        mask = (1 << bits) - 1
        dr = (~da) & mask
        return {"operation": f"NOT ({bits}-bit)", "a": "0x" + h_a.upper(),
                "result": "0x" + hex(dr)[2:].upper().zfill(bits // 4), "bit_width": bits,
                "decimal_result": dr}

    # -----------------------------------------------------------------------
    # Display helpers
    # -----------------------------------------------------------------------

    def format_hex_table(self, value: int) -> dict:
        """Return a complete base-representation table for value."""
        return {
            "decimal": value,
            "hexadecimal": "0x" + hex(value)[2:].upper(),
            "binary": bin(value)[2:],
            "octal": oct(value)[2:],
            "nibbles": self.nibble_breakdown(hex(value)[2:]),
        }
