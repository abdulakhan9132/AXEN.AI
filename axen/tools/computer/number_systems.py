"""
axen/tools/computer/number_systems.py
========================================
Unified number system converter for AXEN.

Provides a single high-level API that converts between:
  - Decimal (base 10)
  - Binary  (base 2)
  - Hexadecimal (base 16)
  - Octal (base 8)

And provides educational context about each number system.
"""

from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Number system metadata
# ---------------------------------------------------------------------------

NUMBER_SYSTEM_INFO = {
    "decimal": {
        "name": "Decimal (Base-10)",
        "digits": "0–9",
        "radix": 10,
        "use": "Everyday human arithmetic.",
        "example": "42",
    },
    "binary": {
        "name": "Binary (Base-2)",
        "digits": "0, 1",
        "radix": 2,
        "use": "Direct representation of digital signals (off/on). Used internally by all CPUs and memory.",
        "example": "101010",
    },
    "hexadecimal": {
        "name": "Hexadecimal (Base-16)",
        "digits": "0–9, A–F",
        "radix": 16,
        "use": "Compact representation of binary data. Used for memory addresses, colors, opcodes, and machine code.",
        "example": "2A",
    },
    "octal": {
        "name": "Octal (Base-8)",
        "digits": "0–7",
        "radix": 8,
        "use": "Historically used in Unix file permissions and older computing systems.",
        "example": "52",
    },
}


# ---------------------------------------------------------------------------
# NumberSystemConverter
# ---------------------------------------------------------------------------

class NumberSystemConverter:
    """
    High-level number system conversion hub.

    Usage::
        ns = NumberSystemConverter()
        ns.convert(42, from_base="decimal")
        # → {"decimal": 42, "binary": "101010", "hexadecimal": "2A", "octal": "52"}

        ns.convert("2A", from_base="hexadecimal")
        # → {"decimal": 42, "binary": "101010", "hexadecimal": "2A", "octal": "52"}
    """

    def convert(self, value: str | int, from_base: str = "decimal") -> dict:
        """
        Convert 'value' from 'from_base' to all supported number systems.

        from_base must be one of:
          "decimal" | "binary" | "hexadecimal" | "hex" | "octal"

        Returns a dict with keys: decimal, binary, hexadecimal, octal, plus
        the identified source base and any educational notes.
        """
        base = from_base.lower().strip()
        if base in ("hex",):
            base = "hexadecimal"

        if base == "decimal":
            n = self._parse_decimal(value)
        elif base == "binary":
            n = self._parse_binary(str(value))
        elif base == "hexadecimal":
            n = self._parse_hex(str(value))
        elif base == "octal":
            n = self._parse_octal(str(value))
        else:
            raise ValueError(
                f"Unknown base: '{from_base}'. "
                "Valid options: decimal, binary, hexadecimal, hex, octal"
            )

        if n < 0:
            raise ValueError("Use IntegerRepresentation for negative numbers (two's complement).")

        return {
            "source_base": base,
            "source_value": str(value),
            "decimal": n,
            "binary": bin(n)[2:],
            "hexadecimal": hex(n)[2:].upper(),
            "octal": oct(n)[2:],
            "bits_required": max(n.bit_length(), 1),
            "bytes_required": max((n.bit_length() + 7) // 8, 1),
        }

    def detect_base(self, value: str) -> str:
        """
        Attempt to detect the likely number base from the string.

        Heuristics:
          - Starts with "0x" or "0X" → hexadecimal
          - Starts with "0b" or "0B" → binary
          - Starts with "0o" or "0O" → octal
          - Contains A-F (case-insensitive) → hexadecimal
          - Contains only 0–7 → possibly octal or decimal (ambiguous, returns "ambiguous")
          - Contains only 0s and 1s → possibly binary (returns "binary_or_decimal")
          - Otherwise → decimal
        """
        s = value.strip()
        if s.startswith(("0x", "0X")):
            return "hexadecimal"
        if s.startswith(("0b", "0B")):
            return "binary"
        if s.startswith(("0o", "0O")):
            return "octal"
        upper = s.upper()
        if re.search(r"[A-F]", upper):
            return "hexadecimal"
        if re.match(r"^[01]+$", s):
            return "binary_or_decimal"  # ambiguous — 10 could be binary (2) or decimal (10)
        if re.match(r"^[0-7]+$", s):
            return "octal_or_decimal"   # ambiguous
        if re.match(r"^\d+$", s):
            return "decimal"
        return "unknown"

    def explain_system(self, system: str) -> dict:
        """Return educational information about a number system."""
        key = system.lower().strip()
        if key in ("hex",):
            key = "hexadecimal"
        info = NUMBER_SYSTEM_INFO.get(key)
        if not info:
            raise ValueError(
                f"Unknown number system: '{system}'. "
                "Choose from: decimal, binary, hexadecimal, octal."
            )
        return info

    def positional_breakdown(self, value: str, base: str) -> list[dict]:
        """
        Show positional value of each digit.

        Example for "101010" in base 2:
          bit 5 (value 32): 1 × 32 = 32
          bit 4 (value 16): 0 × 16 = 0
          ...
        Returns list of dicts with: position, digit, place_value, contribution.
        """
        base_map = {
            "decimal": 10, "binary": 2, "hexadecimal": 16, "hex": 16, "octal": 8
        }
        radix = base_map.get(base.lower())
        if radix is None:
            raise ValueError(f"Unknown base: '{base}'")

        clean = value.strip()
        # Strip prefix
        if clean.startswith(("0x", "0X", "0b", "0B", "0o", "0O")):
            clean = clean[2:]

        breakdown = []
        n = len(clean)
        for i, digit_char in enumerate(clean):
            position = n - 1 - i
            digit_val = int(digit_char, radix)
            place_value = radix ** position
            contribution = digit_val * place_value
            breakdown.append({
                "position": position,
                "digit": digit_char.upper(),
                "place_value": place_value,
                "digit_decimal": digit_val,
                "contribution": contribution,
            })

        total = sum(d["contribution"] for d in breakdown)
        return breakdown

    # -----------------------------------------------------------------------
    # Internal parsers
    # -----------------------------------------------------------------------

    def _parse_decimal(self, value: str | int) -> int:
        try:
            return int(str(value).strip())
        except ValueError:
            raise ValueError(f"Invalid decimal value: '{value}'")

    def _parse_binary(self, value: str) -> int:
        s = value.strip().lstrip("0b").lstrip("0B")
        if not s or not all(c in "01" for c in s):
            raise ValueError(f"Invalid binary value: '{value}'")
        return int(s, 2)

    def _parse_hex(self, value: str) -> int:
        s = value.strip().lstrip("0x").lstrip("0X")
        try:
            return int(s, 16)
        except ValueError:
            raise ValueError(f"Invalid hexadecimal value: '{value}'")

    def _parse_octal(self, value: str) -> int:
        s = value.strip().lstrip("0o").lstrip("0O")
        try:
            return int(s, 8)
        except ValueError:
            raise ValueError(f"Invalid octal value: '{value}'")
