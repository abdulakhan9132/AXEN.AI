"""
axen/tools/computer/ascii_tool.py
===================================
Deterministic ASCII and Unicode encoding for AXEN.

Supports:
  - Character → decimal (ASCII code point)
  - Character → binary (8-bit)
  - Character → hexadecimal
  - Decimal → character
  - String → binary (character-by-character)
  - String → hex encoding
  - Basic Unicode support (code points beyond ASCII 127)
  - Printable / non-printable detection

All operations are deterministic Python — no LLM involvement.
"""

from __future__ import annotations

from typing import Union


# ---------------------------------------------------------------------------
# AsciiTool
# ---------------------------------------------------------------------------

class AsciiTool:
    """
    ASCII and Unicode character encoding tool.

    Usage::
        at = AsciiTool()
        at.char_to_info("A")             → {"char": "A", "decimal": 65, ...}
        at.string_to_binary("Hi")        → [{"char": "H", ...}, {"char": "i", ...}]
        at.decimal_to_char(65)           → "A"
    """

    # -----------------------------------------------------------------------
    # Single character operations
    # -----------------------------------------------------------------------

    def char_to_info(self, char: str) -> dict:
        """
        Return full encoding information for a single character.

        Returns dict with:
          char, decimal, binary (8-bit for ASCII), hexadecimal, is_ascii,
          is_printable, unicode_name (if available), category.
        """
        if len(char) != 1:
            raise ValueError(f"Expected a single character, got: {repr(char)}")

        code_point = ord(char)
        is_ascii = code_point <= 127
        # Binary: use 8-bit for ASCII, extend for wider Unicode
        if is_ascii:
            binary_str = bin(code_point)[2:].zfill(8)
        else:
            # Use enough bits for the code point
            bits_needed = code_point.bit_length()
            width = max(8, ((bits_needed + 7) // 8) * 8)  # round up to byte
            binary_str = bin(code_point)[2:].zfill(width)

        hex_str = hex(code_point)[2:].upper().zfill(2 if is_ascii else 4)

        # Unicode name
        import unicodedata
        try:
            unicode_name = unicodedata.name(char)
        except ValueError:
            unicode_name = "(unnamed)"

        try:
            category = unicodedata.category(char)
        except Exception:
            category = "unknown"

        return {
            "char": char,
            "decimal": code_point,
            "binary": binary_str,
            "hexadecimal": hex_str,
            "is_ascii": is_ascii,
            "is_printable": char.isprintable(),
            "unicode_name": unicode_name,
            "unicode_category": category,
            "note": (
                f"ASCII printable range: 32–126. "
                f"'{char}' is {'printable' if char.isprintable() else 'non-printable'}."
            ),
        }

    def decimal_to_char(self, code_point: int) -> dict:
        """Convert a Unicode code point (decimal) to its character and full info."""
        if not isinstance(code_point, int) or code_point < 0 or code_point > 0x10FFFF:
            raise ValueError(
                f"Code point must be an integer in [0, 1114111], got: {code_point}"
            )
        char = chr(code_point)
        return self.char_to_info(char)

    def hex_to_char(self, hex_str: str) -> dict:
        """Convert a hex code point string to its character and full info."""
        try:
            code_point = int(hex_str.strip().lstrip("0x").lstrip("0X"), 16)
        except ValueError:
            raise ValueError(f"Invalid hex code point: '{hex_str}'")
        return self.decimal_to_char(code_point)

    # -----------------------------------------------------------------------
    # String operations
    # -----------------------------------------------------------------------

    def string_to_binary(self, text: str) -> list[dict]:
        """
        Convert each character in a string to its encoding info.

        Returns a list of char_info dicts, one per character.
        """
        if not text:
            raise ValueError("Input string is empty.")
        return [self.char_to_info(c) for c in text]

    def string_to_binary_compact(self, text: str) -> str:
        """
        Return a compact binary representation of a string.

        Example: "Hi" → "01001000 01101001"
        """
        if not text:
            raise ValueError("Input string is empty.")
        parts = []
        for c in text:
            code = ord(c)
            if code <= 127:
                parts.append(bin(code)[2:].zfill(8))
            else:
                bits = code.bit_length()
                width = ((bits + 7) // 8) * 8
                parts.append(bin(code)[2:].zfill(width))
        return " ".join(parts)

    def string_to_hex(self, text: str) -> str:
        """
        Return a hex representation of a string (space-separated bytes for ASCII).

        Example: "Hi" → "48 69"
        """
        if not text:
            raise ValueError("Input string is empty.")
        return " ".join(hex(ord(c))[2:].upper().zfill(2) for c in text)

    def binary_to_string(self, binary_str: str) -> str:
        """
        Convert a binary string (space-separated bytes) back to a text string.

        Example: "01001000 01101001" → "Hi"
        """
        groups = binary_str.strip().split()
        chars = []
        for group in groups:
            if not all(c in "01" for c in group):
                raise ValueError(f"Invalid binary group: '{group}'")
            chars.append(chr(int(group, 2)))
        return "".join(chars)

    # -----------------------------------------------------------------------
    # ASCII table helpers
    # -----------------------------------------------------------------------

    def ascii_table_range(self, start: int = 32, end: int = 126) -> list[dict]:
        """
        Return ASCII table entries for code points in [start, end].

        Default range (32–126) covers all standard printable ASCII characters.
        """
        if start < 0 or end > 127 or start > end:
            raise ValueError("Range must be within [0, 127] with start <= end.")
        return [self.char_to_info(chr(i)) for i in range(start, end + 1)]

    def is_printable_ascii(self, char: str) -> bool:
        """Return True if char is in the printable ASCII range (32–126)."""
        if len(char) != 1:
            raise ValueError("Expected a single character.")
        return 32 <= ord(char) <= 126

    # -----------------------------------------------------------------------
    # Control character descriptions
    # -----------------------------------------------------------------------

    _CONTROL_CHARS: dict[int, str] = {
        0: "NUL (Null)", 1: "SOH (Start of Heading)", 2: "STX (Start of Text)",
        3: "ETX (End of Text)", 4: "EOT (End of Transmission)", 5: "ENQ (Enquiry)",
        6: "ACK (Acknowledge)", 7: "BEL (Bell)", 8: "BS (Backspace)",
        9: "HT (Horizontal Tab)", 10: "LF (Line Feed)", 11: "VT (Vertical Tab)",
        12: "FF (Form Feed)", 13: "CR (Carriage Return)", 14: "SO (Shift Out)",
        15: "SI (Shift In)", 27: "ESC (Escape)", 127: "DEL (Delete)",
    }

    def describe_control_char(self, code_point: int) -> str:
        """Return a human-readable name for an ASCII control character."""
        return self._CONTROL_CHARS.get(code_point, f"ASCII {code_point}")
