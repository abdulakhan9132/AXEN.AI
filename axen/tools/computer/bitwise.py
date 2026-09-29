"""
axen/tools/computer/bitwise.py
================================
Deterministic bitwise operations for AXEN Computer Systems module.

Supports:
  - AND, OR, XOR, NOT (with explicit bit width)
  - Left shift (<<), Right shift (>>)
  - Set bit, Clear bit, Toggle bit, Check bit
  - Extract bits (range extraction)
  - Mask bits
  - Visual binary representation

IMPORTANT on NOT:
  NOT requires an explicit bit width.  Without specifying the width,
  NOT is mathematically ambiguous (depends on word size).
  AXEN always requires the user to specify or defaults to 8-bit.
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_SUPPORTED_WIDTHS = (4, 8, 16, 32, 64)


def _validate_bits(bits: int) -> None:
    if bits not in _SUPPORTED_WIDTHS:
        raise ValueError(
            f"Bit width {bits} not supported. Choose from {_SUPPORTED_WIDTHS}."
        )


def _mask_for(bits: int) -> int:
    return (1 << bits) - 1


# ---------------------------------------------------------------------------
# BitwiseOps
# ---------------------------------------------------------------------------

class BitwiseOps:
    """
    Deterministic bitwise operations engine.

    All results include: the operation name, decimal operands, binary
    representations, the integer result, and a binary string of the result.

    Usage::
        bw = BitwiseOps()
        bw.AND(0b1010, 0b1100)               → {..., "result_bin": "1000"}
        bw.NOT(0b1010, bits=4)               → {..., "result_bin": "0101"}
        bw.left_shift(0b0001, 3)             → {..., "result_bin": "1000"}
        bw.set_bit(0b1000, bit_position=2)   → {..., "result_bin": "1100"}
    """

    # -----------------------------------------------------------------------
    # Core bitwise gates
    # -----------------------------------------------------------------------

    def AND(self, a: int, b: int) -> dict:
        result = a & b
        return self._format("AND", a, b, result)

    def OR(self, a: int, b: int) -> dict:
        result = a | b
        return self._format("OR", a, b, result)

    def XOR(self, a: int, b: int) -> dict:
        result = a ^ b
        return self._format("XOR", a, b, result)

    def NOT(self, a: int, bits: int = 8) -> dict:
        """
        Bitwise NOT with explicit bit width.

        Without a bit width, NOT is undefined because Python integers
        are arbitrary-precision (NOT of 1010 in Python gives a negative number).
        We mask to 'bits' width.
        """
        _validate_bits(bits)
        mask = _mask_for(bits)
        result = (~a) & mask
        width = bits
        return {
            "operation": f"NOT ({bits}-bit)",
            "a_decimal": a,
            "a_binary": bin(a)[2:].zfill(bits),
            "bit_width": bits,
            "result_decimal": result,
            "result_bin": bin(result)[2:].zfill(bits),
            "note": f"NOT is computed within {bits}-bit word size. Mask = {bin(mask)[2:]}"
        }

    # -----------------------------------------------------------------------
    # Shift operations
    # -----------------------------------------------------------------------

    def left_shift(self, value: int, n: int, bits: Optional[int] = None) -> dict:
        """
        Left shift 'value' by 'n' positions.

        If 'bits' is provided, the result is masked to that bit width
        (simulating fixed-width register overflow).
        """
        if n < 0:
            raise ValueError("Shift amount must be non-negative.")
        result = value << n
        if bits is not None:
            _validate_bits(bits)
            result = result & _mask_for(bits)
            width_note = f"(masked to {bits} bits)"
        else:
            bits = max(value.bit_length() + n, 1)
            width_note = "(unbounded)"

        return {
            "operation": f"LEFT SHIFT by {n}",
            "value_decimal": value,
            "value_binary": bin(value)[2:],
            "shift_amount": n,
            "result_decimal": result,
            "result_bin": bin(result)[2:],
            "note": f"Equivalent to multiplying by 2^{n} = {2**n} {width_note}",
        }

    def right_shift(self, value: int, n: int, bits: Optional[int] = None) -> dict:
        """
        Logical right shift 'value' by 'n' positions.

        If 'bits' is provided, treats as unsigned fixed-width integer.
        """
        if n < 0:
            raise ValueError("Shift amount must be non-negative.")
        if bits is not None:
            _validate_bits(bits)
            # Mask first to treat as unsigned
            value = value & _mask_for(bits)
            width_note = f"(unsigned {bits}-bit)"
        else:
            width_note = "(unbounded)"

        result = value >> n
        return {
            "operation": f"RIGHT SHIFT by {n}",
            "value_decimal": value,
            "value_binary": bin(value)[2:],
            "shift_amount": n,
            "result_decimal": result,
            "result_bin": bin(result)[2:],
            "note": f"Equivalent to dividing by 2^{n} = {2**n} (integer) {width_note}",
        }

    # -----------------------------------------------------------------------
    # Bit manipulation
    # -----------------------------------------------------------------------

    def set_bit(self, value: int, bit_position: int) -> dict:
        """Set (force to 1) the bit at bit_position (0 = LSB)."""
        if bit_position < 0:
            raise ValueError("Bit position must be >= 0.")
        result = value | (1 << bit_position)
        return {
            "operation": f"SET bit {bit_position}",
            "value_decimal": value,
            "value_binary": bin(value)[2:],
            "bit_position": bit_position,
            "result_decimal": result,
            "result_bin": bin(result)[2:],
            "note": f"OR with mask 1<<{bit_position} = {bin(1 << bit_position)[2:]}"
        }

    def clear_bit(self, value: int, bit_position: int) -> dict:
        """Clear (force to 0) the bit at bit_position (0 = LSB)."""
        if bit_position < 0:
            raise ValueError("Bit position must be >= 0.")
        result = value & ~(1 << bit_position)
        return {
            "operation": f"CLEAR bit {bit_position}",
            "value_decimal": value,
            "value_binary": bin(value)[2:],
            "bit_position": bit_position,
            "result_decimal": result,
            "result_bin": bin(result)[2:],
            "note": f"AND with NOT of mask 1<<{bit_position}"
        }

    def toggle_bit(self, value: int, bit_position: int) -> dict:
        """Toggle (flip) the bit at bit_position (0 = LSB)."""
        if bit_position < 0:
            raise ValueError("Bit position must be >= 0.")
        result = value ^ (1 << bit_position)
        return {
            "operation": f"TOGGLE bit {bit_position}",
            "value_decimal": value,
            "value_binary": bin(value)[2:],
            "bit_position": bit_position,
            "result_decimal": result,
            "result_bin": bin(result)[2:],
            "note": f"XOR with mask 1<<{bit_position} = {bin(1 << bit_position)[2:]}"
        }

    def check_bit(self, value: int, bit_position: int) -> dict:
        """Check whether the bit at bit_position (0 = LSB) is set."""
        if bit_position < 0:
            raise ValueError("Bit position must be >= 0.")
        is_set = bool(value & (1 << bit_position))
        return {
            "operation": f"CHECK bit {bit_position}",
            "value_decimal": value,
            "value_binary": bin(value)[2:],
            "bit_position": bit_position,
            "is_set": is_set,
            "note": f"Bit {bit_position} is {'SET (1)' if is_set else 'CLEAR (0)'}",
        }

    def extract_bits(self, value: int, high: int, low: int) -> dict:
        """
        Extract a range of bits [high:low] inclusive (0 = LSB).

        Example: extract_bits(0b10110100, 6, 4) → bits 6,5,4 → "011" → 3
        """
        if high < low:
            raise ValueError("high must be >= low.")
        if low < 0:
            raise ValueError("low must be >= 0.")
        width = high - low + 1
        mask = _mask_for(width)
        result = (value >> low) & mask
        return {
            "operation": f"EXTRACT bits [{high}:{low}]",
            "value_decimal": value,
            "value_binary": bin(value)[2:],
            "high_bit": high,
            "low_bit": low,
            "extracted_width": width,
            "result_decimal": result,
            "result_bin": bin(result)[2:].zfill(width),
        }

    def mask_bits(self, value: int, mask: int) -> dict:
        """Apply a bitmask (AND operation) to value."""
        result = value & mask
        return {
            "operation": "MASK",
            "value_decimal": value,
            "value_binary": bin(value)[2:],
            "mask_decimal": mask,
            "mask_binary": bin(mask)[2:],
            "result_decimal": result,
            "result_bin": bin(result)[2:],
            "note": "Result = value AND mask",
        }

    # -----------------------------------------------------------------------
    # Internal format helper
    # -----------------------------------------------------------------------

    def _format(self, op: str, a: int, b: int, result: int) -> dict:
        """Build a standard result dict for binary gate operations."""
        width = max(a.bit_length(), b.bit_length(), result.bit_length(), 1)
        return {
            "operation": op,
            "a_decimal": a,
            "b_decimal": b,
            "a_binary": bin(a)[2:].zfill(width),
            "b_binary": bin(b)[2:].zfill(width),
            "result_decimal": result,
            "result_bin": bin(result)[2:].zfill(width),
        }

    # -----------------------------------------------------------------------
    # Convenience: parse binary strings and run operation
    # -----------------------------------------------------------------------

    def from_binary_strings(self, op: str, a_str: str, b_str: Optional[str] = None,
                             bits: int = 8) -> dict:
        """
        Accept binary strings and dispatch to the appropriate gate operation.

        op must be one of: "AND", "OR", "XOR", "NOT"
        """
        a = int(a_str, 2)
        op_upper = op.upper()
        if op_upper == "NOT":
            return self.NOT(a, bits=bits)
        if b_str is None:
            raise ValueError(f"Operation '{op}' requires two operands.")
        b = int(b_str, 2)
        if op_upper == "AND":
            return self.AND(a, b)
        if op_upper == "OR":
            return self.OR(a, b)
        if op_upper == "XOR":
            return self.XOR(a, b)
        raise ValueError(f"Unsupported operation: '{op}'. Use AND, OR, XOR, NOT.")
